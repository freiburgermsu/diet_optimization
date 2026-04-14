#!/usr/bin/env python3
"""Normalize raw retailer prices to $/100g edible, one winner per search term.

Kroger's search returns noisy results — "carrots" matches "Honey Glazed
Carrots with Sage Butter", "Carrots & Celery Snack Tray with Ranch Dip",
etc. Those prices pair incoherently with raw-carrot nutrition data.

Pipeline:
    prices_raw.json (from fetch_prices.py)
        ↓  filter out prepared/flavored/multi-ingredient products
    simple products only
        ↓  pick cheapest $/100g per search term
    one winner per term
        ↓  optional FDC ID join
    prices.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Words that disqualify a product as "raw/plain". Kroger descriptions in
# lowercase. Err on the side of filtering out — a smaller but cleaner
# winner set is more useful than a larger noisy one.
PREPARED_DISQUALIFIERS = {
    "glazed", "seasoned", "flavored", "honey", "butter", "sage",
    "ranch", "dip", "tray", "snack pack", "cup",
    "roasted", "grilled", "fried", "baked",
    "soup", "stew", "salad", "casserole", "medley",
    "with ", " and ", " & ",
    "mix", "blend", "assorted", "combo",
    "pickle", "pickled", "canned", "jar",
    "frozen ",           # exclude frozen prepared; simple "frozen broccoli" is fine
    "candy", "chocolate", "sweetened",
    "marinade", "sauce",
    "puree", "juice", "concentrate",
    "chip", "cracker", "cookie",
    "bar ",
    "fries", "tots", "nugget",
}

# Words that, when present alone, strongly suggest raw/plain. A product
# matching any of these without any disqualifier is a good candidate.
PLAIN_AFFIRMATIVES = {"fresh", "raw", "organic", "whole", "plain"}


def is_simple_product(description: str) -> bool:
    """Return True if the product description looks like a raw/plain item.

    A product is `simple` iff it contains NO disqualifier substring.
    Affirmative words are tracked separately for scoring but don't
    override a disqualifier.
    """
    d = (description or "").lower()
    if not d:
        return False
    for bad in PREPARED_DISQUALIFIERS:
        if bad in d:
            return False
    return True


def description_simplicity_score(description: str) -> float:
    """Higher = simpler. Used to break price ties."""
    d = (description or "").lower()
    word_count = len(re.findall(r"\w+", d))
    # Short descriptions (2-4 words like "Kroger Carrots") score highest
    base = max(0, 10 - word_count)
    for good in PLAIN_AFFIRMATIVES:
        if good in d:
            base += 2
    return float(base)


def price_per_100g_edible(
    package_price: float,
    package_size_g: float,
    yield_factor: float = 1.0,
) -> float:
    """Convert package price + size + yield into $/100g edible."""
    if package_size_g <= 0 or yield_factor <= 0:
        raise ValueError("package_size_g and yield_factor must be positive")
    gross_per_100g = package_price / package_size_g * 100
    return gross_per_100g / yield_factor


def term_words_in_description(term: str, description: str) -> bool:
    """Relevance guard: every word of the search term must appear in description.

    Kroger cross-sells unrelated products (searching "broccoli" returns
    "Cauliflower"). Without this guard, the cheapest cross-sell wins.
    Singular/plural is naively handled by stripping trailing 's'.
    """
    d = (description or "").lower()
    for word in (term or "").lower().split():
        w = word.strip(",.")
        if not w:
            continue
        # Accept plural ↔ singular ("carrots" matches "carrot")
        stem = w[:-1] if w.endswith("s") else w
        if w not in d and stem not in d:
            return False
    return True


def select_winner_per_term(
    products: list[dict],
    filter_simple: bool = True,
    require_term_match: bool = True,
    yields_by_term: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Return {search_term: winning_product_with_price_per_100g}.

    Selection: lowest $/100g edible among simple, on-topic products;
    ties broken by description simplicity.
    """
    yields = yields_by_term or {}
    by_term: dict[str, list[dict]] = {}
    for p in products:
        term = p.get("search_term", "")
        description = p.get("description", "")
        if filter_simple and not is_simple_product(description):
            continue
        if require_term_match and not term_words_in_description(term, description):
            continue
        try:
            ppg = price_per_100g_edible(
                p["price"], p["package_size_g"], yields.get(term, 1.0)
            )
        except (KeyError, ValueError):
            continue
        p = dict(p)  # don't mutate input
        p["price_per_100g"] = ppg
        p["simplicity_score"] = description_simplicity_score(description)
        by_term.setdefault(term, []).append(p)

    winners: dict[str, dict] = {}
    for term, candidates in by_term.items():
        # Primary: cheapest; tie-break: simplest description
        candidates.sort(key=lambda x: (x["price_per_100g"], -x["simplicity_score"]))
        winners[term] = candidates[0]
    return winners


def build_prices_by_term(
    raw: dict,
    filter_simple: bool = True,
    require_term_match: bool = True,
) -> dict[str, dict]:
    """Produce a term-keyed price table (no FDC ID join)."""
    winners = select_winner_per_term(
        raw.get("products", []),
        filter_simple=filter_simple,
        require_term_match=require_term_match,
    )
    out: dict[str, dict] = {}
    fetched_at = raw.get("fetched_at")
    for term, winner in winners.items():
        out[term] = {
            "price_per_100g": winner["price_per_100g"],
            "price_source": raw.get("retailer", "retailer"),
            "fetched_at": fetched_at,
            "raw_description": winner.get("description"),
            "package_size_g": winner.get("package_size_g"),
            "simplicity_score": winner.get("simplicity_score"),
        }
    return out


def diagnose_dropped_terms(raw: dict, winners: dict[str, dict]) -> list[str]:
    """Return terms that had products but no simple winner."""
    terms_with_results: set[str] = {
        p.get("search_term", "") for p in raw.get("products", []) if p.get("search_term")
    }
    return sorted(terms_with_results - set(winners))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="prices_raw.json")
    parser.add_argument("--output", default="prices.json")
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="skip the simple-product filter (for debugging only)",
    )
    parser.add_argument(
        "--no-term-check",
        action="store_true",
        help="skip the relevance check (search-term words must appear in description). "
             "For debugging only — disabling lets cross-sells like 'Cauliflower' "
             "win a 'broccoli' search.",
    )
    args = parser.parse_args()

    raw = json.loads(Path(args.raw).read_text())
    prices = build_prices_by_term(
        raw,
        filter_simple=not args.no_filter,
        require_term_match=not args.no_term_check,
    )

    total_terms = len(raw.get("terms", []))
    dropped = diagnose_dropped_terms(raw, prices)

    Path(args.output).write_text(json.dumps(prices, indent=2))
    print(
        f"wrote {len(prices)}/{total_terms} priced terms to {args.output}",
        file=sys.stderr,
    )
    if dropped:
        print(
            f"{len(dropped)} terms had results but no simple winner — "
            f"consider --no-filter to inspect:",
            file=sys.stderr,
        )
        for t in dropped[:10]:
            print(f"  {t}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
