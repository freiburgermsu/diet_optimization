#!/usr/bin/env python3
"""Merge all price sources into a single final_prices.json.

Precedence (per search term):
  1. Claude-ranked retailer price (prices_claude.json)
     — best signal; retailer stocked the ingredient and Claude picked a
       clean match
  2. TFP national-average (tfp_price_lookup.csv + CPI inflation)
     — fallback when retailer doesn't stock it or only has processed forms
  3. Nothing — term is dropped from the LP (stderr names the gap)

Output: final_prices.json keyed on search_term with price_per_100g
already CPI-inflated to current dollars. Every entry includes
`price_source` provenance.

Usage:
    uv run python scripts/merge_prices.py \\
        --claude prices_claude.json \\
        --tfp data/tfp_price_lookup.csv \\
        --terms prices_raw.json \\
        --output final_prices.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfp_pricing import inflate_cpi, DEFAULT_BASE, DEFAULT_CURRENT  # noqa: E402


def load_claude_prices(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_tfp_lookup(path: Path) -> dict[str, dict]:
    """Read tfp_price_lookup.csv → {term: {tfp_category, price_2021, ...}}."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("tfp_category"):
                continue  # null categorizations
            term = row["search_term"]
            try:
                price_2021 = float(row["price_per_100g_2021"])
            except (KeyError, ValueError):
                continue
            out[term] = {
                "tfp_category": row["tfp_category"],
                "price_per_100g_2021": price_2021,
                "confidence": row.get("confidence", ""),
                "reason": row.get("reason", ""),
            }
    return out


def load_terms(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return list(data.get("terms", []))


def build_final_prices(
    terms: list[str],
    claude: dict[str, dict],
    tfp: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, int]]:
    """Return (final_prices, counts_by_source)."""
    final: dict[str, dict] = {}
    counts = {"kroger": 0, "tfp": 0, "unpriced": 0}
    inflation_factor = inflate_cpi(1.0)

    for term in terms:
        # Precedence 1: Claude-ranked retailer
        if term in claude:
            entry = dict(claude[term])  # copy
            entry["price_source"] = "kroger"
            final[term] = entry
            counts["kroger"] += 1
            continue

        # Precedence 2: TFP fallback, CPI-inflated
        if term in tfp:
            t = tfp[term]
            final[term] = {
                "price_per_100g": t["price_per_100g_2021"] * inflation_factor,
                "price_source": "tfp",
                "tfp_category": t["tfp_category"],
                "confidence": t["confidence"],
                "reason": t["reason"],
                "inflation_factor": inflation_factor,
                "cpi_base": DEFAULT_BASE,
                "cpi_current": DEFAULT_CURRENT,
            }
            counts["tfp"] += 1
            continue

        # Precedence 3: no source covers it
        counts["unpriced"] += 1

    return final, counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--claude", default="prices_claude.json")
    p.add_argument("--tfp", default="data/tfp_price_lookup.csv")
    p.add_argument("--terms", default="prices_raw.json")
    p.add_argument("--output", default="final_prices.json")
    args = p.parse_args()

    terms = load_terms(Path(args.terms))
    claude = load_claude_prices(Path(args.claude))
    tfp = load_tfp_lookup(Path(args.tfp))

    final, counts = build_final_prices(terms, claude, tfp)
    Path(args.output).write_text(json.dumps(final, indent=2, sort_keys=True))

    total = len(terms)
    priced = counts["kroger"] + counts["tfp"]
    print(f"merged {priced}/{total} priced terms → {args.output}", file=sys.stderr)
    print(f"  kroger (Claude-picked): {counts['kroger']}", file=sys.stderr)
    print(f"  tfp (national-average): {counts['tfp']}", file=sys.stderr)
    print(f"  unpriced (dropped):     {counts['unpriced']}", file=sys.stderr)

    if counts["unpriced"] > 0:
        unpriced_terms = [
            t for t in terms if t not in claude and t not in tfp
        ]
        print("\nFirst 10 unpriced terms:", file=sys.stderr)
        for t in unpriced_terms[:10]:
            print(f"  {t}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
