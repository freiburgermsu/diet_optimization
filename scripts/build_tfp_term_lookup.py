#!/usr/bin/env python3
"""Map LP search terms to TFP price categories using Claude.

Fuzzy-matching short search terms against long TFP category labels
doesn't work — "salmon fillet" shares no tokens with "Seafood". We use
Claude for the semantic categorization: given a search term, pick the
best of the 67 TFP categories, or return null if none fit.

Output: data/tfp_price_lookup.csv keyed on search_term with the TFP
category + its median price. Downstream pricing (tfp_pricing.py)
applies CPI inflation.

Usage:
    export ANTHROPIC_API_KEY=...
    uv run --with "anthropic>=0.40" --with pydantic python \\
        scripts/build_tfp_term_lookup.py --model claude-haiku-4-5

Caches per-term decisions at cache/tfp_categorizer/<slug>.json — re-runs
skip cached terms (same pattern as claude_rank_products.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import anthropic
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(
        f"Missing: {e}. Install: pip install 'anthropic>=0.40' pydantic"
    )


DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_CACHE_DIR = Path("cache/tfp_categorizer")


@dataclass(frozen=True)
class CategoryStat:
    name: str
    median_price_per_100g_2021: float
    n_foods: int
    min_price: float
    max_price: float


def slugify(term: str) -> str:
    s = (term or "").lower().strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_-]+", "", s)
    return s or "_unnamed"


def load_tfp_category_stats(path: Path) -> dict[str, CategoryStat]:
    """Read tfp_prices.csv → {category_name: CategoryStat with median price}."""
    by_cat: dict[str, list[float]] = {}
    with open(path) as f:
        reader = csv.DictReader(l for l in f if not l.startswith("#"))
        for row in reader:
            cat = row["tfp_category"]
            by_cat.setdefault(cat, []).append(float(row["price_per_100g_2021"]))
    return {
        cat: CategoryStat(
            name=cat,
            median_price_per_100g_2021=statistics.median(prices),
            n_foods=len(prices),
            min_price=min(prices),
            max_price=max(prices),
        )
        for cat, prices in by_cat.items()
    }


class CategoryChoice(BaseModel):
    chosen_category: str | None = Field(
        default=None,
        description="Exact name of the best-fitting TFP category, or null",
    )
    reason: str = Field(description="One-sentence explanation")
    confidence: str = Field(description="high / medium / low")


SYSTEM_INSTRUCTIONS = """You map food ingredient names to USDA Thrifty Food Plan 2021 categories for a nutritional diet optimization project.

## Task

Given a search term like "salmon fillet" or "pinto beans" or "agave", pick the single best-fitting TFP category name from the provided list. Return the EXACT category string so downstream code can look up its price.

Return `null` when:
- The food is not meaningfully covered by any TFP category (e.g. exotic items like "bear game meat", "beaver", "abalone" — TFP is a US-mainstream basket)
- The food is a non-food product (e.g. "caribou" when the match would be a coffee brand)
- The food crosses multiple categories in an ambiguous way that picking one would mislead

## Selection rules

1. **Match ingredient type to the most-specific relevant category.** "Chicken breast" → "Poultry, higher nutrient density" (not just "Mixed Dishes - Meat").

2. **Raw ingredients beat prepared-dishes categories.** Prefer a raw/single-ingredient category ("Beans, peas, legumes") over a "Mixed Dishes" category when possible.

3. **Nutrient-density categories.** TFP splits categories by "higher" vs "lower" nutrient density. Rule of thumb:
   - "Higher nutrient density" = plainer/rawer form (e.g., plain chicken breast, lean fish, whole grains)
   - "Lower nutrient density" = more processed/fattier (e.g., fried chicken, fatty cuts, refined grains)
   When a search term is a plain ingredient, pick the "higher nutrient density" variant.

4. **Seafood covers fish and shellfish broadly.** Salmon, cod, tuna, shrimp, crab, lobster, clams, mussels all → "Seafood".

5. **Meat categories.**
   - Beef/pork/lamb cuts → "Meats, red, lean" (higher nutrient) or "Meats, red, higher fat" (for fattier cuts)
   - Poultry → "Poultry, higher nutrient density" (usually fine for raw cuts) or "Poultry, lower nutrient density" (prepared)
   - Cured/processed meats (bacon, sausage, deli meat) → "Cured meat"

6. **Dairy.** Whole milk, plain yogurt → "Milk, yogurt, higher nutrient density for all ages". Cheeses → "Cheese".

7. **Vegetables.** TFP has ~6 vegetable categories by color/type. Pick the most specific:
   - "Vegetables - Dark green leafy" (spinach, kale, arugula, collards)
   - "Vegetables - Red/Orange" (carrots, peppers, squash, tomatoes)
   - "Vegetables - Beans/Peas" (green beans, snap peas)
   - "Vegetables - Starchy" (potatoes, corn, plantain)
   - "Other vegetables + vegetable combinations"

8. **Fruits.** Most fresh fruits → "Whole Fruit, higher nutrient density" or similar. Fruit juices go elsewhere.

9. **Grains.**
   - Whole grains (brown rice, oats, quinoa, whole wheat) → "Staple grains - higher nutrient density"
   - Refined (white rice, white flour) → "Staple grains - lower nutrient density"

10. **Return null examples:**
    - Exotic/game items TFP doesn't model (beaver, bear, antelope, bison rarely sold)
    - Brand-name collisions (the "caribou" that's actually a coffee brand)
    - Non-US-mainstream items (abiyuch, burdock, yautia, ataulfo mango if no matching fruit category fits)

## Output

JSON matching:
```json
{"chosen_category": "<exact name from the list>" or null,
 "reason": "one sentence",
 "confidence": "high|medium|low"}
```

## Examples

Search term: "salmon fillet"
→ `{"chosen_category": "Seafood", "reason": "Salmon is a fish; fresh fillets fit the Seafood category", "confidence": "high"}`

Search term: "pinto beans"
→ `{"chosen_category": "Beans, peas, legumes", "reason": "Dry legumes map directly to the beans category", "confidence": "high"}`

Search term: "agave"
→ `{"chosen_category": "Sugars, sweets, and beverages", "reason": "Agave is typically consumed as a sweetener; the closest TFP category is sugars/sweets", "confidence": "medium"}`

Search term: "arrowroot"
→ `{"chosen_category": "Staple grains - higher nutrient density", "reason": "Arrowroot starch is a whole-grain-type starch; cooking behavior similar to other whole starches", "confidence": "medium"}`

Search term: "beaver game meat"
→ `{"chosen_category": null, "reason": "TFP does not include game meats; no national-average price available", "confidence": "high"}`

Search term: "caribou"
→ `{"chosen_category": null, "reason": "Caribou game meat not in TFP categories; unrelated coffee brand matches would mislead", "confidence": "high"}`

Search term: "adzuki beans"
→ `{"chosen_category": "Beans, peas, legumes", "reason": "Adzuki are beans; general legume category provides the national-average price", "confidence": "high"}`

Search term: "arugula lettuce"
→ `{"chosen_category": "Vegetables - Dark green leafy", "reason": "Arugula is a dark leafy green", "confidence": "high"}`

Search term: "atlantic cod"
→ `{"chosen_category": "Seafood", "reason": "Atlantic cod is a common fish; Seafood category covers it", "confidence": "high"}`

Search term: "bottom sirloin beef"
→ `{"chosen_category": "Meats, red, lean", "reason": "Sirloin is a lean beef cut", "confidence": "high"}`

Search term: "baking chocolate"
→ `{"chosen_category": "Sugars, sweets, and beverages", "reason": "Unsweetened baking chocolate still fits confection category best", "confidence": "low"}`

Search term: "flax seeds"
→ `{"chosen_category": "Nuts, seeds, and soy products", "reason": "Flax seeds fall under seeds", "confidence": "high"}`

Search term: "yogurt"
→ `{"chosen_category": "Milk, yogurt, higher nutrient density for all ages", "reason": "Plain yogurt maps directly", "confidence": "high"}`

Search term: "brown rice"
→ `{"chosen_category": "Staple grains - higher nutrient density", "reason": "Brown rice is a whole grain", "confidence": "high"}`

Search term: "parsley"
→ `{"chosen_category": "Vegetables - Dark green leafy", "reason": "Parsley is a leafy herb / dark green; no separate herb category in TFP", "confidence": "medium"}`

Search term: "eggs"
→ `{"chosen_category": "Eggs", "reason": "Direct match", "confidence": "high"}`

Search term: "raw honey"
→ `{"chosen_category": "Sugars, sweets, and beverages", "reason": "Honey is a sweetener", "confidence": "high"}`

Search term: "tofu"
→ `{"chosen_category": "Nuts, seeds, and soy products", "reason": "Tofu is a soy product", "confidence": "high"}`

Search term: "anchovy fish"
→ `{"chosen_category": "Seafood", "reason": "Anchovies are fish", "confidence": "high"}`

Search term: "antelope game meat"
→ `{"chosen_category": null, "reason": "Game meat not represented in TFP mainstream-consumer basket", "confidence": "high"}`

Search term: "wild blueberries"
→ `{"chosen_category": "Whole Fruit, higher nutrient density", "reason": "Berries are whole fruit", "confidence": "high"}`
"""


def build_user_prompt(term: str, categories: list[str]) -> str:
    lines = [f'Search term: "{term}"', "", "TFP categories (pick exactly one, or return null):"]
    for cat in categories:
        lines.append(f"  - {cat}")
    return "\n".join(lines)


def rank_term(
    client: anthropic.Anthropic, model: str, term: str, categories: list[str]
) -> dict:
    user_msg = build_user_prompt(term, categories)
    response = client.messages.parse(
        model=model,
        max_tokens=512,
        system=[{
            "type": "text",
            "text": SYSTEM_INSTRUCTIONS,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_msg}],
        output_format=CategoryChoice,
    )
    choice: CategoryChoice = response.parsed_output
    # Coerce any category Claude returned to the closest exact category in list
    chosen = choice.chosen_category
    if chosen is not None and chosen not in categories:
        chosen = None  # hallucinated category; reject
    return {
        "term": term,
        "choice": {
            **choice.model_dump(),
            "chosen_category": chosen,
        },
        "usage": {
            "cache_read_input_tokens": response.usage.cache_read_input_tokens,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--terms", default="prices_raw.json",
                   help="prices_raw.json or plain text file, one term per line")
    p.add_argument("--tfp-prices", default="data/tfp_prices.csv")
    p.add_argument("--output", default="data/tfp_price_lookup.csv")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only-nulls-from", default=None,
                   help="path to prices_claude.json; only categorize terms that Claude "
                        "ranker evaluated and rejected (had candidates, chose null). "
                        "Terms with zero Kroger hits are NOT included here unless "
                        "--include-zero-result is also set.")
    p.add_argument("--include-zero-result", action="store_true",
                   help="with --only-nulls-from, also categorize terms that had zero "
                        "Kroger hits (never entered the ranker). Expands the set from "
                        "Claude-nulls only to all-unpriced.")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stats = load_tfp_category_stats(Path(args.tfp_prices))
    categories = sorted(stats)
    print(f"loaded {len(categories)} TFP categories", file=sys.stderr)

    # Load search terms
    terms_path = Path(args.terms)
    if terms_path.suffix == ".json":
        raw = json.loads(terms_path.read_text())
        all_terms = list(raw.get("terms", []))
    else:
        all_terms = [l.strip() for l in terms_path.read_text().splitlines() if l.strip()]

    # Optionally filter to only terms Claude ranker evaluated and rejected.
    if args.only_nulls_from:
        priced_claude = set(json.loads(Path(args.only_nulls_from).read_text()))
        # Build the set of terms that actually entered the ranker (had ≥1 Kroger
        # candidate product). Without this filter, 237 zero-result terms would
        # be mixed in — they never saw a ranker decision.
        terms_with_candidates = set()
        if terms_path.suffix == ".json":
            for p in raw.get("products", []):
                t = p.get("search_term")
                if t:
                    terms_with_candidates.add(t)

        if args.include_zero_result:
            terms = [t for t in all_terms if t not in priced_claude]
            scope = "all unpriced terms (Claude nulls + zero-result)"
        else:
            # Default: only terms that had candidates but Claude said null
            terms = [
                t for t in all_terms
                if t in terms_with_candidates and t not in priced_claude
            ]
            scope = "Claude-ranker nulls (had candidates, chose null)"

        print(
            f"filtering to {len(terms)}/{len(all_terms)} terms — {scope}",
            file=sys.stderr,
        )
    else:
        terms = all_terms

    if args.limit:
        terms = terms[: args.limit]

    client = anthropic.Anthropic()
    results: dict[str, dict] = {}
    for i, term in enumerate(terms, start=1):
        cache_file = cache_dir / f"{slugify(term)}.json"
        if cache_file.exists():
            results[term] = json.loads(cache_file.read_text())
            continue
        print(f"[{i}/{len(terms)}] categorizing: {term}", file=sys.stderr)
        try:
            r = rank_term(client, args.model, term, categories)
        except anthropic.APIError as e:
            print(f"  [skip] {term}: {e}", file=sys.stderr)
            continue
        cache_file.write_text(json.dumps(r, indent=2))
        results[term] = r
        cr = r["usage"]["cache_read_input_tokens"]
        print(f"  → {r['choice'].get('chosen_category') or 'null'} "
              f"({r['choice'].get('confidence')}, cache_read={cr})",
              file=sys.stderr)

    # Emit the lookup CSV
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "search_term", "tfp_category", "price_per_100g_2021",
            "confidence", "reason",
        ])
        for term, r in results.items():
            cat = r["choice"].get("chosen_category")
            if cat:
                s = stats[cat]
                w.writerow([
                    term, cat, f"{s.median_price_per_100g_2021:.4f}",
                    r["choice"].get("confidence", ""),
                    r["choice"].get("reason", ""),
                ])
            else:
                w.writerow([
                    term, "", "",
                    r["choice"].get("confidence", "high"),
                    r["choice"].get("reason", ""),
                ])

    priced = sum(1 for r in results.values() if r["choice"].get("chosen_category"))
    print(
        f"\ncategorized {priced}/{len(results)} terms → {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
