#!/usr/bin/env python3
"""Join final_prices.json + FDC nutrients into priced_foods.json.

Each search term in final_prices.json came from running
fetch_prices.extract_search_term() over multiple FDC descriptions
(e.g. "pinto beans" ← "Beans, pinto, mature seeds, raw" + sprouted
variant). To assign a nutrient vector per search term, we reverse-run
the extractor: group FDC entries by their extracted term, average the
nutrient values within each bucket.

Output: priced_foods.json keyed by search term, with:
  - price_per_100g (from final_prices.json)
  - price_source   (kroger | tfp)
  - nutrients      (mean of all FDC descriptions that map to this term)
  - fdc_count      (how many FDC descriptions contributed)
  - fdc_sources    (first 10 descriptions, for audit)

Usage:
    python scripts/build_priced_foods.py \\
        --prices final_prices.json \\
        --nutrients fresh_foods_nutrients_names_physiology.json \\
        --output priced_foods.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the extractor that produced the search terms in the first place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_prices import extract_search_term  # noqa: E402


def group_fdc_by_search_term(fdc_nutrients: dict) -> dict[str, list[tuple[str, dict]]]:
    """Reverse-run the extractor; return {search_term: [(fdc_desc, nutrients), ...]}."""
    buckets: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for fdc_desc, nutrients in fdc_nutrients.items():
        term = extract_search_term(fdc_desc)
        if term is None:
            continue
        buckets[term].append((fdc_desc, nutrients))
    return buckets


def average_nutrients(entries: list[tuple[str, dict]]) -> dict[str, float]:
    """Mean nutrient values across FDC entries in the same search-term bucket."""
    per_nutrient: dict[str, list[float]] = defaultdict(list)
    for _desc, nutrients in entries:
        for name, value in nutrients.items():
            try:
                per_nutrient[name].append(float(value))
            except (TypeError, ValueError):
                continue
    return {k: statistics.mean(v) for k, v in per_nutrient.items() if v}


def build_priced_foods(
    final_prices: dict[str, dict],
    fdc_buckets: dict[str, list[tuple[str, dict]]],
) -> tuple[dict[str, dict], dict[str, int]]:
    """Join final_prices.json entries with their nutrient buckets."""
    out: dict[str, dict] = {}
    counts = {"matched": 0, "no_fdc_bucket": 0, "no_nutrients": 0}
    for term, price_entry in final_prices.items():
        bucket = fdc_buckets.get(term, [])
        if not bucket:
            counts["no_fdc_bucket"] += 1
            continue
        nutrients = average_nutrients(bucket)
        if not nutrients:
            counts["no_nutrients"] += 1
            continue
        counts["matched"] += 1
        out[term] = {
            "price_per_100g": price_entry["price_per_100g"],
            "price_source": price_entry["price_source"],
            "nutrients": nutrients,
            "fdc_count": len(bucket),
            "fdc_sources": [desc for desc, _ in bucket[:10]],
        }
        # Preserve optional provenance fields when available
        for field in ("claude_confidence", "claude_reason",
                      "raw_description", "tfp_category",
                      "inflation_factor", "cpi_current",
                      "package_size_g"):
            if field in price_entry:
                out[term][field] = price_entry[field]
    return out, counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="final_prices.json")
    p.add_argument("--nutrients",
                   default="fresh_foods_nutrients_names_physiology.json")
    p.add_argument("--output", default="priced_foods.json")
    args = p.parse_args()

    prices = json.loads(Path(args.prices).read_text())
    fdc_nutrients = json.loads(Path(args.nutrients).read_text())
    print(f"loaded {len(prices)} priced terms and "
          f"{len(fdc_nutrients)} FDC descriptions", file=sys.stderr)

    buckets = group_fdc_by_search_term(fdc_nutrients)
    print(f"extracted {len(buckets)} unique search terms "
          f"(avg {sum(len(b) for b in buckets.values())/max(len(buckets),1):.1f} "
          f"FDC descriptions per term)", file=sys.stderr)

    priced, counts = build_priced_foods(prices, buckets)

    # Inject pipeline metadata (zip code, location, fetch date) from
    # prices_raw.json if available — used by the web UI header.
    raw_path = Path("prices_raw.json")
    if raw_path.exists():
        try:
            raw_meta = json.loads(raw_path.read_text())
            priced["_metadata"] = {
                "zip_code": raw_meta.get("zip_code"),
                "location_id": raw_meta.get("location_id"),
                "fetched_at": raw_meta.get("fetched_at"),
                "retailer": raw_meta.get("retailer"),
            }
        except Exception:
            pass

    Path(args.output).write_text(json.dumps(priced, indent=2, sort_keys=True))

    print(f"\nwrote {counts['matched']}/{len(prices)} priced foods "
          f"with nutrients → {args.output}", file=sys.stderr)
    if counts["no_fdc_bucket"]:
        print(f"  {counts['no_fdc_bucket']} priced terms had no FDC bucket "
              f"(search-term format drift?)", file=sys.stderr)
    if counts["no_nutrients"]:
        print(f"  {counts['no_nutrients']} terms had FDC buckets but no "
              f"numeric nutrients", file=sys.stderr)

    # Spot-check by price source
    from collections import Counter
    by_source = Counter(v["price_source"] for k, v in priced.items() if k != "_metadata")
    print(f"\nBy price source:", file=sys.stderr)
    for src, n in by_source.most_common():
        print(f"  {src}: {n}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
