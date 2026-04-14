#!/usr/bin/env python3
"""Normalize raw retailer prices to $/100g edible and join to FDC IDs.

Input:  prices_raw.json from fetch_prices.py
Output: prices.json  {fdc_id: {price_per_100g, price_source, yield,
                               fetched_at, raw_description}}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def match_to_fdc(
    products: list[dict],
    fdc_name_to_id: dict[str, int],
    min_fuzzy_score: float = 85.0,
) -> dict[int, dict]:
    """Match retailer products to FDC IDs by description.

    Uses rapidfuzz if available (see #5); otherwise falls back to exact
    lowercase string match. Low-confidence matches are dropped — they
    show up in `unresolved_matches.csv` for review, not silently
    mis-joined.
    """
    try:
        from rapidfuzz import fuzz, process
        use_fuzzy = True
    except ImportError:
        use_fuzzy = False

    matched: dict[int, dict] = {}
    names = list(fdc_name_to_id)
    for p in products:
        desc = (p.get("description") or "").lower()
        if not desc:
            continue
        if use_fuzzy:
            hit = process.extractOne(desc, names, scorer=fuzz.WRatio)
            if hit is None or hit[1] < min_fuzzy_score:
                continue
            fdc_id = fdc_name_to_id[hit[0]]
        else:
            if desc not in fdc_name_to_id:
                continue
            fdc_id = fdc_name_to_id[desc]
        if fdc_id in matched:
            continue  # first match wins — assumes retailer products are sorted by relevance
        matched[fdc_id] = p
    return matched


def build_normalized_prices(
    raw: dict,
    fdc_name_to_id: dict[str, int],
    yields: dict[int, float] | None = None,
) -> dict[str, dict]:
    yields = yields or {}
    matched = match_to_fdc(raw.get("products", []), fdc_name_to_id)
    out: dict[str, dict] = {}
    fetched_at = raw.get("fetched_at")
    for fdc_id, product in matched.items():
        y = yields.get(fdc_id, 1.0)
        out[str(fdc_id)] = {
            "price_per_100g": price_per_100g_edible(
                product["price"], product["package_size_g"], y
            ),
            "price_source": raw.get("retailer", "retailer"),
            "yield": y,
            "fetched_at": fetched_at,
            "raw_description": product.get("description"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="prices_raw.json")
    parser.add_argument("--output", default="prices.json")
    parser.add_argument("--fdc-lookup", help="optional FDC food.csv for name→ID", default=None)
    args = parser.parse_args()

    raw = json.loads(Path(args.raw).read_text())

    fdc_lookup: dict[str, int] = {}
    if args.fdc_lookup and Path(args.fdc_lookup).exists():
        import csv
        with open(args.fdc_lookup) as f:
            for row in csv.DictReader(f):
                try:
                    fdc_lookup[row["description"].lower()] = int(row["fdc_id"])
                except (KeyError, ValueError):
                    continue

    normalized = build_normalized_prices(raw, fdc_lookup)
    Path(args.output).write_text(json.dumps(normalized, indent=2))
    print(f"wrote {len(normalized)} normalized entries to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
