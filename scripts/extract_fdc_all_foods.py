#!/usr/bin/env python3
"""Extract ALL sr_legacy + foundation foods from the local USDA FDC CSVs.

Replaces the manually curated fresh_foods_nutrients_names_physiology.json
with a comprehensive extraction of ~8,000 foods from the full FDC database.
The `whole_foods` dietary preset filters back to fresh/unprocessed foods.

Reads:
  ../food.csv           (206 MB — food descriptions + categories)
  ../food_nutrient.csv  (1.6 GB — nutrient values per food)
  nutrient.csv          (nutrient ID → name mapping)

Writes:
  fresh_foods_nutrients_names_physiology.json  (overwritten with expanded set)

Usage:
  uv run python scripts/extract_fdc_all_foods.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FDC_DIR = REPO.parent  # food.csv and food_nutrient.csv are one level up

FOOD_CSV = FDC_DIR / "food.csv"
FOOD_NUTRIENT_CSV = FDC_DIR / "food_nutrient.csv"
NUTRIENT_CSV = REPO / "nutrient.csv"
OUTPUT = REPO / "fresh_foods_nutrients_names_physiology.json"

# Data types to include (high-quality curated data, not branded products)
INCLUDE_DATA_TYPES = {"sr_legacy_food", "foundation_food"}

# Map USDA FDC nutrient names → our pipeline's nutrient names.
# Only nutrients we track in nutrition.json + amino acids + fatty acids.
NUTRIENT_MAP = {
    "Energy": "Energy",
    "Protein": "Protein",
    "Total lipid (fat)": "Fat",
    "Carbohydrate, by difference": "Carbohydrate",
    "Fiber, total dietary": "Total Fiber",
    "Fatty acids, total saturated": "Saturated fatty acids",
    # Linoleic acid (omega-6) — multiple FDC name variants
    "PUFA 18:2 n-6 c,c": "Linoleic Acid",
    "PUFA 18:2": "Linoleic Acid",               # general form used by many foods
    "18:2 n-6 c,c": "Linoleic Acid",            # legacy name
    "PUFA 18:2 c": "Linoleic Acid",
    # Linolenic acid / ALA (omega-3) — multiple FDC name variants
    "PUFA 18:3 n-3 c,c,c (ALA)": "Linolenic Acid",
    "PUFA 18:3": "Linolenic Acid",              # general form
    "18:3 n-3 c,c,c (ALA)": "Linolenic Acid",  # legacy name
    "PUFA 18:3 c": "Linolenic Acid",
    "Cholesterol": "Dietary Cholesterol",
    "Water": "Total Water",
    # Vitamins
    "Vitamin A, RAE": "Vitamin A",
    "Vitamin C, total ascorbic acid": "Vitamin C",
    "Vitamin D (D2 + D3)": "Vitamin D",
    "Vitamin E (alpha-tocopherol)": "Vitamin E",
    "Vitamin K (phylloquinone)": "Vitamin K",
    "Thiamin": "Thiamin",
    "Riboflavin": "Riboflavin",
    "Niacin": "Niacin",
    "Pantothenic acid": "Pantothenic Acid",
    "Vitamin B-6": "Vitamin B6",
    "Folate, total": "Folate",
    "Vitamin B-12": "Vitamin B12",
    "Choline, total": "Choline",
    "Biotin": "Biotin",
    "Carotene, beta": "Carotenoids",
    # Minerals
    "Calcium, Ca": "Calcium",
    "Iron, Fe": "Iron",
    "Magnesium, Mg": "Magnesium",
    "Phosphorus, P": "Phosphorus",
    "Potassium, K": "Potassium",
    "Sodium, Na": "Sodium",
    "Zinc, Zn": "Zinc",
    "Copper, Cu": "Copper",
    "Manganese, Mn": "Manganese",
    "Selenium, Se": "Selenium",
    "Iodine, I": "Iodine",
    "Molybdenum, Mo": "Molybdenum",
    "Chromium, Cr": "Chromium",
    "Fluoride, F": "Fluoride",
    "Chlorine, Cl": "Chloride",
    # Amino acids
    "Histidine": "Histidine",
    "Isoleucine": "Isoleucine",
    "Leucine": "Leucine",
    "Lysine": "Lysine",
    "Methionine": "Methionine",
    "Phenylalanine": "Phenylalanine",
    "Threonine": "Threonine",
    "Tryptophan": "Tryptophan",
    "Valine": "Valine",
    "Tyrosine": "Tyrosine",
    "Arginine": "Arginine",
    # EPA / DHA — multiple FDC name variants
    "PUFA 20:5 n-3 (EPA)": "PUFA 20:5 n-3 (EPA)",
    "20:5 n-3 (EPA)": "PUFA 20:5 n-3 (EPA)",
    "PUFA 20:5c": "PUFA 20:5 n-3 (EPA)",
    "PUFA 22:6 n-3 (DHA)": "PUFA 22:6 n-3 (DHA)",
    "22:6 n-3 (DHA)": "PUFA 22:6 n-3 (DHA)",
    "PUFA 22:6 c": "PUFA 22:6 n-3 (DHA)",
    # Monounsaturated / polyunsaturated totals
    "Fatty acids, total monounsaturated": "Fatty acids, total monounsaturated",
    "Fatty acids, total polyunsaturated": "Fatty acids, total polyunsaturated",
}


def main() -> int:
    t0 = time.perf_counter()

    # --- Step 1: Load nutrient ID → name mapping ---
    nut_id_to_name: dict[str, str] = {}
    with open(NUTRIENT_CSV) as f:
        for row in csv.DictReader(f):
            nut_id_to_name[row["id"]] = row["name"]
    print(f"Loaded {len(nut_id_to_name)} nutrient IDs from {NUTRIENT_CSV.name}",
          file=sys.stderr)

    # Build nutrient_id → our_name mapping (via the two-step lookup)
    nut_id_to_our_name: dict[str, str] = {}
    for nut_id, fdc_name in nut_id_to_name.items():
        our_name = NUTRIENT_MAP.get(fdc_name)
        if our_name:
            nut_id_to_our_name[nut_id] = our_name

    # --- Step 2: Load food descriptions + filter to SR Legacy / Foundation ---
    target_foods: dict[str, str] = {}  # fdc_id → description
    with open(FOOD_CSV) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            fdc_id, data_type, description = row[0], row[1], row[2]
            if data_type in INCLUDE_DATA_TYPES:
                target_foods[fdc_id] = description
    print(f"Found {len(target_foods)} foods ({', '.join(INCLUDE_DATA_TYPES)}) "
          f"from {FOOD_CSV.name}", file=sys.stderr)

    # --- Step 3: Read food_nutrient.csv (1.6 GB) and collect nutrients ---
    food_nutrients: dict[str, dict[str, float]] = {fdc: {} for fdc in target_foods}
    target_set = set(target_foods)  # fast lookup
    mapped_ids = set(nut_id_to_our_name)

    lines_read = 0
    with open(FOOD_NUTRIENT_CSV) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            lines_read += 1
            if lines_read % 5_000_000 == 0:
                print(f"  ...{lines_read:,} lines read", file=sys.stderr)
            fdc_id = row[1]
            if fdc_id not in target_set:
                continue
            nut_id = row[2]
            if nut_id not in mapped_ids:
                continue
            amount = float(row[3]) if row[3] else 0.0
            our_name = nut_id_to_our_name[nut_id]
            food_nutrients[fdc_id][our_name] = amount

    print(f"Read {lines_read:,} nutrient rows in {time.perf_counter()-t0:.1f}s",
          file=sys.stderr)

    # --- Step 4: Build output dict keyed by FDC description ---
    # If multiple FDC IDs have the same description, average their nutrients.
    from collections import defaultdict
    desc_buckets: dict[str, list[dict]] = defaultdict(list)
    for fdc_id, desc in target_foods.items():
        nuts = food_nutrients[fdc_id]
        if nuts:  # skip foods with no nutrient data
            desc_buckets[desc].append(nuts)

    output: dict[str, dict[str, float]] = {}
    for desc, nut_list in desc_buckets.items():
        if not nut_list:
            continue
        # Average across duplicates
        merged: dict[str, list[float]] = {}
        for nuts in nut_list:
            for k, v in nuts.items():
                merged.setdefault(k, []).append(v)
        output[desc] = {k: sum(vs) / len(vs) for k, vs in merged.items()}

    # Also preserve any manually added entries (dairy, etc.) from the existing file
    existing_count = 0
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text())
        for desc, nuts in existing.items():
            if desc not in output:
                output[desc] = nuts
                existing_count += 1

    OUTPUT.write_text(json.dumps(output, indent=2))
    elapsed = time.perf_counter() - t0
    print(f"\nWrote {len(output)} foods → {OUTPUT.name} "
          f"({existing_count} preserved from existing, "
          f"{len(output) - existing_count} from FDC) in {elapsed:.1f}s",
          file=sys.stderr)

    # Summary by nutrient coverage
    has_energy = sum(1 for v in output.values() if "Energy" in v)
    has_protein = sum(1 for v in output.values() if "Protein" in v)
    has_ala = sum(1 for v in output.values() if "Linolenic Acid" in v)
    has_epa = sum(1 for v in output.values() if "PUFA 20:5 n-3 (EPA)" in v)
    print(f"Coverage: Energy={has_energy}, Protein={has_protein}, "
          f"ALA={has_ala}, EPA/DHA={has_epa}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
