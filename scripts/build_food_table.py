#!/usr/bin/env python3
"""Build the unified FDC-keyed food table.

Input:
  - food_info.json              (name-keyed price + cup eq + yield)
  - fresh_foods_nutrients_names_physiology.json (name-keyed nutrients,
    after the physiology alias mapping in notebook cell 15)
  - data/food_name_mapping.csv   (variant -> canonical, from #5)
  - Optional: FDC food.csv to resolve canonical names to fdc_id

Output:
  - food_table.json   {fdc_id: {name, price_per_100g, price_source,
                               cup_equivalent_g, yield, nutrients}}

If FDC food.csv is not available, synthetic negative fdc_ids are
assigned so downstream code is still fdc_id-keyed and the join will
lock in once real IDs are present.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_name_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path) as f:
        for row in csv.DictReader(f):
            mapping[row["variant"]] = row["canonical"]
    return mapping


def load_fdc_id_lookup(fdc_csv: Path | None) -> dict[str, int]:
    """Return {lowercased_food_name: fdc_id} from FDC food.csv when present."""
    if fdc_csv is None or not fdc_csv.exists():
        return {}
    out: dict[str, int] = {}
    with open(fdc_csv) as f:
        for row in csv.DictReader(f):
            try:
                out[row["description"].strip().lower()] = int(row["fdc_id"])
            except (KeyError, ValueError):
                continue
    return out


def build_unified(
    food_info: dict,
    nutrients_by_name: dict,
    name_mapping: dict[str, str],
    fdc_lookup: dict[str, int],
    price_source: str = "ers_2022",
) -> dict[str, dict[str, Any]]:
    """Assemble the unified table, keyed on fdc_id (str for JSON)."""
    unified: dict[str, dict[str, Any]] = {}
    next_synthetic = -1
    for info_name, info in food_info.items():
        canonical = name_mapping.get(info_name, info_name)
        fdc_id = fdc_lookup.get(canonical.lower())
        if fdc_id is None:
            fdc_id = next_synthetic
            next_synthetic -= 1
        nutrients = nutrients_by_name.get(info_name, {})
        price = info.get("price")
        unified[str(fdc_id)] = {
            "name": canonical,
            "price_per_100g": (price / info.get("yield", 1.0) / 4.54) if price else None,
            "price_source": price_source,
            "cup_equivalent_g": info.get("cupEQ"),
            "yield": info.get("yield"),
            "nutrients": nutrients,
        }
    return unified


REQUIRED_FIELDS = {"name", "price_per_100g", "cup_equivalent_g", "yield", "nutrients"}


def validate_unified(unified: dict) -> list[str]:
    """Pydantic-light schema check; returns list of violations."""
    errors = []
    for key, entry in unified.items():
        try:
            int(key)  # fdc_id must be int-parseable
        except ValueError:
            errors.append(f"{key}: not an fdc_id")
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            errors.append(f"{key}: missing fields {missing}")
        if not isinstance(entry.get("nutrients"), dict):
            errors.append(f"{key}: nutrients must be dict")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    food_info = load_json(root / "food_info.json")
    nutrients = load_json(root / "fresh_foods_nutrients_names_physiology.json")
    mapping = load_name_mapping(root / "data" / "food_name_mapping.csv")
    fdc_csv = root.parent / "food.csv"  # gitignored; optional
    fdc_lookup = load_fdc_id_lookup(fdc_csv if fdc_csv.exists() else None)
    unified = build_unified(food_info, nutrients, mapping, fdc_lookup)
    violations = validate_unified(unified)
    if violations:
        print("VALIDATION FAILED:", *violations, sep="\n  ", file=sys.stderr)
        return 1
    out = root / "food_table.json"
    with open(out, "w") as f:
        json.dump(unified, f, indent=2)
    synthetic = sum(1 for k in unified if int(k) < 0)
    print(f"wrote {out} ({len(unified)} foods; {synthetic} synthetic fdc_ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
