"""Load and normalize USDA + DRI inputs.

Pulled from `optimization_diet.ipynb` cells 13–19. The heavy FDC CSV
preprocessing (cells 2–11) still lives in the notebook — extracting it
requires the >1.7 GB raw CSVs which are gitignored (see #6 for version
pinning, #4 for unified schema).
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import inf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent


def load_json(name: str) -> dict:
    with open(DATA_DIR / name) as f:
        return json.load(f)


def load_pipeline_inputs() -> tuple[dict, dict, dict]:
    """Load the three JSON inputs the LP consumes.

    Returns: (food_info, food_matches, nutrition)
    """
    food_info = load_json("food_info.json")
    food_matches = load_json("food_matches.json")
    nutrition = load_json("nutrition.json")
    return food_info, food_matches, nutrition


DEFAULT_CUP_EQ = 1.0   # cups per 100g-edible; used when priced_foods.json
                       # has no per-food value (most of the 610-food table).


def load_priced_foods(
    name: str = "priced_foods.json",
    default_cup_eq: float = DEFAULT_CUP_EQ,
) -> tuple[dict, dict, dict]:
    """Load priced_foods.json and split it into the (food_info, food_matches,
    nutrition) triple the existing model.build_model() expects.

    Shape of priced_foods.json (per entry):
      {price_per_100g, price_source, nutrients: {...}, ...}

    Returns:
      food_info    → {term: {price, yield, cupEQ}} where price is back-solved
                     so that `price/yield/4.54` equals the original
                     price_per_100g (preserves the existing objective formula).
      food_matches → {term: nutrients-dict}
      nutrition    → loaded from nutrition.json (unchanged)
    """
    priced = load_json(name)
    food_info: dict[str, dict] = {}
    food_matches: dict[str, dict] = {}
    for term, entry in priced.items():
        # build_model uses `price / yield / 4.54` as the per-100g cost.
        # We already have price_per_100g; set yield=1.0 and price = ppg*4.54.
        ppg = entry["price_per_100g"]
        food_info[term] = {
            "price": ppg * 4.54,
            "yield": 1.0,
            "cupEQ": entry.get("cup_equivalent", default_cup_eq),
        }
        food_matches[term] = entry.get("nutrients", {})
    nutrition = load_json("nutrition.json")
    return food_info, food_matches, nutrition


def parse_bound(raw: str | float | int) -> float:
    """Parse a DRI bound like '100', '1,200', or 'inf' into a float."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace(",", "")
    if s.lower() in {"inf", "nd", ""}:
        return inf
    return float(s.split()[0])


def validate_bounds(nutrition: dict) -> list[str]:
    """Return a list of nutrients whose lower bound exceeds upper.

    Empty list = valid. See #7 for override handling and citations.
    """
    violations = []
    for nutrient, content in nutrition.items():
        lb = parse_bound(content.get("low_bound", 0))
        ub = parse_bound(content.get("high_bound", inf))
        if lb > ub:
            violations.append(f"{nutrient}: lb={lb} > ub={ub}")
    return violations


def average_dict_values(dicts: list[dict]) -> dict:
    """Mean-across-sources for nested nutrient dicts.

    Extracted from notebook cell 16. Used to collapse multiple FDC rows
    for the same canonical food into one nutrient vector.
    """
    sum_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for d in dicts:
        for _food, nutrients in d.items():
            for nutrient, value in nutrients.items():
                sum_counts[nutrient][0] += value
                sum_counts[nutrient][1] += 1
    return {k: s / n for k, (s, n) in sum_counts.items()}
