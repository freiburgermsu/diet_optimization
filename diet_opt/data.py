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
