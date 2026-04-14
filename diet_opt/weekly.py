"""Weekly (7-day) extension of the diet LP.

Replaces scalar `f` variables with `f[food, day]` matrix variables.
Nutrient constraints apply per-day (sodium, energy, protein/EAAs) or
weekly-averaged (fiber, most vitamins/minerals) per
`data/nutrient_cadence.yaml`.

Package-level (MILP) purchase granularity is explicitly out of scope:
issue tagged it as "stretch", solve times jump from 1s to 10s+, and the
weekly LP is already a useful artifact on its own.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .data import parse_bound

DAYS_PER_WEEK = 7
DEFAULT_CADENCE_PATH = Path(__file__).resolve().parent.parent / "data" / "nutrient_cadence.yaml"


def load_cadence(path: Path | str = DEFAULT_CADENCE_PATH) -> dict[str, str]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def is_per_day(nutrient: str, cadence: dict[str, str]) -> bool:
    """Per issue discussion, default to weekly for unlisted nutrients."""
    return cadence.get(nutrient) == "daily"


def scale_bounds_for_cadence(
    lb: float, ub: float, nutrient: str, cadence: dict[str, str], days: int = DAYS_PER_WEEK
) -> tuple[float, float, str]:
    """Return (lb, ub, cadence) bounds appropriate for the constraint.

    For per-day nutrients: bounds unchanged (each day must satisfy).
    For weekly-averaged: bounds multiplied by `days` (sum over week must
    satisfy the daily DRI × 7).
    """
    if is_per_day(nutrient, cadence):
        return lb, ub, "daily"
    return lb * days, ub * days, "weekly"


def build_weekly_variables(food_info: dict, days: int = DAYS_PER_WEEK):
    """Build {(food_key, day): variable} for a 7-day model.

    Placeholder that returns variable-spec tuples instead of constructing
    Optlang objects directly — lets the module remain testable without
    requiring the modelseedpy fork for pytest.
    """
    return {
        (food.replace(" ", "_"), d): {"name": f"{food.replace(' ', '_')}__d{d}", "lb": 0.0, "ub": 5.0}
        for food in food_info
        for d in range(days)
    }


def weekly_cost_objective(
    variables: dict,
    food_info: dict,
    food_price: callable | None = None,
) -> list[tuple[str, float]]:
    """Emit (variable_name, cost_coefficient) pairs for the weekly objective.

    One entry per (food, day) variable. Cost coefficient is price-per-100g
    (matches scalar daily model's `price/yield/4.54`).
    """
    out = []
    for (food_key, day), spec in variables.items():
        food = food_key.replace("_", " ")
        info = food_info.get(food, {})
        coef = food_price(food) if food_price else info["price"] / max(info.get("yield", 1.0), 0.01) / 4.54
        out.append((spec["name"], coef))
    return out


def distinct_foods_across_week(solution: dict[tuple[str, int], float], min_grams: float = 20.0) -> int:
    """Count foods that appear (>= min_grams on any day) anywhere in the week.

    Used by the `≥N distinct foods per week` variety constraint envisioned
    in the issue, applied post-solve as a realism check or pre-solve as
    a diversity lower bound (future work — requires binary indicator vars).
    """
    foods_used: set[str] = set()
    for (food, _day), amount in solution.items():
        if amount * 100 >= min_grams:
            foods_used.add(food)
    return len(foods_used)
