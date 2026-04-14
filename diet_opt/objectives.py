"""Secondary objective terms (sodium, ORAC, carbon).

ε-constraint is the right pattern for non-convex Pareto fronts (see my
analysis on #14). Weighted-sum is provided too but only recommended for
quick exploration — it misses parts of the front.

Shipping: sodium minimization only. ORAC and carbon are stubbed; they
need source decisions (USDA ORAC was withdrawn as unscientific; need
Phenol-Explorer or similar. Carbon: Poore-Nemecek 2018 categories don't
map cleanly to FDC IDs). Flagged for user input before wiring.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveConfig:
    minimize_sodium_weight: float = 0.0
    # maximize_orac_weight: float = 0.0   # blocked on source decision
    # carbon_ceiling_kg: float | None = None  # blocked on P-N 2018 mapping


def sodium_contribution(food_matches: dict, food: str) -> float:
    """Return sodium in mg per 100g edible, or 0 if not reported."""
    return food_matches.get(food, {}).get("Sodium", 0.0)


def build_secondary_term(
    food_info: dict, food_matches: dict, variables: dict, cfg: ObjectiveConfig
) -> list[dict]:
    """Produce objective-expression elements for secondary terms.

    Returns a list of {elements, operation} dicts compatible with the
    tupObjective.expr format used in diet_opt/model.py.
    Empty list when no secondary terms are active.
    """
    terms: list[dict] = []
    if cfg.minimize_sodium_weight > 0:
        for food in food_info:
            sodium_mg = sodium_contribution(food_matches, food)
            if sodium_mg == 0:
                continue
            key = food.replace(" ", "_")
            terms.append({
                "elements": [{
                    "elements": [variables[key].name, cfg.minimize_sodium_weight * sodium_mg],
                    "operation": "Mul",
                }],
                "operation": "Add",
            })
    return terms
