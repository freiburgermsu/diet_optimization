"""Secondary objective terms: sodium, polyphenols (was ORAC), carbon.

Source decisions (locked in after review):
- Polyphenols replace ORAC. USDA's ORAC database was withdrawn in 2012
  as non-predictive of in-vivo antioxidant activity. Phenol-Explorer 3.6
  (http://phenol-explorer.eu) is peer-reviewed, widely cited, and its
  "total polyphenol content by Folin method" is a defensible dietary
  bioactive proxy without ORAC's scientific baggage.
- Carbon footprint from Poore & Nemecek 2018 (Science). Resolution is
  food-category, not per-FDC-ID — acknowledged lossy join. Foods in the
  same P&N category (e.g. all pulses) get identical kg CO2e/kg.

ε-constraint is preferred over weighted-sum for non-convex Pareto
fronts (see my earlier analysis on #14). For simple "max polyphenols
given a cost budget" or "min cost given a carbon ceiling", ε-constraint
is one extra LP per Pareto point and always lands on a supported point.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class ObjectiveConfig:
    minimize_sodium_weight: float = 0.0
    maximize_polyphenols_weight: float = 0.0
    carbon_ceiling_kg_co2e: float | None = None   # per day, implemented as an ε-constraint


def sodium_contribution(food_matches: dict, food: str) -> float:
    """Return sodium in mg per 100g edible, or 0 if not reported."""
    return food_matches.get(food, {}).get("Sodium", 0.0)


def _load_csv_map(path: Path, key_col: str, value_col: str, skip_comments: bool = True) -> dict[str, float]:
    """Load a two-column lookup from a commented CSV."""
    out: dict[str, float] = {}
    with open(path) as f:
        lines = (l for l in f if not (skip_comments and l.lstrip().startswith("#")))
        for row in csv.DictReader(lines):
            try:
                out[row[key_col]] = float(row[value_col])
            except (KeyError, ValueError):
                continue
    return out


def load_polyphenol_content(path: Path | str = DATA_DIR / "polyphenol_content.csv") -> dict[str, float]:
    """Return {food_name: polyphenol_mg_per_100g}."""
    return _load_csv_map(Path(path), "food_name", "polyphenol_mg_per_100g")


def load_carbon_footprint(path: Path | str = DATA_DIR / "carbon_footprint.csv") -> dict[str, float]:
    """Return {food_name: kg_co2e_per_kg}."""
    return _load_csv_map(Path(path), "food_name", "kg_co2e_per_kg")


def build_secondary_term(
    food_info: dict,
    food_matches: dict,
    variables: dict,
    cfg: ObjectiveConfig,
    polyphenols: dict[str, float] | None = None,
) -> list[dict]:
    """Emit objective-expression elements for enabled secondary terms.

    Variable units: 1.0 = 100 g (matches scalar model convention).
    - Sodium term: weight × (mg sodium / 100 g) per food
    - Polyphenol term: (-weight) × (mg polyphenol / 100 g) per food
      (negated because it's a maximize disguised as a cost to add)
    Carbon is not an objective term — it's an ε-constraint (see
    build_carbon_ceiling_constraint).
    """
    terms: list[dict] = []
    polyphenols = polyphenols if polyphenols is not None else {}

    for food in food_info:
        key = food.replace(" ", "_")
        if key not in variables:
            continue

        if cfg.minimize_sodium_weight > 0:
            sodium_mg = sodium_contribution(food_matches, food)
            if sodium_mg > 0:
                terms.append({
                    "elements": [{
                        "elements": [variables[key].name, cfg.minimize_sodium_weight * sodium_mg],
                        "operation": "Mul",
                    }],
                    "operation": "Add",
                })

        if cfg.maximize_polyphenols_weight > 0:
            polyphenol_mg = polyphenols.get(food, 0.0)
            if polyphenol_mg > 0:
                terms.append({
                    "elements": [{
                        "elements": [variables[key].name, -cfg.maximize_polyphenols_weight * polyphenol_mg],
                        "operation": "Mul",
                    }],
                    "operation": "Add",
                })

    return terms


def build_carbon_ceiling_constraint(
    food_info: dict,
    variables: dict,
    carbon: dict[str, float],
    ceiling_kg_co2e: float,
) -> dict:
    """Return an ε-constraint spec: total daily CO2e ≤ ceiling.

    Variable units: 1.0 = 100 g = 0.1 kg, so per-food coefficient is
    kg_co2e_per_kg * 0.1.

    Returns {name, bounds, expr} compatible with tupConstraint format.
    """
    elements = []
    for food in food_info:
        key = food.replace(" ", "_")
        if key not in variables:
            continue
        co2e_per_kg = carbon.get(food)
        if co2e_per_kg is None:
            continue
        elements.append({
            "elements": [variables[key].name, co2e_per_kg * 0.1],
            "operation": "Mul",
        })
    return {
        "name": "carbon_footprint_ceiling",
        "lb": 0.0,
        "ub": ceiling_kg_co2e,
        "expr": {"elements": elements, "operation": "Add"},
    }


def pareto_sweep_points(
    objective_values: list[float], num_points: int = 10
) -> list[float]:
    """Return evenly spaced ε-constraint bounds between min and max.

    Use case: solve the cost-only LP to find max achievable carbon;
    solve the carbon-only LP to find min achievable carbon; sweep
    `num_points` ε values between those and re-solve the cost LP at
    each to trace the Pareto front.
    """
    if not objective_values:
        return []
    lo, hi = min(objective_values), max(objective_values)
    if num_points < 2:
        return [lo]
    step = (hi - lo) / (num_points - 1)
    return [lo + step * i for i in range(num_points)]
