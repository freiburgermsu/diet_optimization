"""Build the LP directly on optlang (no private-fork dependency).

Originally extracted from the notebook and wrapped through
`modelseedpy.core.optlanghelper` (a private fork). Rewritten to use
optlang's native Model/Variable/Constraint/Objective so the LP runs on
any installation with `pip install optlang`.

Behavior preserved bit-for-bit from cell 21:
  - Per-nutrient linear constraints with DRI lower/upper bounds
  - Water units converted from g to L via grams-per-liter=998
  - Skip nutrients reported by <SPARSE_NUTRIENT_THRESHOLD foods
  - 5-20 cup volume constraint (skippable via include_volume=False
    since the 610-food priced table rarely has per-food cupEQ data)
  - Objective: sum_f(var_f * price_per_100g_edible)
"""
from __future__ import annotations

from .data import parse_bound

GRAMS_PER_LITER = 998
SPARSE_NUTRIENT_THRESHOLD = 6  # see #8 for the per-nutrient triage follow-up


def _sparse_nutrients(food_info: dict, food_matches: dict, nutrition: dict) -> dict[str, int]:
    """Count foods that report each nutrient — used to skip sparse ones."""
    return {
        nutrient: sum(
            1 for food in food_info if nutrient in food_matches.get(food, {})
        )
        for nutrient in nutrition
    }


def _safe_name(food: str) -> str:
    """Optlang variable names can't contain spaces / some punctuation."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in food.replace(" ", "_"))


def build_model(
    food_info: dict,
    food_matches: dict,
    nutrition: dict,
    include_volume: bool = True,
):
    """Construct the LP from the three input tables using optlang directly.

    Returns: (model, variables, constraints) where:
      - model: optlang.Model
      - variables: {safe_name: Variable}
      - constraints: {name: Constraint}
    """
    import optlang

    variables: dict[str, "optlang.Variable"] = {
        _safe_name(food): optlang.Variable(_safe_name(food), lb=0, ub=5, type="continuous")
        for food in food_info
    }

    model = optlang.Model(name="minimize_nutrition_cost")
    model.add(list(variables.values()))

    constraints: dict[str, "optlang.Constraint"] = {}
    support = _sparse_nutrients(food_info, food_matches, nutrition)

    for nutrient, content in nutrition.items():
        if support[nutrient] < SPARSE_NUTRIENT_THRESHOLD:
            continue
        lb = parse_bound(content["low_bound"])
        ub = parse_bound(content["high_bound"])

        expr_terms = []
        for food in food_info:
            if nutrient not in food_matches[food]:
                continue
            amount = food_matches[food][nutrient]
            if nutrient == "Total Water":
                amount /= GRAMS_PER_LITER
            expr_terms.append(amount * variables[_safe_name(food)])
        if not expr_terms:
            continue

        expr = sum(expr_terms)
        cname = _safe_name(nutrient)
        # Replace +inf with None so optlang treats it as unbounded
        c_lb = lb if lb != float("inf") else None
        c_ub = ub if ub != float("inf") else None
        c = optlang.Constraint(expr, lb=c_lb, ub=c_ub, name=cname)
        constraints[cname] = c
        model.add(c)

    if include_volume:
        volume_expr = sum(
            info["cupEQ"] * variables[_safe_name(f)]
            for f, info in food_info.items()
        )
        vol = optlang.Constraint(volume_expr, lb=5, ub=20, name="volume")
        constraints["volume"] = vol
        model.add(vol)

    # Objective: minimize sum_f(var_f * price_per_100g_edible)
    obj_expr = sum(
        (pricing["price"] / pricing["yield"] / 4.54) * variables[_safe_name(food)]
        for food, pricing in food_info.items()
    )
    model.objective = optlang.Objective(obj_expr, direction="min")

    return model, variables, constraints
