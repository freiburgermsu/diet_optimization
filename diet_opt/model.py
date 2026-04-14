"""Build the Optlang LP from loaded data.

Extracted from `optimization_diet.ipynb` cell 21. Behavior is preserved
bit-for-bit — constraint shape, volume cap, objective formula, and the
`<6 foods => skip nutrient` heuristic all unchanged.
"""
from __future__ import annotations

from .data import parse_bound

GRAMS_PER_LITER = 998
SPARSE_NUTRIENT_THRESHOLD = 6  # nutrients with fewer supporting foods are skipped; see #8


def _sparse_nutrients(food_info: dict, food_matches: dict, nutrition: dict) -> dict[str, int]:
    """Count foods that report each nutrient — used to skip sparse ones."""
    counts = {}
    for nutrient in nutrition:
        counts[nutrient] = sum(
            1 for food in food_info if nutrient in food_matches.get(food, {})
        )
    return counts


def build_model(food_info: dict, food_matches: dict, nutrition: dict):
    """Construct the LP model from the three input tables.

    Requires a build of `modelseedpy` that includes `core.optlanghelper`
    (the notebook's private fork); PyPI `modelseedpy==0.4.2` does not.
    See #18 follow-up for replacing this with a direct optlang build.
    """
    from modelseedpy.core.optlanghelper import (
        Bounds,
        OptlangHelper,
        tupConstraint,
        tupObjective,
        tupVariable,
    )

    variables = {
        food.replace(" ", "_"): tupVariable(food.replace(" ", "_"), Bounds(0, 5), "continuous")
        for food in food_info
    }

    constraints = {}
    support = _sparse_nutrients(food_info, food_matches, nutrition)

    for nutrient, content in nutrition.items():
        if support[nutrient] < SPARSE_NUTRIENT_THRESHOLD:
            continue
        lb = parse_bound(content["low_bound"])
        ub = parse_bound(content["high_bound"])
        nutrient_foods = {}
        for food in food_info:
            if nutrient not in food_matches[food]:
                continue
            amount = food_matches[food][nutrient]
            if nutrient == "Total Water":
                amount /= GRAMS_PER_LITER
            key = food.replace(" ", "_")
            nutrient_foods[key] = {
                "elements": [variables[key].name, amount],
                "operation": "Mul",
            }
        cname = nutrient.replace(" ", "_")
        constraints[cname] = tupConstraint(
            name=cname,
            bounds=Bounds(lb, ub),
            expr={"elements": list(nutrient_foods.values()), "operation": "Add"},
        )

    volume_expr = {
        "elements": [
            {"elements": [variables[f.replace(" ", "_")].name, info["cupEQ"]], "operation": "Mul"}
            for f, info in food_info.items()
        ],
        "operation": "Add",
    }
    constraints["volume"] = tupConstraint(name="volume", bounds=Bounds(5, 20), expr=volume_expr)

    objective = tupObjective("minimize cost of nutritional diet", [], "min")
    for food, pricing in food_info.items():
        key = food.replace(" ", "_")
        objective.expr.append({
            "elements": [{
                "elements": [variables[key].name, pricing["price"] / pricing["yield"] / 4.54],
                "operation": "Mul",
            }],
            "operation": "Add",
        })

    model = OptlangHelper.define_model(
        "minimize_nutrition_cost",
        list(variables.values()),
        list(constraints.values()),
        objective,
        True,
    )
    return model, variables, constraints
