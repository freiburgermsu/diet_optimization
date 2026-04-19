"""Build the LP as a raw JSON dict and load via optlang.Model.from_json.

Previously this module constructed the model through optlang's Python API
(`optlang.Variable(...)`, `amount * var` → sympy, `sum(expr_terms)`,
`model.add(...)`), which on the 610-food priced table spent ~26 s in
Python/sympy before the solver ever ran.

Raw-JSON construction — the pattern from the local OptlangHelper package
(`/Users/andrewfreiburger/Documents/Research/OptlangHelper`) — skips that
overhead: we assemble the model as a plain dict of primitive types, then
call `Model.from_json(model)` once. No per-variable Variable() calls, no
per-term sympy Mul objects, no O(n²) `sum()` reduction.

Behavior preserved bit-for-bit from the previous implementation:
  - Per-nutrient linear constraints with DRI lower/upper bounds
  - Water units converted from g to L via grams-per-liter=998
  - Skip nutrients reported by <SPARSE_NUTRIENT_THRESHOLD foods
  - 5-20 cup volume constraint (skippable via include_volume=False)
  - Objective: sum_f(var_f * price_per_100g_edible)
"""
from __future__ import annotations

from .data import parse_bound

GRAMS_PER_LITER = 998
SPARSE_NUTRIENT_THRESHOLD = 6
MAX_SERVINGS_PER_FOOD = 4


def _sparse_nutrients(food_info: dict, food_matches: dict, nutrition: dict) -> dict[str, int]:
    return {
        nutrient: sum(
            1 for food in food_info if nutrient in food_matches.get(food, {})
        )
        for nutrient in nutrition
    }


def _safe_name(food: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in food.replace(" ", "_"))


def _linear_expr(terms: list[tuple[float, str]]) -> dict:
    """Build the optlang-JSON expression dict for sum(coef_i * var_i).

    Output shape (matches OptlangHelper._define_expression's output):
        {"type": "Add", "args": [
            {"type": "Mul", "args": [
                {"type": "Number", "value": c1},
                {"type": "Symbol", "name": "v1"},
            ]},
            ...
        ]}
    """
    return {
        "type": "Add",
        "args": [
            {
                "type": "Mul",
                "args": [
                    {"type": "Number", "value": coef},
                    {"type": "Symbol", "name": var_name},
                ],
            }
            for coef, var_name in terms
        ],
    }


def build_model(
    food_info: dict,
    food_matches: dict,
    nutrition: dict,
    include_volume: bool = True,
):
    """Construct the LP via raw-JSON → Model.from_json.

    Returns: (model, variables, constraints) where:
      - model: optlang.Model loaded from the JSON dict
      - variables: {safe_name: optlang.Variable} looked up from model
      - constraints: {safe_name: optlang.Constraint} looked up from model
    """
    from optlang import Model

    # Dedup by safe_name: two food_info keys can collide (e.g. "passion fruit"
    # vs "passion-fruit" → "passion_fruit"). The previous dict-comprehension
    # silently kept only the last; preserve that behavior while letting the
    # constraint terms still reference the shared symbol.
    seen_vars: set[str] = set()
    variables_json: list[dict] = []
    for food in food_info:
        safe = _safe_name(food)
        if safe in seen_vars:
            continue
        seen_vars.add(safe)
        variables_json.append({
            "name": safe,
            "lb": 0.0,
            "ub": float(MAX_SERVINGS_PER_FOOD),
            "type": "continuous",
        })

    constraints_json: list[dict] = []
    support = _sparse_nutrients(food_info, food_matches, nutrition)

    for nutrient, content in nutrition.items():
        if support[nutrient] < SPARSE_NUTRIENT_THRESHOLD:
            continue
        lb = parse_bound(content["low_bound"])
        ub = parse_bound(content["high_bound"])

        terms: list[tuple[float, str]] = []
        for food in food_info:
            matches = food_matches.get(food, {})
            if nutrient not in matches:
                continue
            amount = matches[nutrient]
            if nutrient == "Total Water":
                amount /= GRAMS_PER_LITER
            terms.append((float(amount), _safe_name(food)))
        if not terms:
            continue

        c_lb = float(lb) if lb != float("inf") else None
        c_ub = float(ub) if ub != float("inf") else None
        constraints_json.append({
            "name": _safe_name(nutrient),
            "expression": _linear_expr(terms),
            "lb": c_lb,
            "ub": c_ub,
            "indicator_variable": None,
            "active_when": 1,
        })

    if include_volume:
        vol_terms = [
            (float(info["cupEQ"]), _safe_name(food))
            for food, info in food_info.items()
        ]
        constraints_json.append({
            "name": "volume",
            "expression": _linear_expr(vol_terms),
            "lb": 5.0,
            "ub": 20.0,
            "indicator_variable": None,
            "active_when": 1,
        })

    obj_terms = [
        (
            float(pricing["price"]) / float(pricing["yield"]) / 4.54,
            _safe_name(food),
        )
        for food, pricing in food_info.items()
    ]
    objective_json = {
        "name": "minimize_cost",
        "expression": _linear_expr(obj_terms),
        "direction": "min",
    }

    model_dict = {
        "name": "minimize_nutrition_cost",
        "variables": variables_json,
        "constraints": constraints_json,
        "objective": objective_json,
    }
    model = Model.from_json(model_dict)

    variables = {vj["name"]: model.variables[vj["name"]] for vj in variables_json}
    constraints = {cj["name"]: model.constraints[cj["name"]] for cj in constraints_json}

    return model, variables, constraints
