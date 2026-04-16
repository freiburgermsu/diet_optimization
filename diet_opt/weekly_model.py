"""Weekly LP with forced per-day food rotation.

The single-day LP produces the cheapest basket that satisfies daily
DRI bounds. Replicated across 7 days it gives seven identical menus —
cheapest but monotonous. This module adds a rotation constraint:
each food can appear on at most `max_days_per_food` of the 7 days.
With typical K=3, the solver is forced to use different foods on
different days, producing a week of distinct menus.

Scale: 610 foods × 7 days × 2 variable types = 8,540 vars. To keep
solve time manageable, `build_weekly_model` accepts a pre-filtered
food pool. Use `preselect_foods` to pick the top-N cost-efficient
foods from a single-day LP solution + its nutrient neighbors.

Each day's nutrient bounds match the daily DRI (same as the 1-day
LP). Rotation ≤ K days per food is the *only* weekly coupling.

Example:
    # Solve single-day LP first to identify "good" foods
    day_model, day_vars, _ = build_model(food_info, food_matches, nutrition)
    obj, primals, _, _ = solve(day_model)

    # Expand the food pool to include the daily optimum + 50 more cheap options
    pool = preselect_foods(food_info, primals, extra_count=50)

    # Build and solve the weekly MILP
    weekly = build_weekly_model(
        {f: food_info[f] for f in pool},
        {f: food_matches[f] for f in pool},
        nutrition,
        days=7, max_days_per_food=3, min_serving_units=0.30,
    )
    weekly.model.optimize()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import parse_bound
from .model import SPARSE_NUTRIENT_THRESHOLD, _safe_name

GRAMS_PER_LITER = 998
DEFAULT_MAX_DAYS_PER_FOOD = 3
DEFAULT_MIN_SERVING_UNITS = 0.30    # 30g/day
DEFAULT_VAR_UB = 4.0                # 400g/day per food
WEEKLY_BIG_M = DEFAULT_VAR_UB


@dataclass
class WeeklyModel:
    """Container for a built weekly MILP and its per-cell variables."""
    model: Any                                  # optlang.Model
    x: dict[tuple[str, int], Any]               # continuous f[food, day]
    y: dict[tuple[str, int], Any]               # binary y[food, day]


def preselect_foods(
    food_info: dict,
    daily_primals: dict[str, float],
    extra_count: int = 50,
) -> list[str]:
    """Return a candidate food pool for the weekly solve.

    Seeds with foods the single-day LP already uses (they're known
    cost-effective), then fills in the cheapest remaining foods by
    `price_per_100g_edible` to reach at least `len(seed) + extra_count`.
    """
    seed = {_unslug(k, food_info) for k in daily_primals if _unslug(k, food_info)}
    # Sort remaining foods by their $/100g-edible (price/yield/4.54)
    remaining = [
        (f, info["price"] / max(info.get("yield", 1.0), 0.01) / 4.54)
        for f, info in food_info.items()
        if f not in seed
    ]
    remaining.sort(key=lambda x: x[1])
    pool = list(seed) + [f for f, _ in remaining[:extra_count]]
    return pool


def _unslug(safe_key: str, food_info: dict) -> str | None:
    """Reverse _safe_name to find the original food_info key (best-effort)."""
    for food in food_info:
        if _safe_name(food) == safe_key:
            return food
    return None


def build_weekly_model(
    food_info: dict,
    food_matches: dict,
    nutrition: dict,
    days: int = 7,
    max_days_per_food: int = DEFAULT_MAX_DAYS_PER_FOOD,
    min_serving_units: float = DEFAULT_MIN_SERVING_UNITS,
    var_ub: float = DEFAULT_VAR_UB,
    include_volume: bool = False,
) -> WeeklyModel:
    """Build the weekly MILP per the module docstring.

    Returns a WeeklyModel bundling the optlang Model and the
    (continuous, binary) variable dicts keyed on (food_name, day_index).
    """
    import optlang

    model = optlang.Model(name=f"weekly_{days}d")

    x: dict[tuple[str, int], optlang.Variable] = {}
    y: dict[tuple[str, int], optlang.Variable] = {}
    for food in food_info:
        safe = _safe_name(food)
        for d in range(days):
            x[(food, d)] = optlang.Variable(
                f"{safe}_d{d}", lb=0, ub=var_ub, type="continuous"
            )
            y[(food, d)] = optlang.Variable(
                f"use_{safe}_d{d}", lb=0, ub=1, type="binary"
            )
    model.add(list(x.values()))
    model.add(list(y.values()))

    # --- Semi-continuous gating: x[f,d] ∈ {0} ∪ [min_serving, var_ub] ---
    for (food, d), xv in x.items():
        yv = y[(food, d)]
        model.add(optlang.Constraint(
            xv - WEEKLY_BIG_M * yv, ub=0,
            name=f"gate_ub_{_safe_name(food)}_d{d}",
        ))
        model.add(optlang.Constraint(
            xv - min_serving_units * yv, lb=0,
            name=f"gate_lb_{_safe_name(food)}_d{d}",
        ))

    # --- Per-food rotation cap across days ---
    for food in food_info:
        model.add(optlang.Constraint(
            sum(y[(food, d)] for d in range(days)),
            ub=max_days_per_food,
            name=f"rot_{_safe_name(food)}",
        ))

    # --- Per-day nutrient constraints (identical DRI each day) ---
    support = {
        n: sum(1 for f in food_info if n in food_matches.get(f, {}))
        for n in nutrition
    }
    for d in range(days):
        for nutrient, content in nutrition.items():
            if support[nutrient] < SPARSE_NUTRIENT_THRESHOLD:
                continue
            lb = parse_bound(content["low_bound"])
            ub = parse_bound(content["high_bound"])
            expr_terms = []
            for food in food_info:
                if nutrient not in food_matches.get(food, {}):
                    continue
                amount = food_matches[food][nutrient]
                if nutrient == "Total Water":
                    amount /= GRAMS_PER_LITER
                expr_terms.append(amount * x[(food, d)])
            if not expr_terms:
                continue
            expr = sum(expr_terms)
            c_lb = lb if lb != float("inf") else None
            c_ub = ub if ub != float("inf") else None
            model.add(optlang.Constraint(
                expr, lb=c_lb, ub=c_ub,
                name=f"{_safe_name(nutrient)}_d{d}",
            ))

    # --- Volume constraint (optional, off by default for weekly) ---
    if include_volume:
        for d in range(days):
            vol_expr = sum(
                info["cupEQ"] * x[(food, d)]
                for food, info in food_info.items()
            )
            model.add(optlang.Constraint(
                vol_expr, lb=5, ub=20, name=f"volume_d{d}",
            ))

    # --- Objective: minimize total weekly cost ---
    obj_terms = []
    for (food, d), xv in x.items():
        info = food_info[food]
        cost_per_100g = info["price"] / max(info.get("yield", 1.0), 0.01) / 4.54
        obj_terms.append(cost_per_100g * xv)
    model.objective = optlang.Objective(sum(obj_terms), direction="min")

    return WeeklyModel(model=model, x=x, y=y)


def extract_weekly_solution(weekly: WeeklyModel) -> dict[int, dict[str, float]]:
    """Return {day_index: {food_name: grams}} from a solved WeeklyModel.

    Caller is responsible for having invoked `weekly.model.optimize()`.
    """
    out: dict[int, dict[str, float]] = {}
    for (food, d), xv in weekly.x.items():
        val = xv.primal
        if val is None or val < 1e-6:
            continue
        out.setdefault(d, {})[food] = val * 100  # back to grams
    return out


def jaccard_similarity(day_a: dict[str, float], day_b: dict[str, float]) -> float:
    """Ingredient-set Jaccard similarity between two days.

    |A ∩ B| / |A ∪ B| where A and B are the food-name sets. Returns
    0.0 when both days are empty (treat as maximally dissimilar rather
    than undefined).
    """
    a = set(day_a)
    b = set(day_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def cluster_days_for_leftovers(
    per_day: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    """Reorder days so adjacent days share the most ingredients.

    Post-solve only: does NOT change any food quantities or the total
    weekly cost. Just renumbers days so that leftover cooked portions
    flow naturally from one day to the next.

    Formulation: maximize sum_{i=0}^{n-2} jaccard(day_perm[i], day_perm[i+1]).
    For n=7 days, 7! = 5040 permutations — trivially enumerable.
    Day 1 of the reordered sequence is fixed as the original day 0 to
    remove the rotational-symmetry degeneracy (the problem is a path,
    not a cycle).

    Returns a new dict keyed from 0 to n-1 with foods unchanged per day,
    just renumbered so output[0] corresponds to whatever original day
    anchors best.
    """
    from itertools import permutations

    day_indices = sorted(per_day)
    if len(day_indices) <= 2:
        return dict(per_day)

    def total_adjacent_similarity(order: tuple[int, ...]) -> float:
        return sum(
            jaccard_similarity(per_day[order[i]], per_day[order[i + 1]])
            for i in range(len(order) - 1)
        )

    # Enumerate every ordering; pick the one with maximum adjacent similarity.
    # For 7 days this is 5040 orderings; trivial.
    best_order = max(permutations(day_indices), key=total_adjacent_similarity)
    # Renumber so consecutive day indices reflect the optimal order.
    return {new_d: per_day[old_d] for new_d, old_d in enumerate(best_order)}
