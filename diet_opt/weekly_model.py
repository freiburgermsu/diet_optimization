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
DEFAULT_MAX_DAYS_PER_FOOD = 4
DEFAULT_MIN_SERVING_UNITS = 0.30    # 30g/day
DEFAULT_VAR_UB = 4.0                # 400g/day per food
WEEKLY_BIG_M = DEFAULT_VAR_UB


@dataclass
class WeeklyModel:
    """Container for a built weekly MILP and its per-cell variables.

    With the HiGHS backend, `x` and `y` map (food, day) → int column
    index (not variable objects). Use `model.getSolution().col_value`
    to read the values.
    """
    model: Any                                  # highspy.Highs
    x: dict[tuple[str, int], Any]               # HiGHS column index for x
    y: dict[tuple[str, int], Any]               # HiGHS column index for y


def preselect_foods(
    food_info: dict,
    daily_primals: dict[str, float],
    extra_count: int = 50,
) -> list[str]:
    """Return a candidate food pool for the weekly solve (price-only).

    Seeds with foods the single-day LP already uses (they're known
    cost-effective), then fills in the cheapest remaining foods by
    `price_per_100g_edible` to reach at least `len(seed) + extra_count`.

    See `preselect_foods_by_profile` for nutrient-aware scoring that
    biases the pool toward the user's priorities (athlete → protein,
    elder → calcium, etc.).
    """
    seed = {_unslug(k, food_info) for k in daily_primals if _unslug(k, food_info)}
    remaining = [
        (f, info["price"] / max(info.get("yield", 1.0), 0.01) / 4.54)
        for f, info in food_info.items()
        if f not in seed
    ]
    remaining.sort(key=lambda x: x[1])
    pool = list(seed) + [f for f, _ in remaining[:extra_count]]
    return pool


# Predefined nutrient emphasis templates. Weights > 1 boost a nutrient's
# contribution to a food's score. Unlisted nutrients default to weight 1.
EMPHASIS_TEMPLATES: dict[str, dict[str, float]] = {
    "budget": {},  # no emphasis — weights all 1.0 (closest to price-only)
    "athlete": {
        "Protein": 3.0,
        "Energy": 2.0,
        "Iron": 2.0,
        "Magnesium": 2.0,
        "Potassium": 2.0,
        "Sodium": 1.5,    # replacing what's lost in sweat
    },
    "recovery": {
        "Protein": 3.0,
        "Vitamin C": 2.0,
        "Iron": 2.0,
        "Zinc": 2.0,
    },
    "older": {
        "Calcium": 3.0,
        "Vitamin D": 3.0,
        "Vitamin B12": 3.0,
        "Fiber": 2.0,
        "Total Fiber": 2.0,
        "Protein": 2.0,
        "Vitamin B6": 1.5,
    },
    "iron_deficient": {
        "Iron": 4.0,
        "Vitamin C": 2.0,      # aids iron absorption
        "Vitamin B12": 2.0,
        "Folate": 2.0,
    },
    "pregnancy": {
        "Folate": 3.0,
        "Iron": 3.0,
        "Calcium": 2.0,
        "Protein": 2.0,
        "Choline": 2.0,
        "Vitamin D": 2.0,
    },
}


def profile_to_emphasis(
    sex: str | None = None,
    age: int | None = None,
    activity: str | None = None,
) -> str:
    """Map a user profile to the most fitting emphasis template.

    Ordering of checks matters: more specific first. Returns the
    template name; caller looks it up in EMPHASIS_TEMPLATES.
    """
    if activity in ("active", "very_active"):
        return "athlete"
    if age is not None and age >= 60:
        return "older"
    if sex == "female" and age is not None and 15 <= age < 51:
        return "iron_deficient"
    return "budget"


def score_foods(
    food_info: dict,
    food_matches: dict,
    nutrition: dict,
    emphasis: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Score each food by (weighted nutrient density) / price.

    Formula:
        score = Σ_n weight_n * (nutrient_per_g_n / DRI_lb_n) / price_per_g

    High score = cheap AND dense in the emphasized nutrients. Foods
    with unknown or zero price are scored 0 (filtered out).

    Returns a list of (food_name, score) sorted by score descending.
    """
    from .data import parse_bound

    emphasis = emphasis or {}
    scores: list[tuple[str, float]] = []

    # Precompute DRI lower bounds for normalization
    dri_lb = {}
    for nutrient, content in nutrition.items():
        lb = parse_bound(content.get("low_bound", 0))
        if lb and lb > 0 and lb != float("inf"):
            dri_lb[nutrient] = lb

    for food, info in food_info.items():
        price_per_g = info["price"] / max(info.get("yield", 1.0), 0.01) / 454.0
        if price_per_g <= 0:
            continue
        nutrients = food_matches.get(food, {})
        numer = 0.0
        for nutrient, amount in nutrients.items():
            if nutrient not in dri_lb:
                continue
            weight = emphasis.get(nutrient, 1.0)
            # amount is per-100g; normalize by DRI lb (mass per day)
            numer += weight * (amount / dri_lb[nutrient])
        score = numer / price_per_g if price_per_g > 0 else 0.0
        scores.append((food, score))

    scores.sort(key=lambda kv: -kv[1])
    return scores


def preselect_foods_by_profile(
    food_info: dict,
    food_matches: dict,
    nutrition: dict,
    daily_primals: dict[str, float] | None = None,
    *,
    emphasis: dict[str, float] | str | None = None,
    extra_count: int = 50,
) -> list[str]:
    """Profile-aware pool selection.

    Always seeds with foods the 1-day LP chose. Then fills `extra_count`
    additional slots by *nutrient-per-dollar* score, using the
    `emphasis` weights (string name → template, or dict → custom).

    When emphasis is None or "budget", falls back to price-only ordering
    via `preselect_foods` (for backward compatibility).
    """
    if isinstance(emphasis, str):
        emphasis = EMPHASIS_TEMPLATES.get(emphasis, {})

    if not emphasis:
        return preselect_foods(food_info, daily_primals or {}, extra_count)

    seed = {_unslug(k, food_info) for k in (daily_primals or {}) if _unslug(k, food_info)}
    ranked = score_foods(food_info, food_matches, nutrition, emphasis)
    extras: list[str] = []
    for food, _score in ranked:
        if food in seed:
            continue
        extras.append(food)
        if len(extras) >= extra_count:
            break
    return list(seed) + extras


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
    time_limit_sec: float = 120.0,
    mip_gap: float = 0.01,
) -> WeeklyModel:
    """Build the weekly MILP per the module docstring.

    Uses HiGHS directly (not via optlang) because optlang's HiGHS
    interface doesn't exist in v1.x and GLPK's MIP solver was too slow
    on this formulation (600+ variable MILPs would time out).

    Returns a WeeklyModel bundling the HiGHS instance and dicts mapping
    (food, day) to column indices for later extraction.
    """
    import highspy

    h = highspy.Highs()
    h.silent()
    # Cap solve time so the branch-and-bound terminates quickly with the
    # incumbent (best-feasible-so-far) solution. Without this, live runs
    # can take hours chasing the last fraction of a percent of optimality.
    try:
        h.setOptionValue("time_limit", float(time_limit_sec))
        # 1% MIP gap: stop once the best feasible is within 1% of the
        # LP relaxation bound. Usually hits this within seconds.
        h.setOptionValue("mip_rel_gap", float(mip_gap))
    except Exception:
        pass

    # Variable registries: map (food, day) → highspy highs_var object
    x_vars: dict[tuple[str, int], Any] = {}   # continuous grams (100g units)
    y_vars: dict[tuple[str, int], Any] = {}   # binary "served that day"

    for food in food_info:
        for d in range(days):
            x_vars[(food, d)] = h.addVariable(lb=0, ub=var_ub)
    for food in food_info:
        for d in range(days):
            y_vars[(food, d)] = h.addBinary()

    # --- Semi-continuous gating: x ∈ {0} ∪ [min_serving, var_ub] ---
    for (food, d), xv in x_vars.items():
        yv = y_vars[(food, d)]
        h.addConstr(xv - WEEKLY_BIG_M * yv <= 0)
        h.addConstr(xv - min_serving_units * yv >= 0)

    # --- Rotation cap: each food appears ≤ max_days_per_food days ---
    for food in food_info:
        h.addConstr(sum(y_vars[(food, d)] for d in range(days)) <= max_days_per_food)

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
                expr_terms.append(amount * x_vars[(food, d)])
            if not expr_terms:
                continue
            expr = sum(expr_terms)
            if lb != float("inf"):
                h.addConstr(expr >= lb)
            if ub != float("inf"):
                h.addConstr(expr <= ub)

    # --- Volume constraint (optional) ---
    if include_volume:
        for d in range(days):
            vol_expr = sum(
                info["cupEQ"] * x_vars[(food, d)]
                for food, info in food_info.items()
            )
            h.addConstr(vol_expr >= 5)
            h.addConstr(vol_expr <= 20)

    # --- Objective: minimize total weekly cost ---
    obj_terms = []
    for (food, d), xv in x_vars.items():
        info = food_info[food]
        cost_per_100g = info["price"] / max(info.get("yield", 1.0), 0.01) / 4.54
        obj_terms.append(cost_per_100g * xv)
    h.minimize(sum(obj_terms))

    return WeeklyModel(model=h, x=x_vars, y=y_vars)


def extract_weekly_solution(weekly: WeeklyModel) -> dict[int, dict[str, float]]:
    """Return {day_index: {food_name: grams}} from a solved WeeklyModel.

    Caller must have invoked `weekly.model.minimize(...)` / `maximize(...)`
    (the HiGHS solve is triggered automatically by those calls).
    """
    h = weekly.model
    out: dict[int, dict[str, float]] = {}
    for (food, d), var in weekly.x.items():
        val = h.variableValue(var)
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
