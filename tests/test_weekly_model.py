"""Tests for the weekly LP with rotation variety."""
import pytest

pytest.importorskip("optlang")
import optlang

from diet_opt.weekly_model import (
    DEFAULT_MAX_DAYS_PER_FOOD,
    build_weekly_model,
    extract_weekly_solution,
    preselect_foods,
)


def _toy_food_info():
    """Small pool with diverse nutrient profiles so a 7-day rotation is possible."""
    return {
        "carrots":   {"price": 0.70, "yield": 0.92, "cupEQ": 2.0},
        "brown_rice":{"price": 1.00, "yield": 1.00, "cupEQ": 3.0},
        "beans":     {"price": 1.20, "yield": 1.00, "cupEQ": 2.0},
        "spinach":   {"price": 2.00, "yield": 0.90, "cupEQ": 2.5},
        "peanuts":   {"price": 1.80, "yield": 0.95, "cupEQ": 3.5},
        "eggs":      {"price": 2.50, "yield": 1.00, "cupEQ": 2.0},
        "tofu":      {"price": 1.50, "yield": 1.00, "cupEQ": 2.0},
        "cod_fish":  {"price": 3.00, "yield": 1.00, "cupEQ": 2.0},
    }


def _toy_food_matches():
    return {
        "carrots":   {"Energy": 41,  "Protein": 0.9, "Iron": 0.3},
        "brown_rice":{"Energy": 370, "Protein": 7.5, "Iron": 1.5},
        "beans":     {"Energy": 347, "Protein": 21,  "Iron": 5.0},
        "spinach":   {"Energy": 23,  "Protein": 2.9, "Iron": 2.7},
        "peanuts":   {"Energy": 567, "Protein": 26,  "Iron": 4.6},
        "eggs":      {"Energy": 155, "Protein": 13,  "Iron": 1.2},
        "tofu":      {"Energy": 144, "Protein": 15,  "Iron": 1.0},
        "cod_fish":  {"Energy": 82,  "Protein": 18,  "Iron": 0.4},
    }


def _toy_nutrition():
    # Very relaxed bounds so feasibility is easy on a small pool.
    return {
        "Energy":  {"low_bound": 1500, "high_bound": 3500, "units": "kcal"},
        "Protein": {"low_bound": 50,   "high_bound": 200,  "units": "grams"},
        "Iron":    {"low_bound": 5,    "high_bound": 50,   "units": "mg"},
    }


def _enough_support(nutrition, food_matches, thresh=6):
    """_sparse_nutrients skips nutrients with <6 foods; our toy has 8."""
    return all(
        sum(1 for f in food_matches if n in food_matches[f]) >= thresh
        for n in nutrition
    )


def test_build_weekly_model_creates_correct_variable_count():
    fi, fm, n = _toy_food_info(), _toy_food_matches(), _toy_nutrition()
    assert _enough_support(n, fm)
    weekly = build_weekly_model(fi, fm, n, days=7, max_days_per_food=3)
    # 8 foods × 7 days continuous + 8 × 7 binary = 112 vars
    assert len(weekly.x) == 56
    assert len(weekly.y) == 56
    # All vars were added to the model
    assert len(weekly.model.variables) == 112


def test_weekly_solve_produces_unique_daily_compositions():
    fi, fm, n = _toy_food_info(), _toy_food_matches(), _toy_nutrition()
    weekly = build_weekly_model(fi, fm, n, days=7, max_days_per_food=3)
    try:
        weekly.model.configuration.lp_method = "simplex"
    except Exception:
        pass
    weekly.model.optimize()
    daily = extract_weekly_solution(weekly)
    assert len(daily) == 7
    # Each day has at least one food
    for d in range(7):
        assert daily.get(d), f"day {d} empty"
    # With max_days_per_food=3, each food appears on ≤3 distinct days
    from collections import Counter
    food_days = Counter()
    for d, foods in daily.items():
        for food in foods:
            food_days[food] += 1
    for food, count in food_days.items():
        assert count <= 3, f"{food} appears on {count} days (cap is 3)"


def test_rotation_cap_forces_food_variety():
    """With cap=2 across 4 days, at least 2 foods must be used (else cap exceeded)."""
    fi, fm, n = _toy_food_info(), _toy_food_matches(), _toy_nutrition()
    weekly = build_weekly_model(fi, fm, n, days=4, max_days_per_food=2)
    weekly.model.optimize()
    daily = extract_weekly_solution(weekly)
    # Any single food appears at most twice
    from collections import Counter
    food_days = Counter()
    for d, foods in daily.items():
        for food in foods:
            food_days[food] += 1
    assert max(food_days.values()) <= 2


def test_preselect_foods_seeds_with_daily_solution():
    fi = {"carrots": {"price": 1.0, "yield": 1.0, "cupEQ": 2},
          "beans": {"price": 1.2, "yield": 1.0, "cupEQ": 2},
          "caviar": {"price": 50, "yield": 1.0, "cupEQ": 0.1}}
    daily = {"carrots": 3.5}   # carrots was in the single-day solution
    pool = preselect_foods(fi, daily, extra_count=1)
    assert "carrots" in pool   # seeded from daily solution
    # Extra fills with cheapest non-seed food
    assert "beans" in pool or "caviar" in pool
    # Cheapest extras first (beans < caviar by price)
    non_seed = [f for f in pool if f != "carrots"]
    assert "beans" in non_seed


def test_min_serving_forces_each_served_portion_above_threshold():
    fi, fm, n = _toy_food_info(), _toy_food_matches(), _toy_nutrition()
    weekly = build_weekly_model(fi, fm, n, days=4, max_days_per_food=2,
                                min_serving_units=0.50)
    weekly.model.optimize()
    daily = extract_weekly_solution(weekly)
    for d, foods in daily.items():
        for food, grams in foods.items():
            assert grams >= 50 - 0.1, f"{food} on day {d} = {grams}g < 50g"


def test_jaccard_similarity_identical_days():
    from diet_opt.weekly_model import jaccard_similarity
    a = {"carrots": 100, "rice": 200}
    b = {"carrots": 50, "rice": 150}
    assert jaccard_similarity(a, b) == 1.0


def test_jaccard_similarity_disjoint_days():
    from diet_opt.weekly_model import jaccard_similarity
    a = {"carrots": 100}
    b = {"tofu": 50}
    assert jaccard_similarity(a, b) == 0.0


def test_jaccard_similarity_partial_overlap():
    from diet_opt.weekly_model import jaccard_similarity
    a = {"carrots": 100, "rice": 200}
    b = {"carrots": 50, "beans": 100}
    # {carrots} ∩ / {carrots, rice, beans} = 1/3
    assert abs(jaccard_similarity(a, b) - 1/3) < 1e-9


def test_jaccard_empty_days():
    from diet_opt.weekly_model import jaccard_similarity
    assert jaccard_similarity({}, {}) == 0.0


def test_cluster_days_places_similar_days_adjacent():
    from diet_opt.weekly_model import cluster_days_for_leftovers, jaccard_similarity
    # 4 days: day 0 and day 3 share carrots+rice; day 1 and day 2 share beans.
    # Between disjoint clusters ONE transition pair will have zero overlap;
    # ≥2 of the 3 adjacent pairs should share ingredients.
    per_day = {
        0: {"carrots": 400, "rice": 200},
        1: {"beans": 300, "tofu": 100},
        2: {"beans": 300, "eggs": 80},
        3: {"carrots": 400, "rice": 200},
    }
    ordered = cluster_days_for_leftovers(per_day)

    # Total adjacent-similarity should beat the original ordering.
    def adj_total(d):
        keys = sorted(d)
        return sum(jaccard_similarity(d[keys[i]], d[keys[i+1]])
                   for i in range(len(keys) - 1))
    assert adj_total(ordered) > adj_total(per_day)

    # At least 2 of 3 adjacent pairs should share an ingredient.
    keys = sorted(ordered)
    shared_pairs = sum(
        1 for i in range(len(keys) - 1)
        if set(ordered[keys[i]]) & set(ordered[keys[i+1]])
    )
    assert shared_pairs >= 2


def test_cluster_days_preserves_total_amounts():
    from diet_opt.weekly_model import cluster_days_for_leftovers
    per_day = {
        0: {"a": 100, "b": 50},
        1: {"b": 40, "c": 75},
        2: {"a": 80, "c": 60},
    }
    ordered = cluster_days_for_leftovers(per_day)
    # Same total grams per food across the week
    def totals(d):
        out = {}
        for day_foods in d.values():
            for food, g in day_foods.items():
                out[food] = out.get(food, 0) + g
        return out
    assert totals(per_day) == totals(ordered)


def test_cluster_days_noop_for_small_inputs():
    from diet_opt.weekly_model import cluster_days_for_leftovers
    assert cluster_days_for_leftovers({}) == {}
    single = {0: {"a": 100}}
    assert cluster_days_for_leftovers(single) == single
    pair = {0: {"a": 100}, 1: {"b": 50}}
    assert cluster_days_for_leftovers(pair) == pair


def test_extract_empty_when_model_has_no_solution():
    model = optlang.Model()
    x = {("a", 0): optlang.Variable("a_d0", lb=0, ub=1)}
    y = {("a", 0): optlang.Variable("y_a_d0", lb=0, ub=1, type="binary")}
    model.add([x[("a", 0)], y[("a", 0)]])
    model.objective = optlang.Objective(x[("a", 0)], direction="min")
    model.optimize()
    from diet_opt.weekly_model import WeeklyModel, extract_weekly_solution
    wm = WeeklyModel(model=model, x=x, y=y)
    result = extract_weekly_solution(wm)
    assert result == {}
