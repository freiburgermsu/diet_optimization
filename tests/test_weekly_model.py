"""Tests for the weekly MILP with rotation variety."""
import pytest

pytest.importorskip("highspy")

from diet_opt.weekly_model import (
    DEFAULT_MAX_DAYS_PER_FOOD,
    WeeklyModel,
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
    # Relaxed bounds so feasibility is easy on a small pool.
    return {
        "Energy":  {"low_bound": 1500, "high_bound": 3500, "units": "kcal"},
        "Protein": {"low_bound": 50,   "high_bound": 200,  "units": "grams"},
        "Iron":    {"low_bound": 5,    "high_bound": 50,   "units": "mg"},
    }


def test_build_weekly_model_creates_correct_variable_count():
    fi, fm, n = _toy_food_info(), _toy_food_matches(), _toy_nutrition()
    weekly = build_weekly_model(fi, fm, n, days=7, max_days_per_food=3)
    # 8 foods × 7 days continuous + 8 × 7 binary = 112 vars
    assert len(weekly.x) == 56
    assert len(weekly.y) == 56
    # HiGHS reports total column count
    assert weekly.model.numVariables == 112


def test_weekly_solve_produces_unique_daily_compositions():
    fi, fm, n = _toy_food_info(), _toy_food_matches(), _toy_nutrition()
    # build_weekly_model invokes the solver (HiGHS.minimize) internally
    weekly = build_weekly_model(fi, fm, n, days=7, max_days_per_food=3)
    daily = extract_weekly_solution(weekly)
    assert len(daily) == 7
    for d in range(7):
        assert daily.get(d), f"day {d} empty"
    from collections import Counter
    food_days = Counter()
    for d, foods in daily.items():
        for food in foods:
            food_days[food] += 1
    for food, count in food_days.items():
        assert count <= 3, f"{food} appears on {count} days (cap is 3)"


def test_rotation_cap_forces_food_variety():
    fi, fm, n = _toy_food_info(), _toy_food_matches(), _toy_nutrition()
    weekly = build_weekly_model(fi, fm, n, days=4, max_days_per_food=2)
    daily = extract_weekly_solution(weekly)
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
    daily = {"carrots": 3.5}
    pool = preselect_foods(fi, daily, extra_count=1)
    assert "carrots" in pool
    non_seed = [f for f in pool if f != "carrots"]
    assert "beans" in non_seed


def test_min_serving_forces_each_served_portion_above_threshold():
    fi, fm, n = _toy_food_info(), _toy_food_matches(), _toy_nutrition()
    weekly = build_weekly_model(fi, fm, n, days=4, max_days_per_food=2,
                                min_serving_units=0.50)
    daily = extract_weekly_solution(weekly)
    for d, foods in daily.items():
        for food, grams in foods.items():
            assert grams >= 50 - 0.1, f"{food} on day {d} = {grams}g < 50g"


def test_default_max_days_per_food_is_four():
    """Updated default so foods can repeat up to 4× per week."""
    assert DEFAULT_MAX_DAYS_PER_FOOD == 4


# --- Jaccard / clustering helpers ---

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
    assert abs(jaccard_similarity(a, b) - 1/3) < 1e-9


def test_jaccard_empty_days():
    from diet_opt.weekly_model import jaccard_similarity
    assert jaccard_similarity({}, {}) == 0.0


def test_cluster_days_improves_adjacent_similarity():
    from diet_opt.weekly_model import cluster_days_for_leftovers, jaccard_similarity
    per_day = {
        0: {"carrots": 400, "rice": 200},
        1: {"beans": 300, "tofu": 100},
        2: {"beans": 300, "eggs": 80},
        3: {"carrots": 400, "rice": 200},
    }
    ordered = cluster_days_for_leftovers(per_day)

    def adj_total(d):
        keys = sorted(d)
        return sum(jaccard_similarity(d[keys[i]], d[keys[i+1]])
                   for i in range(len(keys) - 1))
    assert adj_total(ordered) > adj_total(per_day)
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
