from diet_opt.weekly import (
    DAYS_PER_WEEK,
    build_weekly_variables,
    distinct_foods_across_week,
    is_per_day,
    load_cadence,
    scale_bounds_for_cadence,
    weekly_cost_objective,
)


def test_cadence_loads_and_classifies():
    cadence = load_cadence()
    assert cadence["Sodium"] == "daily"
    assert cadence["Iron"] == "weekly"


def test_is_per_day_defaults_to_weekly_for_unknown():
    assert is_per_day("Unobtainium", {}) is False
    assert is_per_day("Sodium", {"Sodium": "daily"}) is True


def test_scale_bounds_daily_unchanged():
    lb, ub, cad = scale_bounds_for_cadence(100, 2300, "Sodium", {"Sodium": "daily"})
    assert (lb, ub, cad) == (100, 2300, "daily")


def test_scale_bounds_weekly_multiplied():
    lb, ub, cad = scale_bounds_for_cadence(25, 35, "Fiber", {"Fiber": "weekly"}, days=7)
    assert lb == 175
    assert ub == 245
    assert cad == "weekly"


def test_build_weekly_variables_cardinality():
    food_info = {"Carrots": {}, "Pinto Beans": {}}
    vars_ = build_weekly_variables(food_info)
    assert len(vars_) == 2 * DAYS_PER_WEEK
    # Spaces replaced with underscores in keys
    assert ("Pinto_Beans", 0) in vars_
    assert vars_[("Pinto_Beans", 0)]["name"] == "Pinto_Beans__d0"


def test_weekly_cost_objective_has_one_term_per_variable():
    food_info = {"Apple": {"price": 1.00, "yield": 0.92}}
    vars_ = build_weekly_variables(food_info, days=7)
    terms = weekly_cost_objective(vars_, food_info)
    assert len(terms) == 7  # one per (food, day)
    # Coefficient is price/yield/4.54 per scalar model convention
    expected = 1.00 / 0.92 / 4.54
    for _name, coef in terms:
        assert abs(coef - expected) < 1e-9


def test_distinct_foods_counts_above_threshold():
    solution = {
        ("Apples", 0): 1.0,   # 100g → counts
        ("Apples", 1): 0.5,   # 50g → counts but same food
        ("Flax", 0): 0.1,     # 10g → below 20g threshold
        ("Rice", 3): 2.0,     # 200g → counts
    }
    assert distinct_foods_across_week(solution, min_grams=20) == 2
