"""Tests for diet_opt.solve.solve_with_min_serving() — MILP version."""
import pytest

pytest.importorskip("optlang")
import optlang

from diet_opt.solve import solve_with_min_serving


def _toy_lp(min_serving_units: float = 0.30):
    """Build a tiny LP: minimize cost of (carrots, trace) subject to
    combined_nutrient ≥ 10 where carrots provides 5/unit and trace 20/unit.
    Without min-serving, the cheapest solution picks lots of carrots.
    With min-serving=30g, the LP must choose between dropping one and
    forcing the other to ≥30g.
    """
    model = optlang.Model(name="toy")
    carrots = optlang.Variable("carrots", lb=0, ub=5, type="continuous")
    trace   = optlang.Variable("trace",   lb=0, ub=5, type="continuous")
    model.add([carrots, trace])
    # Nutrient requirement: 5*carrots + 20*trace >= 10
    c = optlang.Constraint(5 * carrots + 20 * trace, lb=10, name="nutrient")
    model.add(c)
    # Cost: carrots $0.10/unit, trace $1.00/unit
    model.objective = optlang.Objective(0.10 * carrots + 1.00 * trace, direction="min")
    return model, {"carrots": carrots, "trace": trace}


def test_milp_semi_continuous_enforces_threshold():
    """With min=30g (=0.30), neither var can be in (0, 0.30)."""
    model, variables = _toy_lp()
    obj, primals, _cv, _sp, iters = solve_with_min_serving(
        model, variables, min_serving_units=0.30,
    )
    for food, value in primals.items():
        assert value == 0 or value >= 0.30 - 1e-6, f"{food} at {value}"


def test_milp_drops_expensive_food_when_cheap_alternative_exists():
    """Trace is 10× more expensive; solver should zero it out and lean on carrots."""
    model, variables = _toy_lp()
    obj, primals, _cv, _sp, _iters = solve_with_min_serving(
        model, variables, min_serving_units=0.30,
    )
    # Cheap path: carrots = 2 (100% of constraint), trace = 0
    assert primals.get("carrots", 0) >= 1.999
    assert primals.get("trace", 0) == 0


def test_milp_forces_threshold_when_no_alternative():
    """If carrots can't alone satisfy the constraint, trace must be ≥ min."""
    model = optlang.Model(name="forced")
    carrots = optlang.Variable("carrots", lb=0, ub=0.5, type="continuous")  # cap
    trace   = optlang.Variable("trace",   lb=0, ub=5,   type="continuous")
    model.add([carrots, trace])
    # Need 5*carrots + 20*trace >= 10. carrots max 0.5 → 2.5; need trace ≥ 0.375
    model.add(optlang.Constraint(5 * carrots + 20 * trace, lb=10, name="n"))
    model.objective = optlang.Objective(0.10 * carrots + 1.00 * trace, direction="min")
    vars_ = {"carrots": carrots, "trace": trace}

    obj, primals, _cv, _sp, _iters = solve_with_min_serving(model, vars_, 0.30)
    # trace must be at least 0.30 (forced by min-serving), and ≥ 0.375 (feasibility)
    assert primals.get("trace", 0) >= 0.30 - 1e-6
    assert primals.get("trace", 0) >= 0.375 - 1e-6


def test_milp_infeasible_returns_none():
    """Infeasible LP should return all-None tuple."""
    model = optlang.Model(name="infeasible")
    x = optlang.Variable("x", lb=0, ub=1, type="continuous")
    model.add(x)
    # Impossible: x >= 10 AND x <= 1
    model.add(optlang.Constraint(x, lb=10, name="bad"))
    model.objective = optlang.Objective(x, direction="min")
    obj, primals, _cv, _sp, _iters = solve_with_min_serving(
        model, {"x": x}, min_serving_units=0.30,
    )
    assert obj is None
    assert primals is None


def test_milp_result_excludes_binary_indicators():
    """The returned primals dict should not contain the binary use_* vars."""
    model, variables = _toy_lp()
    obj, primals, _cv, _sp, _iters = solve_with_min_serving(
        model, variables, min_serving_units=0.30,
    )
    for key in primals:
        assert not key.startswith("use_")
