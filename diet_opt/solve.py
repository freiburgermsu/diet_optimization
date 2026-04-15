"""Solve the LP and extract primals, constraint values, and duals."""
from __future__ import annotations

from dataclasses import dataclass

DUAL_EPSILON = 1e-6


@dataclass(frozen=True)
class ShadowPrice:
    constraint: str
    bound: str          # "lower" or "upper"
    bound_value: float
    dual: float         # positive magnitude of ∂objective / ∂bound


def solve(model, extract_duals: bool = False):
    """Solve and return (objective, primals, constraint_values, shadow_prices).

    When `extract_duals=False`, runs in exact-arithmetic mode and returns
    an empty shadow_prices list. When True, falls back to simplex — GLPK's
    exact mode does not populate duals.
    """
    # GLPK's "exact" method is iterating-simplex with rational arithmetic; duals
    # aren't populated. Only force "exact" when we don't need duals; fall back
    # to the solver's default (primal simplex) for simplex/dual extraction.
    try:
        if extract_duals:
            model.configuration.lp_method = "simplex"
        else:
            model.configuration.lp_method = "exact"
    except (AttributeError, ValueError):
        pass  # non-GLPK backend; use defaults
    model.optimize()
    primals = {k: v for k, v in model.primal_values.items() if v > 0}
    constraint_values = {
        c.name: {"lb": c.lb, "val": c.primal, "ub": c.ub} for c in model.constraints
    }
    shadow_prices = _extract_duals(model) if extract_duals else []
    return model.objective.value, primals, constraint_values, shadow_prices


def _extract_duals(model) -> list[ShadowPrice]:
    out: list[ShadowPrice] = []
    for c in model.constraints:
        try:
            dual = c.dual
        except Exception:
            continue
        if dual is None or abs(dual) < DUAL_EPSILON:
            continue
        at_lb = c.lb is not None and abs(c.primal - c.lb) < 1e-6
        at_ub = c.ub is not None and abs(c.primal - c.ub) < 1e-6
        if at_lb:
            out.append(ShadowPrice(c.name, "lower", c.lb, abs(dual)))
        elif at_ub:
            out.append(ShadowPrice(c.name, "upper", c.ub, abs(dual)))
    out.sort(key=lambda s: -s.dual)
    return out


def solve_with_min_serving(
    model,
    variables: dict,
    min_serving_units: float,
    extract_duals: bool = False,
    big_m: float = 5.0,
):
    """Enforce a true semi-continuous constraint via MILP.

    Each food variable f_i becomes semi-continuous: f_i = 0 OR f_i ≥ min.
    Implemented with a binary indicator b_i per food and two linked
    constraints:
        f_i ≤ big_m × b_i          (if b_i = 0 then f_i = 0)
        f_i ≥ min_units × b_i      (if b_i = 1 then f_i ≥ min_units)

    This converts the LP into a MILP. GLPK solves it; expect 5-60 sec
    depending on food pool size (610 binaries currently).

    Previous implementation (drop-and-resolve heuristic) cycled on the
    610-food table — dropped foods got replaced by other tiny-quantity
    foods every iteration. MILP gives the globally optimal answer.

    Returns (objective, primals, constraint_values, shadow_prices, iterations)
    where `iterations` is always 1 (kept for API compatibility).
    """
    import optlang

    added_binaries = []
    added_constraints = []
    for food_key, var in variables.items():
        b_name = f"use_{food_key}"
        if b_name in {v.name for v in model.variables}:
            continue  # already added (e.g. re-solve of same model)
        b = optlang.Variable(b_name, lb=0, ub=1, type="binary")
        model.add(b)
        added_binaries.append(b)
        c_ub = optlang.Constraint(var - big_m * b, ub=0, name=f"mincap_ub_{food_key}")
        c_lb = optlang.Constraint(var - min_serving_units * b, lb=0, name=f"mincap_lb_{food_key}")
        model.add(c_ub)
        model.add(c_lb)
        added_constraints.extend([c_ub, c_lb])

    # GLPK's exact mode is LP-only; drop to simplex for the MILP solve.
    try:
        model.configuration.lp_method = "simplex"
    except (AttributeError, ValueError):
        pass

    try:
        model.optimize()
        if getattr(model, "status", None) in ("infeasible", "infeasible_inaccurate"):
            return None, None, None, None, 1
    except Exception:
        return None, None, None, None, 1

    # Filter out the binary indicators from the primals we return.
    primals = {
        k: v for k, v in model.primal_values.items()
        if v > 0 and not k.startswith("use_")
    }
    constraint_values = {
        c.name: {"lb": c.lb, "val": c.primal, "ub": c.ub}
        for c in model.constraints
        if not c.name.startswith("mincap_")
    }
    shadow_prices = _extract_duals(model) if extract_duals else []
    return model.objective.value, primals, constraint_values, shadow_prices, 1


def explain_shadow_prices(shadows: list[ShadowPrice], nutrition: dict, top_k: int = 5) -> list[str]:
    """Render shadow prices as one natural-language line each."""
    lines = []
    for s in shadows[:top_k]:
        key = s.constraint.replace("_", " ")
        unit = nutrition.get(key, {}).get("units", "")
        direction = "Raising" if s.bound == "upper" else "Lowering"
        lines.append(
            f"{s.constraint} is at its {s.bound} bound ({s.bound_value:g} {unit}). "
            f"{direction} it by 1 {unit} would save ${s.dual:.3f}/day."
        )
    return lines
