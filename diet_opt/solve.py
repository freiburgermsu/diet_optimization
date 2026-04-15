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
