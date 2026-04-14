"""Solve the LP and extract primals + constraint values.

Duals (shadow prices) are not extracted here — `lp_method = "exact"`
does not populate them on GLPK. See #17 for the dual-extraction path,
which requires switching to simplex mode.
"""
from __future__ import annotations


def solve(model):
    """Solve with exact arithmetic and return (objective, primals, constraint_values)."""
    model.configuration.lp_method = "exact"
    model.optimize()
    primals = {k: v for k, v in model.primal_values.items() if v > 0}
    constraint_values = {
        c.name: {"lb": c.lb, "val": c.primal, "ub": c.ub} for c in model.constraints
    }
    return model.objective.value, primals, constraint_values
