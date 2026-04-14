"""Structured LLM meal-plan generation with mass-conservation validation.

The Figure 2 failure mode in the report (corn double-counted across meals)
was a free-form prompt symptom. Here we:

1. Define a strict JSON schema for the LLM's response.
2. Validate every call: the sum of `grams` for each food across all meals
   must equal the LP-produced total, ±`tolerance_g` grams.
3. On mismatch, craft an error message naming the specific food and the
   delta, feed it back to the LLM, and retry.

This module does not call the LLM directly — it provides `MealPlanSchema`,
`validate_plan()`, and `format_retry_message()` that the caller uses
alongside their Anthropic/OpenAI client of choice.
"""
from __future__ import annotations

from dataclasses import dataclass


MEAL_PLAN_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["meals"],
    "properties": {
        "meals": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "ingredients"],
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ["breakfast", "lunch", "dinner", "snack"],
                    },
                    "cooking_instructions": {"type": "string"},
                    "ingredients": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["food", "grams"],
                            "properties": {
                                "food": {"type": "string"},
                                "grams": {"type": "number", "minimum": 0},
                            },
                        },
                    },
                },
            },
        }
    },
}


@dataclass(frozen=True)
class Discrepancy:
    food: str
    lp_grams: float
    plan_grams: float

    @property
    def delta(self) -> float:
        return self.plan_grams - self.lp_grams


def aggregate_plan_totals(plan: dict) -> dict[str, float]:
    """Sum grams per food across all meals in the LLM plan."""
    totals: dict[str, float] = {}
    for meal in plan.get("meals", []):
        for ing in meal.get("ingredients", []):
            totals[ing["food"]] = totals.get(ing["food"], 0.0) + ing["grams"]
    return totals


def validate_plan(
    plan: dict, lp_totals: dict[str, float], tolerance_g: float = 2.0
) -> list[Discrepancy]:
    """Compare aggregated plan totals to LP totals.

    Returns a list of Discrepancy for every food whose plan total differs
    from the LP total by more than `tolerance_g`. Foods in the LP but
    absent from the plan are reported (plan_grams=0); foods in the plan
    but absent from the LP are reported (lp_grams=0).
    """
    plan_totals = aggregate_plan_totals(plan)
    all_foods = set(plan_totals) | set(lp_totals)
    discrepancies = []
    for food in sorted(all_foods):
        lp = lp_totals.get(food, 0.0)
        plan_g = plan_totals.get(food, 0.0)
        if abs(plan_g - lp) > tolerance_g:
            discrepancies.append(Discrepancy(food, lp, plan_g))
    return discrepancies


def format_retry_message(discrepancies: list[Discrepancy]) -> str:
    lines = ["The meal plan you returned does not match the LP output. Please fix:"]
    for d in discrepancies:
        if d.lp_grams == 0:
            lines.append(f"  - '{d.food}': not in LP output (you included {d.plan_grams}g)")
        elif d.plan_grams == 0:
            lines.append(f"  - '{d.food}': missing (LP requires {d.lp_grams}g)")
        else:
            lines.append(
                f"  - '{d.food}': you allocated {d.plan_grams}g total across meals but LP output is {d.lp_grams}g (delta {d.delta:+.1f}g)"
            )
    return "\n".join(lines)
