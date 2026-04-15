#!/usr/bin/env python3
"""Generate a 1-week meal plan from the LP's optimum_diet output via Claude.

Takes the LP's food totals (7-day grams by food) and asks Claude to
distribute them across 7 days × 4 meals with simple cooking instructions,
a consolidated shopping list, and nutritional commentary.

Features implemented here:
  - One week at a time (7 days × breakfast/lunch/dinner/snack)
  - Simple 2-3 line cooking instructions per meal
  - Varied cuisine across the week by default (Mediterranean, Asian,
    Mexican, Indian, American, Middle Eastern, simple/leftover) — or
    --cuisine-style <name> to force one style
  - Prep-time ceiling via --max-prep-min
  - Batch-cook / leftover suggestions (Sunday cook-ahead)
  - Per-meal nutritional narration ("Wed dinner: 70% daily iron")
  - Consolidated shopping list aggregated across all 7 days
  - Allergen filtering via --blacklist / --whitelist (reuses prefs schema)

Mass-conservation validation: every food's total across the 7 days must
match the LP output within --tolerance-g (default 5g). Retries up to
3× on mismatch, feeding the delta back to Claude.

Usage:
    export ANTHROPIC_API_KEY=...
    uv run --with "anthropic>=0.40" --with pydantic python \\
        scripts/generate_meal_plan.py \\
        --diet optimum_diet.csv \\
        --output meal_plan.json \\
        --days 7
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Literal

try:
    import anthropic
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(
        f"Missing: {e}. Install: pip install 'anthropic>=0.40' pydantic"
    )


DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_CACHE_FILE = Path("cache/meal_plan.json")
DEFAULT_CUISINES = [
    "Mediterranean",
    "Southeast Asian",
    "Mexican",
    "Indian",
    "Plain American",
    "Middle Eastern",
    "Simple / leftover day",
]
DEFAULT_MAX_PREP_MIN = 30


class Ingredient(BaseModel):
    food: str = Field(description="Food name, must exactly match LP output")
    grams: float = Field(ge=0)


class Meal(BaseModel):
    name: Literal["breakfast", "lunch", "dinner", "snack"]
    dish_name: str = Field(description="Short descriptive name, e.g. 'Carrot lentil dal'")
    cuisine: str = Field(description="Cuisine style tag")
    prep_time_min: int = Field(ge=0)
    ingredients: list[Ingredient]
    cooking_instructions: str = Field(
        description="2-3 short sentences covering prep + cook"
    )
    nutritional_highlight: str | None = Field(
        default=None,
        description="One-line callout like 'Covers 65% of daily iron'",
    )


class DayPlan(BaseModel):
    day: int = Field(ge=1, le=7)
    meals: list[Meal]
    leftover_note: str | None = Field(
        default=None,
        description="Batch-cook / reheat / prep-ahead guidance for this day",
    )


class WeeklyMealPlan(BaseModel):
    days: list[DayPlan]
    shopping_list: list[Ingredient] = Field(
        description="Aggregated totals across all 7 days"
    )
    weekly_summary: str = Field(
        description="2-3 sentence overview of the week's cuisines and nutritional arc"
    )


SYSTEM_PROMPT = """You are a meal planner for a diet optimization research tool. A linear-program solver produces a list of foods and daily-equivalent gram totals that satisfy nutritional constraints at minimum cost. Your job: distribute those weekly totals across 7 days of realistic, cookable meals.

## Mass conservation is mandatory

For each food in the LP output, sum the grams across all meals in your plan. That sum MUST equal the LP total for that food (the input is already multiplied by 7 for a week-view). A tolerance of ±5g is acceptable for rounding.

If your plan's totals don't match, the validator will reject it and hand you a specific food + delta. Fix the named discrepancy and try again; don't reshuffle the entire plan.

## Cuisine variety (default)

Unless the user specifies a single cuisine style, give each day a distinct cuisine from: Mediterranean, Southeast Asian, Mexican, Indian, Plain American, Middle Eastern, Simple/leftover. Rotate them so two adjacent days aren't the same.

## Cooking instructions

- 2-3 short sentences per meal
- Plain cookware assumed (pot, pan, oven, knife)
- Specify heat level and rough timing ("medium heat, 8 min")
- Combine steps into sentences rather than numbered lists
- Target: a literate adult who doesn't cook professionally

## Prep time

Respect the user's max-prep-minutes budget (default 30 min/meal). Breakfasts should be faster (<15 min). One dinner per week can be slower (batch-cook meal, up to 60 min) and explicitly flagged as the "prep-ahead" day.

## Leftover / batch-cook strategy

Designate one day (usually Sunday) as the batch-cook day: cook a larger portion that Monday and Tuesday reuse. Note this in the day's `leftover_note`. Similarly, any dish that reheats well (rice, soup, stew, roasted vegetables) should be deliberately scaled up when it appears, with the leftover noted.

## Nutritional highlights

On ~30-50% of meals, add a short `nutritional_highlight` like:
- "High iron — covers ~65% of daily RDA"
- "Complete protein pairing (rice + beans)"
- "Rich in Vitamin A from the carrots"

Don't force it on every meal. Aim for one callout per day roughly.

## Shopping list

Aggregate the totals: for each food across the 7 days, sum grams and return a single entry per food. This is what the user takes to the store.

## Weekly summary

One paragraph (2-3 sentences) describing the week's cuisines and any nutritional arc the user should notice (e.g., "protein anchors rotate between legumes, fish, and eggs; omega-3s come from salmon on Wednesday and flax throughout").

## Ingredient names

Match the LP output's food names EXACTLY. The LP will give you strings like "pinto_beans" (with underscores) or "chicken breast" (with spaces). Use the exact string you were given — don't pluralize, don't capitalize, don't split.

## Output

Valid JSON matching the WeeklyMealPlan schema. No prose outside the JSON.

## Worked example

LP output: `{"brown_rice": 1309, "great_northern_beans": 1162, "carrots": 3479, "yolk_egg": 532, "spinach": 273}`  (grams for the full week)
User request: 7-day plan, varied cuisines, max 30 min prep.

Expected output shape (abbreviated):
```json
{
  "days": [
    {
      "day": 1,
      "meals": [
        {
          "name": "breakfast",
          "dish_name": "Spinach and egg-yolk scramble",
          "cuisine": "Plain American",
          "prep_time_min": 8,
          "ingredients": [{"food": "yolk_egg", "grams": 76}, {"food": "spinach", "grams": 40}],
          "cooking_instructions": "Warm a nonstick pan over medium heat. Wilt the spinach for 30 seconds, then pour in the whisked egg yolks and stir until softly set, about 1 minute.",
          "nutritional_highlight": "Iron-rich start — ~30% RDA"
        },
        ...
      ],
      "leftover_note": "Cook extra rice and beans; portion for Monday lunch"
    },
    ...
  ],
  "shopping_list": [
    {"food": "brown_rice", "grams": 1309},
    {"food": "great_northern_beans", "grams": 1162},
    ...
  ],
  "weekly_summary": "The week rotates through Mediterranean, Asian, Mexican, Indian, and American styles, with carrots as the thread across every day. Beans anchor protein Mon-Wed; yolks and fish take over Thu-Sun."
}
```

## Retry feedback

If the validator rejects your plan and names a specific discrepancy, fix ONLY that food's allocation. Don't rewrite the whole plan. The mismatch is usually off-by-a-few-grams arithmetic; move the delta to the meal where that ingredient already appears most.

## Edge cases

- **Very small totals (< 10g for the week)**: distribute as a single meal's accent, not across multiple days. Trying to split 5g of lentils across 7 days is pointless.
- **Very large totals for one food (e.g., 3500g carrots)**: lean into it — carrots everywhere, multiple forms (raw, roasted, shredded, in soup). Don't try to hide it.
- **Foods without obvious cuisine fit**: default to "Plain American" if a food doesn't naturally land in the day's cuisine.
- **Allergen-filtered diets**: the user will include their restrictions; do not suggest ingredients from those categories even if the LP didn't include them.
"""


def load_lp_diet(path: Path, days: int = 7) -> dict[str, float]:
    """Read the LP output (optimum_diet.csv). Returns {food: weekly_grams}.

    optimum_diet.csv columns: '', food, grams (where grams is daily avg × 100
    per the original notebook's writer — actually grams/day). We multiply by
    `days` to get the week total.
    """
    out: dict[str, float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            food = row.get("food", "").strip()
            if not food:
                continue
            try:
                daily_g = float(row.get("grams", "0"))
            except ValueError:
                continue
            out[food] = daily_g * days
    return out


def validate_plan(plan: WeeklyMealPlan, lp_totals: dict[str, float],
                  tolerance_g: float = 5.0) -> list[str]:
    """Return a list of mass-conservation violations. Empty list = valid."""
    plan_totals: dict[str, float] = {}
    for day in plan.days:
        for meal in day.meals:
            for ing in meal.ingredients:
                plan_totals[ing.food] = plan_totals.get(ing.food, 0.0) + ing.grams

    violations = []
    for food, expected in lp_totals.items():
        got = plan_totals.get(food, 0.0)
        if abs(got - expected) > tolerance_g:
            violations.append(
                f"'{food}': plan sums to {got:.1f}g but LP total is {expected:.1f}g "
                f"(delta {got - expected:+.1f}g)"
            )
    for food, got in plan_totals.items():
        if food not in lp_totals:
            violations.append(f"'{food}': in plan ({got:.1f}g) but not in LP output")
    return violations


def rebalance_plan(
    plan: "WeeklyMealPlan",
    lp_totals: dict[str, float],
    tolerance_g: float = 0.5,
) -> list[str]:
    """Post-hoc nudge Claude's plan to match LP totals exactly.

    For each food, scales its existing appearances proportionally to hit the
    LP target. Preserves Claude's culinary choices (which meals contain
    which foods) while fixing the arithmetic. Foods that Claude omitted
    entirely are appended to day 1's snack as a fallback. Foods Claude
    hallucinated (not in LP) are removed.

    Returns a list of human-readable change descriptions.
    """
    # Aggregate current plan totals
    plan_totals: dict[str, float] = {}
    for day in plan.days:
        for meal in day.meals:
            for ing in meal.ingredients:
                plan_totals[ing.food] = plan_totals.get(ing.food, 0.0) + ing.grams

    changes: list[str] = []

    # (1) Scale each LP food to hit its exact target
    for food, target in lp_totals.items():
        current = plan_totals.get(food, 0.0)
        delta = target - current
        if abs(delta) < tolerance_g:
            continue

        if current == 0:
            # Food entirely missing from plan — add to day 1 snack
            day1 = plan.days[0]
            snack = next((m for m in day1.meals if m.name == "snack"), None)
            if snack is None:
                snack = day1.meals[-1]  # fall back to last meal
            snack.ingredients.append(Ingredient(food=food, grams=target))
            changes.append(f"{food}: added {target:.0f}g (was absent, placed in Day 1 {snack.name})")
            continue

        scale = target / current
        for day in plan.days:
            for meal in day.meals:
                for ing in meal.ingredients:
                    if ing.food == food:
                        ing.grams *= scale
        changes.append(
            f"{food}: scaled ×{scale:.3f} (from {current:.0f}g to {target:.0f}g, delta {delta:+.0f}g)"
        )

    # (2) Remove foods that Claude hallucinated (in plan but not in LP)
    for food, got in plan_totals.items():
        if food not in lp_totals:
            for day in plan.days:
                for meal in day.meals:
                    meal.ingredients = [i for i in meal.ingredients if i.food != food]
            changes.append(f"{food}: removed ({got:.0f}g — not in LP output)")

    # (3) Round all ingredient grams to whole numbers; absorb rounding error
    #     in the largest appearance of each food
    for food, target in lp_totals.items():
        refs = [
            ing for day in plan.days for meal in day.meals
            for ing in meal.ingredients if ing.food == food
        ]
        if not refs:
            continue
        for ing in refs:
            ing.grams = round(ing.grams)
        current_sum = sum(ing.grams for ing in refs)
        if current_sum != round(target):
            # Assign any residual to the largest allocation
            refs.sort(key=lambda i: -i.grams)
            refs[0].grams += round(target) - current_sum

    # (4) Replace shopping list with canonical LP totals
    plan.shopping_list = [
        Ingredient(food=f, grams=round(g))
        for f, g in sorted(lp_totals.items(), key=lambda kv: -kv[1])
        if round(g) > 0
    ]

    return changes


def call_claude(
    client: anthropic.Anthropic,
    model: str,
    lp_totals: dict[str, float],
    days: int,
    cuisine_style: str | None,
    max_prep_min: int,
    blacklist: list[str],
    whitelist: list[str],
    retry_feedback: str | None = None,
) -> WeeklyMealPlan:
    user_parts = [
        f"LP output (weekly grams per food, {days}-day total):",
        json.dumps(lp_totals, indent=2),
        "",
        f"Max prep time per meal: {max_prep_min} minutes (breakfast <15, one dinner may be up to 60 for batch-cook)",
    ]
    if cuisine_style:
        user_parts.append(
            f"CUISINE OVERRIDE: user wants all days in {cuisine_style!r} style, not varied"
        )
    else:
        user_parts.append(
            f"Rotate through these cuisines (one per day): {', '.join(DEFAULT_CUISINES)}"
        )
    if blacklist:
        user_parts.append(f"AVOID these ingredients entirely: {', '.join(blacklist)}")
    if whitelist:
        user_parts.append(f"INCLUDE these preferred ingredients when possible: {', '.join(whitelist)}")
    if retry_feedback:
        user_parts.append("")
        user_parts.append(f"VALIDATOR FEEDBACK — fix these specific discrepancies:\n{retry_feedback}")

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": "\n".join(user_parts)}],
        output_format=WeeklyMealPlan,
    )
    return response.parsed_output


def generate_with_retries(
    client: anthropic.Anthropic,
    model: str,
    lp_totals: dict[str, float],
    days: int,
    cuisine_style: str | None,
    max_prep_min: int,
    blacklist: list[str],
    whitelist: list[str],
    tolerance_g: float,
    max_retries: int = 3,
) -> tuple[WeeklyMealPlan, list[str]]:
    """Generate a plan, validate, retry on mismatch. Returns (plan, final_violations)."""
    feedback: str | None = None
    plan: WeeklyMealPlan | None = None
    violations: list[str] = []
    for attempt in range(max_retries):
        print(f"[attempt {attempt + 1}/{max_retries}] generating plan...",
              file=sys.stderr)
        plan = call_claude(
            client, model, lp_totals, days, cuisine_style, max_prep_min,
            blacklist, whitelist, feedback,
        )
        violations = validate_plan(plan, lp_totals, tolerance_g)
        if not violations:
            print(f"  ✓ valid on attempt {attempt + 1}", file=sys.stderr)
            return plan, []
        print(f"  ✗ {len(violations)} mass-conservation violations; retrying",
              file=sys.stderr)
        feedback = "\n".join(f"  - {v}" for v in violations[:10])
    return plan, violations


def render_markdown(plan: WeeklyMealPlan) -> str:
    """Pretty-print the plan as Markdown for human consumption."""
    lines = ["# Weekly Meal Plan", "", plan.weekly_summary, ""]
    for day in plan.days:
        lines.append(f"## Day {day.day}")
        if day.leftover_note:
            lines.append(f"*Batch-cook note: {day.leftover_note}*")
            lines.append("")
        for meal in day.meals:
            lines.append(f"### {meal.name.title()}: {meal.dish_name} _({meal.cuisine}, {meal.prep_time_min} min)_")
            for ing in meal.ingredients:
                lines.append(f"- {ing.food}: {ing.grams:.0f}g")
            lines.append("")
            lines.append(meal.cooking_instructions)
            if meal.nutritional_highlight:
                lines.append(f"_{meal.nutritional_highlight}_")
            lines.append("")
    lines.append("## Shopping list")
    lines.append("")
    for ing in sorted(plan.shopping_list, key=lambda i: -i.grams):
        lines.append(f"- {ing.food}: {ing.grams:.0f}g")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--diet", default="optimum_diet.csv",
                   help="path to optimum_diet.csv from the LP solver")
    p.add_argument("--output", default="meal_plan.json")
    p.add_argument("--markdown", default=None,
                   help="also write a human-readable .md version")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--cuisine-style", default=None,
                   help="override varied-cuisines default with a single style "
                        "(e.g. 'Mediterranean', 'Indian vegetarian')")
    p.add_argument("--max-prep-min", type=int, default=DEFAULT_MAX_PREP_MIN,
                   help=f"max prep minutes per meal (default {DEFAULT_MAX_PREP_MIN})")
    p.add_argument("--blacklist", default="",
                   help="comma-separated ingredients to avoid")
    p.add_argument("--whitelist", default="",
                   help="comma-separated preferred ingredients")
    p.add_argument("--tolerance-g", type=float, default=5.0,
                   help="mass-conservation tolerance in grams per food")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--no-rebalance", action="store_true",
        help="skip the post-hoc rebalancer (keep Claude's raw allocations; "
             "mass-conservation violations may remain)",
    )
    args = p.parse_args()

    # Defensive strip of ANTHROPIC_API_KEY — trailing whitespace / non-
    # breaking spaces from copy-pasted console values crash the HTTP layer.
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"].strip()

    lp_totals = load_lp_diet(Path(args.diet), days=args.days)
    if not lp_totals:
        print(f"ERROR: no foods parsed from {args.diet}", file=sys.stderr)
        return 1
    print(f"loaded {len(lp_totals)} foods from {args.diet} "
          f"(total {sum(lp_totals.values()):.0f}g for {args.days} days)",
          file=sys.stderr)

    client = anthropic.Anthropic()
    plan, violations = generate_with_retries(
        client=client,
        model=args.model,
        lp_totals=lp_totals,
        days=args.days,
        cuisine_style=args.cuisine_style,
        max_prep_min=args.max_prep_min,
        blacklist=[s.strip() for s in args.blacklist.split(",") if s.strip()],
        whitelist=[s.strip() for s in args.whitelist.split(",") if s.strip()],
        tolerance_g=args.tolerance_g,
        max_retries=args.max_retries,
    )
    if plan is None:
        print("no plan produced", file=sys.stderr)
        return 2

    # Post-hoc rebalance: scale ingredient portions to match LP totals exactly.
    if not args.no_rebalance and violations:
        print(f"\nrebalancing {len(violations)} mass-conservation deltas...",
              file=sys.stderr)
        changes = rebalance_plan(plan, lp_totals, tolerance_g=args.tolerance_g)
        for change in changes[:15]:
            print(f"  {change}", file=sys.stderr)
        # Re-validate after rebalancing
        post_violations = validate_plan(plan, lp_totals, tolerance_g=args.tolerance_g)
        if post_violations:
            print(f"\nWARNING: {len(post_violations)} violations remain after "
                  f"rebalance (likely rounding edge cases):", file=sys.stderr)
            for v in post_violations[:5]:
                print(f"  - {v}", file=sys.stderr)
        else:
            print(f"  ✓ mass conservation holds after rebalance", file=sys.stderr)
        violations = post_violations

    Path(args.output).write_text(plan.model_dump_json(indent=2))
    print(f"wrote {args.output}", file=sys.stderr)

    if args.markdown:
        Path(args.markdown).write_text(render_markdown(plan))
        print(f"wrote {args.markdown}", file=sys.stderr)

    if violations:
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
