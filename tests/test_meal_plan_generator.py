"""Tests for scripts/generate_meal_plan.py (pure helpers; no API calls)."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _load():
    pytest.importorskip("anthropic")
    pytest.importorskip("pydantic")
    import generate_meal_plan as m
    return m


def _make_diet_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "food", "grams"])
        for i, (food, grams) in enumerate(rows):
            w.writerow([i, food, grams])


def test_load_lp_diet_multiplies_by_days(tmp_path: Path):
    m = _load()
    p = tmp_path / "diet.csv"
    _make_diet_csv(p, [("carrots", 100), ("brown_rice", 50)])
    weekly = m.load_lp_diet(p, days=7)
    assert weekly["carrots"] == 700
    assert weekly["brown_rice"] == 350


def test_load_lp_diet_handles_bad_rows(tmp_path: Path):
    m = _load()
    p = tmp_path / "diet.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "food", "grams"])
        w.writerow([0, "carrots", 100])
        w.writerow([1, "", 50])  # empty food
        w.writerow([2, "nonsense", "not_a_number"])  # bad grams
    out = m.load_lp_diet(p, days=1)
    assert out == {"carrots": 100}


def test_validate_plan_catches_mismatch():
    m = _load()
    Ing = m.Ingredient
    Meal = m.Meal
    Day = m.DayPlan
    Plan = m.WeeklyMealPlan

    plan = Plan(
        days=[
            Day(
                day=1,
                meals=[
                    Meal(
                        name="breakfast",
                        dish_name="X",
                        cuisine="Test",
                        prep_time_min=10,
                        ingredients=[Ing(food="carrots", grams=50)],
                        cooking_instructions="Cook.",
                    )
                ],
            )
        ],
        shopping_list=[Ing(food="carrots", grams=50)],
        weekly_summary="test",
    )
    violations = m.validate_plan(plan, {"carrots": 100}, tolerance_g=5)
    assert len(violations) == 1
    assert "carrots" in violations[0]


def test_validate_plan_accepts_within_tolerance():
    m = _load()
    Ing, Meal, Day, Plan = m.Ingredient, m.Meal, m.DayPlan, m.WeeklyMealPlan
    plan = Plan(
        days=[
            Day(
                day=1,
                meals=[
                    Meal(name="breakfast", dish_name="X", cuisine="T",
                         prep_time_min=5,
                         ingredients=[Ing(food="carrots", grams=97)],
                         cooking_instructions="Eat."),
                ],
            )
        ],
        shopping_list=[Ing(food="carrots", grams=97)],
        weekly_summary="s",
    )
    # LP expected 100, plan gives 97; delta = 3, tolerance = 5 → OK
    assert m.validate_plan(plan, {"carrots": 100}, tolerance_g=5) == []


def test_validate_plan_flags_extra_foods_in_plan():
    m = _load()
    Ing, Meal, Day, Plan = m.Ingredient, m.Meal, m.DayPlan, m.WeeklyMealPlan
    plan = Plan(
        days=[
            Day(
                day=1,
                meals=[
                    Meal(name="breakfast", dish_name="X", cuisine="T",
                         prep_time_min=5,
                         ingredients=[
                             Ing(food="carrots", grams=100),
                             Ing(food="unicorn", grams=10),  # not in LP
                         ],
                         cooking_instructions="Blend."),
                ],
            )
        ],
        shopping_list=[Ing(food="carrots", grams=100), Ing(food="unicorn", grams=10)],
        weekly_summary="s",
    )
    violations = m.validate_plan(plan, {"carrots": 100})
    assert any("unicorn" in v for v in violations)


def test_render_markdown_includes_all_sections():
    m = _load()
    Ing, Meal, Day, Plan = m.Ingredient, m.Meal, m.DayPlan, m.WeeklyMealPlan
    plan = Plan(
        days=[
            Day(
                day=1,
                leftover_note="cook extra rice",
                meals=[
                    Meal(name="breakfast", dish_name="Oatmeal", cuisine="American",
                         prep_time_min=5,
                         ingredients=[Ing(food="oats", grams=50)],
                         cooking_instructions="Boil with water.",
                         nutritional_highlight="Fiber-rich"),
                ],
            )
        ],
        shopping_list=[Ing(food="oats", grams=50)],
        weekly_summary="One day of oatmeal.",
    )
    md = m.render_markdown(plan)
    assert "Weekly Meal Plan" in md
    assert "Day 1" in md
    assert "cook extra rice" in md
    assert "Oatmeal" in md
    assert "Boil with water" in md
    assert "Fiber-rich" in md
    assert "Shopping list" in md
    assert "oats: 50g" in md


def test_default_cuisines_are_varied():
    m = _load()
    assert len(m.DEFAULT_CUISINES) == 7
    assert len(set(m.DEFAULT_CUISINES)) == 7   # all unique


def test_system_prompt_exceeds_cache_threshold():
    """Opus 4.6 needs ~4096 tokens to cache the prompt."""
    m = _load()
    tokens = len(m.SYSTEM_PROMPT) / 4
    assert tokens > 700, f"only ~{tokens:.0f} tokens — won't cache on Opus 4.6"


def test_pydantic_schema_enforces_meal_name_enum():
    m = _load()
    with pytest.raises(Exception):  # ValidationError
        m.Meal(name="brunch", dish_name="X", cuisine="Y",
               prep_time_min=5, ingredients=[], cooking_instructions="Z")


def test_pydantic_schema_enforces_day_range():
    m = _load()
    with pytest.raises(Exception):
        m.DayPlan(day=8, meals=[])  # >7
    with pytest.raises(Exception):
        m.DayPlan(day=0, meals=[])  # <1
