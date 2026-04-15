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


# --- Rebalancer ---

def _make_simple_plan(m):
    """Build a plan with ingredient appearances for rebalancer tests."""
    return m.WeeklyMealPlan(
        days=[
            m.DayPlan(
                day=d,
                meals=[
                    m.Meal(
                        name="breakfast", dish_name="B", cuisine="C",
                        prep_time_min=5,
                        ingredients=[m.Ingredient(food="carrots", grams=50)],
                        cooking_instructions="Eat.",
                    ),
                    m.Meal(
                        name="snack", dish_name="S", cuisine="C",
                        prep_time_min=2,
                        ingredients=[],
                        cooking_instructions="Pick.",
                    ),
                ],
            )
            for d in range(1, 4)
        ],
        shopping_list=[m.Ingredient(food="carrots", grams=150)],
        weekly_summary="test",
    )


def test_rebalance_scales_underallocated_food():
    m = _load()
    plan = _make_simple_plan(m)
    # Plan has 3 × 50g = 150g carrots; LP wants 300g
    changes = m.rebalance_plan(plan, {"carrots": 300.0})
    total = sum(
        ing.grams for day in plan.days for meal in day.meals
        for ing in meal.ingredients if ing.food == "carrots"
    )
    assert total == 300
    assert any("carrots" in c for c in changes)


def test_rebalance_scales_overallocated_food():
    m = _load()
    plan = _make_simple_plan(m)
    # LP wants only 75g; plan has 150g → scale down 0.5×
    m.rebalance_plan(plan, {"carrots": 75.0})
    total = sum(
        ing.grams for day in plan.days for meal in day.meals
        for ing in meal.ingredients if ing.food == "carrots"
    )
    assert total == 75


def test_rebalance_adds_absent_food_to_day1_snack():
    m = _load()
    plan = _make_simple_plan(m)
    changes = m.rebalance_plan(plan, {"carrots": 150.0, "tofu": 80.0})
    day1_snack = next(meal for meal in plan.days[0].meals if meal.name == "snack")
    tofu_in_snack = [i for i in day1_snack.ingredients if i.food == "tofu"]
    assert len(tofu_in_snack) == 1
    assert tofu_in_snack[0].grams == 80
    assert any("tofu" in c for c in changes)


def test_rebalance_removes_hallucinated_food():
    m = _load()
    plan = _make_simple_plan(m)
    plan.days[0].meals[0].ingredients.append(m.Ingredient(food="unicorn", grams=10))
    changes = m.rebalance_plan(plan, {"carrots": 150.0})
    unicorn_count = sum(
        1 for day in plan.days for meal in day.meals
        for ing in meal.ingredients if ing.food == "unicorn"
    )
    assert unicorn_count == 0
    assert any("unicorn" in c for c in changes)


def test_rebalance_replaces_shopping_list():
    m = _load()
    plan = _make_simple_plan(m)
    plan.shopping_list = [m.Ingredient(food="stale", grams=99)]
    m.rebalance_plan(plan, {"carrots": 300.0, "tofu": 50.0})
    foods = {i.food for i in plan.shopping_list}
    assert foods == {"carrots", "tofu"}
    carrots = next(i for i in plan.shopping_list if i.food == "carrots")
    assert carrots.grams == 300


def test_rebalance_preserves_validation():
    m = _load()
    plan = _make_simple_plan(m)
    lp = {"carrots": 430.0, "tofu": 75.0}
    m.rebalance_plan(plan, lp)
    assert m.validate_plan(plan, lp, tolerance_g=1.0) == []


def test_rebalance_skips_within_tolerance():
    m = _load()
    plan = _make_simple_plan(m)
    changes = m.rebalance_plan(plan, {"carrots": 150.3}, tolerance_g=0.5)
    assert not changes


# --- Cooking yields + dual-weight rendering ---

def test_lookup_yield_substring_match():
    m = _load()
    yields = {
        "pinto_beans": {"raw_to_cooked": 2.8, "state": "dry"},
        "beans": {"raw_to_cooked": 2.7, "state": "dry"},
    }
    # Longest match wins
    assert m.lookup_yield("pinto_beans", yields)["raw_to_cooked"] == 2.8
    # Partial match falls back to shorter key
    assert m.lookup_yield("navy_beans", yields)["raw_to_cooked"] == 2.7


def test_lookup_yield_case_insensitive():
    m = _load()
    yields = {"brown_rice": {"raw_to_cooked": 3.0, "state": "dry"}}
    assert m.lookup_yield("Brown Rice", yields)["raw_to_cooked"] == 3.0


def test_lookup_yield_missing_returns_none():
    m = _load()
    assert m.lookup_yield("unicorn_meat", {}) is None


def test_format_ingredient_dry_shows_cooked():
    m = _load()
    ing = m.Ingredient(food="pinto_beans", grams=100)
    yields = {"pinto_beans": {"raw_to_cooked": 2.8, "state": "dry"}}
    line = m.format_ingredient(ing, yields)
    assert "100g dry" in line
    assert "280g cooked" in line


def test_format_ingredient_fresh_no_annotation():
    m = _load()
    ing = m.Ingredient(food="carrots", grams=100)
    yields = {"carrots": {"raw_to_cooked": 0.9, "state": "fresh"}}
    line = m.format_ingredient(ing, yields)
    assert "dry" not in line
    assert line == "- carrots: 100g"


def test_format_ingredient_no_yield_data_no_annotation():
    m = _load()
    ing = m.Ingredient(food="mystery", grams=50)
    line = m.format_ingredient(ing, {})
    assert line == "- mystery: 50g"


def test_render_markdown_filters_zero_ingredients():
    m = _load()
    plan = _make_simple_plan(m)
    # Insert a zero-ingredient "reheat" meal
    plan.days[0].meals[0].ingredients = [
        m.Ingredient(food="carrots", grams=0),
        m.Ingredient(food="pinto_beans", grams=0),
    ]
    md = m.render_markdown(plan)
    assert "carrots: 0g" not in md
    assert "reheat from an earlier day" in md


def test_render_markdown_uses_dual_weight_for_dry_foods():
    m = _load()
    plan = _make_simple_plan(m)
    plan.shopping_list = [
        m.Ingredient(food="pinto_beans", grams=3500),
        m.Ingredient(food="carrots", grams=2000),
    ]
    yields = {
        "pinto_beans": {"raw_to_cooked": 2.8, "state": "dry"},
        "carrots": {"raw_to_cooked": 0.9, "state": "fresh"},
    }
    md = m.render_markdown(plan, yields=yields)
    assert "3500g dry" in md
    assert "cooked" in md  # 9800g cooked appears for pinto
    # carrots get no annotation
    assert "- carrots: 2000g" in md
    assert "carrots: 2000g dry" not in md


def test_load_cooking_yields_returns_empty_if_file_missing(tmp_path):
    m = _load()
    assert m.load_cooking_yields(tmp_path / "nonexistent.yaml") == {}


def test_load_cooking_yields_live_file_has_pinto_beans():
    m = _load()
    root = Path(__file__).resolve().parent.parent
    p = root / "data" / "cooking_yields.yaml"
    if not p.exists():
        pytest.skip("cooking_yields.yaml not present")
    yields = m.load_cooking_yields(p)
    assert "pinto_beans" in yields
    assert yields["pinto_beans"]["raw_to_cooked"] > 2.5
    assert yields["pinto_beans"]["state"] == "dry"
