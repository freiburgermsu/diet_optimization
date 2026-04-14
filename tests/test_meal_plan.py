from diet_opt.meal_plan import (
    MEAL_PLAN_JSON_SCHEMA,
    aggregate_plan_totals,
    format_retry_message,
    validate_plan,
)


def _plan(*ingredients_per_meal):
    return {
        "meals": [
            {"name": f"meal{i}", "ingredients": ings}
            for i, ings in enumerate(ingredients_per_meal)
        ]
    }


def test_aggregate_sums_across_meals():
    plan = _plan(
        [{"food": "corn", "grams": 100}, {"food": "rice", "grams": 50}],
        [{"food": "corn", "grams": 73}],
    )
    totals = aggregate_plan_totals(plan)
    assert totals == {"corn": 173, "rice": 50}


def test_valid_plan_no_discrepancies():
    lp = {"corn": 173, "rice": 50}
    plan = _plan([{"food": "corn", "grams": 173}, {"food": "rice", "grams": 50}])
    assert validate_plan(plan, lp) == []


def test_double_counted_corn_detected():
    # The exact Figure-2 failure mode from the report
    lp = {"corn": 173}
    plan = _plan([{"food": "corn", "grams": 173}], [{"food": "corn", "grams": 173}])
    d = validate_plan(plan, lp)
    assert len(d) == 1
    assert d[0].food == "corn"
    assert d[0].lp_grams == 173
    assert d[0].plan_grams == 346


def test_tolerance_respected():
    lp = {"corn": 173}
    plan = _plan([{"food": "corn", "grams": 174.5}])
    assert validate_plan(plan, lp, tolerance_g=2) == []


def test_missing_food_flagged():
    lp = {"rice": 50, "corn": 173}
    plan = _plan([{"food": "corn", "grams": 173}])
    d = validate_plan(plan, lp)
    assert any(x.food == "rice" and x.plan_grams == 0 for x in d)


def test_extra_food_flagged():
    lp = {"corn": 173}
    plan = _plan([{"food": "corn", "grams": 173}, {"food": "unicorn", "grams": 10}])
    d = validate_plan(plan, lp)
    assert any(x.food == "unicorn" and x.lp_grams == 0 for x in d)


def test_retry_message_mentions_food_and_delta():
    lp = {"corn": 173}
    plan = _plan([{"food": "corn", "grams": 346}])
    msg = format_retry_message(validate_plan(plan, lp))
    assert "corn" in msg
    assert "346" in msg
    assert "173" in msg
    assert "+173" in msg


def test_schema_requires_meals():
    assert "meals" in MEAL_PLAN_JSON_SCHEMA["required"]


def test_schema_forbids_extra_properties_on_meal():
    assert MEAL_PLAN_JSON_SCHEMA["properties"]["meals"]["items"]["additionalProperties"] is False
