from math import inf

from diet_opt.data import average_dict_values, parse_bound, validate_bounds


def test_parse_bound_numeric():
    assert parse_bound(100) == 100.0
    assert parse_bound(1.5) == 1.5


def test_parse_bound_string_with_comma():
    assert parse_bound("1,200") == 1200.0


def test_parse_bound_string_with_unit():
    assert parse_bound("100 mg") == 100.0
    assert parse_bound("2.5 grams") == 2.5


def test_parse_bound_sentinel():
    assert parse_bound("ND") == inf
    assert parse_bound("inf") == inf
    assert parse_bound("") == inf


def test_validate_bounds_ok():
    assert validate_bounds({"Iron": {"low_bound": 8, "high_bound": 45}}) == []


def test_validate_bounds_detects_inversion():
    nutrition = {"Magnesium": {"low_bound": 400, "high_bound": 350}}
    violations = validate_bounds(nutrition)
    assert len(violations) == 1
    assert "Magnesium" in violations[0]


def test_validate_bounds_handles_infinity():
    assert validate_bounds({"Potassium": {"low_bound": 4700, "high_bound": "ND"}}) == []


def test_average_dict_values_single_source():
    result = average_dict_values([{"apple": {"carbs": 25, "protein": 1}}])
    assert result == {"carbs": 25.0, "protein": 1.0}


def test_average_dict_values_multi_source():
    dicts = [
        {"apple1": {"carbs": 20, "protein": 1}},
        {"apple2": {"carbs": 30, "protein": 2}},
    ]
    result = average_dict_values(dicts)
    assert result == {"carbs": 25.0, "protein": 1.5}


def test_average_dict_values_partial_overlap():
    dicts = [
        {"a": {"x": 10}},
        {"b": {"x": 20, "y": 5}},
    ]
    result = average_dict_values(dicts)
    assert result["x"] == 15.0
    assert result["y"] == 5.0
