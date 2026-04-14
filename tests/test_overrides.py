from diet_opt.overrides import apply_overrides


def test_apply_overrides_fixes_inversion():
    nutrition = {"Magnesium": {"low_bound": 400, "high_bound": 350, "units": "mg"}}
    overrides = {"Magnesium": {"high_bound": 10000, "citation": "raw DRI inverted"}}
    merged = apply_overrides(nutrition, overrides)
    assert merged["Magnesium"]["high_bound"] == 10000
    assert merged["Magnesium"]["low_bound"] == 400  # unchanged


def test_apply_overrides_adds_missing_nutrient():
    nutrition = {}
    overrides = {"Carotenoids": {"high_bound": 100000, "units": "mg", "citation": "no UL"}}
    merged = apply_overrides(nutrition, overrides)
    assert merged["Carotenoids"]["high_bound"] == 100000
    assert "citation" not in merged["Carotenoids"]


def test_apply_overrides_leaves_unrelated_untouched():
    nutrition = {
        "Iron": {"low_bound": 8, "high_bound": 45, "units": "mg"},
        "Magnesium": {"low_bound": 400, "high_bound": 350, "units": "mg"},
    }
    overrides = {"Magnesium": {"high_bound": 10000, "citation": "fix"}}
    merged = apply_overrides(nutrition, overrides)
    assert merged["Iron"] == {"low_bound": 8, "high_bound": 45, "units": "mg"}


def test_apply_overrides_is_nondestructive():
    nutrition = {"X": {"low_bound": 1, "high_bound": 2}}
    original = {"X": {"low_bound": 1, "high_bound": 2}}
    apply_overrides(nutrition, {"X": {"high_bound": 99, "citation": "c"}})
    assert nutrition == original
