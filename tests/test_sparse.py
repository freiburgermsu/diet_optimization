from diet_opt.sparse import Triage, categorize_nutrient, impute_from_group_medians, load_triage


def test_load_triage_has_b12():
    triage = load_triage()
    assert triage["B12"].tier == "supplement"


def test_categorize_hard_when_sufficient_support():
    triage = {"B12": Triage("supplement", "r", None)}
    assert categorize_nutrient("B12", foods_supporting=10, triage=triage) == "hard"


def test_categorize_uses_triage_when_sparse():
    triage = {"B12": Triage("supplement", "r", None)}
    assert categorize_nutrient("B12", foods_supporting=3, triage=triage) == "supplement"


def test_categorize_drop_when_unknown_and_sparse():
    assert categorize_nutrient("Mystery", 2, {}) == "drop"


def test_impute_from_group_medians_basic():
    food_matches = {
        "Carrots": {"Phosophorous": 40},
        "Beets": {"Phosophorous": 50},
        "Radishes": {},  # missing
    }
    groups = {"Carrots": "root_veg", "Beets": "root_veg", "Radishes": "root_veg"}
    imputed = impute_from_group_medians(food_matches, groups, "Phosophorous")
    assert imputed == {"Radishes": 45.0}  # median of 40, 50


def test_impute_skips_food_with_no_group_data():
    food_matches = {
        "Carrots": {},  # missing, and no other root veg has data
        "Apples": {"Phosophorous": 10},
    }
    groups = {"Carrots": "root_veg", "Apples": "fruit"}
    imputed = impute_from_group_medians(food_matches, groups, "Phosophorous")
    assert imputed == {}  # Carrots has no root_veg data to impute from


def test_impute_does_not_overwrite_present():
    food_matches = {
        "Carrots": {"Phosophorous": 40},
        "Beets": {"Phosophorous": 50},
    }
    groups = {"Carrots": "root_veg", "Beets": "root_veg"}
    imputed = impute_from_group_medians(food_matches, groups, "Phosophorous")
    assert imputed == {}  # nothing missing
