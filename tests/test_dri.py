"""Tests for diet_opt.dri — profile-scaled DRI bounds."""
import pytest

from diet_opt.dri import (
    ACTIVITY_PAL,
    PROTEIN_G_PER_KG,
    UserProfile,
    apply_profile,
    energy_kcal,
    fiber_rda_g,
    linoleic_g,
    linolenic_g,
    load_profile_overrides,
    mifflin_st_jeor_bmr,
    pick_bracket,
    protein_rda_g,
    water_rda_L,
)


def _p(**kw):
    defaults = dict(sex="male", age=28, weight_kg=68, height_cm=175, activity="active")
    defaults.update(kw)
    return UserProfile(**defaults)


# --- Validation ---

def test_profile_rejects_bad_sex():
    with pytest.raises(ValueError, match="sex"):
        _p(sex="mystery")


def test_profile_rejects_age_out_of_range():
    with pytest.raises(ValueError, match="age"):
        _p(age=-1)
    with pytest.raises(ValueError, match="age"):
        _p(age=200)


def test_profile_rejects_weight_out_of_range():
    with pytest.raises(ValueError, match="weight"):
        _p(weight_kg=5)
    with pytest.raises(ValueError, match="weight"):
        _p(weight_kg=500)


def test_profile_rejects_bad_activity():
    with pytest.raises(ValueError, match="activity"):
        _p(activity="hyperactive")


# --- BMR / energy ---

def test_mifflin_st_jeor_male_reference():
    # 28yo 68kg 175cm male → 10*68 + 6.25*175 - 5*28 + 5 = 680 + 1093.75 - 140 + 5 = 1638.75
    bmr = mifflin_st_jeor_bmr(_p())
    assert 1635 < bmr < 1645


def test_mifflin_female_offset_is_lower():
    male = mifflin_st_jeor_bmr(_p(sex="male"))
    female = mifflin_st_jeor_bmr(_p(sex="female"))
    assert female < male   # women have −161 offset


def test_energy_scales_with_activity():
    sedentary = energy_kcal(_p(activity="sedentary"))
    active = energy_kcal(_p(activity="active"))
    very_active = energy_kcal(_p(activity="very_active"))
    assert sedentary < active < very_active


def test_energy_in_realistic_range_for_default_profile():
    kcal = energy_kcal(_p())
    # 28yo 68kg active male should land ~2500-3000 kcal TDEE
    assert 2500 < kcal < 3100


# --- Protein ---

def test_protein_rda_weight_scaling():
    light_low = protein_rda_g(_p(weight_kg=50, activity="sedentary"))
    heavy_high = protein_rda_g(_p(weight_kg=100, activity="very_active"))
    assert light_low < heavy_high
    # Sedentary 50kg → 0.7 × 50 = 35 g
    assert abs(light_low - 35) < 0.1
    # Very active 100kg → 1.5 × 100 = 150 g
    assert abs(heavy_high - 150) < 0.1


# --- Fiber ---

def test_fiber_scales_with_energy():
    low = fiber_rda_g(1500)
    high = fiber_rda_g(3000)
    assert abs(low - 21.0) < 0.01   # 1500/1000 × 14
    assert abs(high - 42.0) < 0.01


# --- Water ---

def test_water_sex_offset():
    assert water_rda_L(_p(sex="male", activity="sedentary")) > water_rda_L(_p(sex="female", activity="sedentary"))


def test_water_activity_boost():
    base = water_rda_L(_p(sex="male", activity="sedentary"))
    vig = water_rda_L(_p(sex="male", activity="very_active"))
    assert vig > base


# --- Essential fatty acids ---

def test_linoleic_sex_specific():
    assert linoleic_g(_p(sex="male")) == 17
    assert linoleic_g(_p(sex="female")) == 12


def test_linolenic_sex_specific():
    assert linolenic_g(_p(sex="male")) == 1.6
    assert linolenic_g(_p(sex="female")) == 1.1


# --- Bracket picker ---

def test_pick_bracket_matches_youngest_eligible():
    brackets = [
        {"min_age": 19, "low_bound": 10, "high_bound": 100, "units": "mg"},
        {"min_age": 51, "low_bound": 8,  "high_bound": 100, "units": "mg"},
    ]
    assert pick_bracket(brackets, 30)["low_bound"] == 10
    assert pick_bracket(brackets, 60)["low_bound"] == 8


def test_pick_bracket_below_earliest_returns_none():
    brackets = [{"min_age": 19, "low_bound": 10, "high_bound": 100}]
    assert pick_bracket(brackets, 10) is None


# --- Overrides file ---

def test_load_overrides_has_iron_and_calcium():
    overrides = load_profile_overrides()
    assert "Iron" in overrides
    assert "Calcium" in overrides
    assert "male" in overrides["Iron"]
    assert "female" in overrides["Iron"]


# --- End-to-end ---

def test_apply_profile_scales_energy_vs_baseline():
    baseline = {"Energy": {"low_bound": 2400, "high_bound": 3200, "units": "kcal"}}
    small_profile = _p(weight_kg=45, activity="sedentary")
    scaled = apply_profile(baseline, small_profile)
    # A small sedentary person needs less energy
    assert scaled["Energy"]["low_bound"] < 2400
    assert scaled["Energy"]["high_bound"] < 3200


def test_apply_profile_patches_iron_by_sex():
    baseline = {}
    male = apply_profile(baseline, _p(sex="male", age=28))
    female = apply_profile(baseline, _p(sex="female", age=28))
    # Premenopausal women need far more iron
    assert female["Iron"]["low_bound"] > male["Iron"]["low_bound"]


def test_apply_profile_calcium_rises_after_51():
    baseline = {}
    young = apply_profile(baseline, _p(sex="female", age=30))
    old = apply_profile(baseline, _p(sex="female", age=60))
    assert old["Calcium"]["low_bound"] > young["Calcium"]["low_bound"]


def test_apply_profile_leaves_unknown_nutrient_untouched():
    baseline = {"Obscure": {"low_bound": 5, "high_bound": 50, "units": "x"}}
    scaled = apply_profile(baseline, _p())
    assert scaled["Obscure"] == {"low_bound": 5, "high_bound": 50, "units": "x"}


def test_apply_profile_patches_protein_from_weight():
    baseline = {"Protein": {"low_bound": 50, "high_bound": 200, "units": "grams"}}
    light = apply_profile(baseline, _p(weight_kg=50, activity="sedentary"))
    heavy = apply_profile(baseline, _p(weight_kg=100, activity="active"))
    assert heavy["Protein"]["low_bound"] > light["Protein"]["low_bound"]
