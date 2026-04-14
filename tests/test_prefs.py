from dataclasses import dataclass

import pytest

from diet_opt.prefs import InvalidPrefsError, UserPrefs, apply_prefs


@dataclass
class FakeVar:
    lb: float = 0.0
    ub: float = 5.0


def _make_vars(names):
    return {n.replace(" ", "_"): FakeVar() for n in names}


def test_blacklist_sets_upper_to_zero():
    vars_ = _make_vars(["Broccoli", "Pinto Beans"])
    prefs = UserPrefs(blacklist=["Broccoli"])
    apply_prefs(vars_, prefs)
    assert vars_["Broccoli"].ub == 0
    assert vars_["Pinto_Beans"].ub == 5.0


def test_whitelist_sets_lower():
    vars_ = _make_vars(["Blueberries"])
    prefs = UserPrefs(whitelist=["Blueberries"], whitelist_min_grams={"Blueberries": 125})
    apply_prefs(vars_, prefs)
    assert vars_["Blueberries"].lb == 1.25  # 125g / 100g-per-unit


def test_whitelist_uses_default_min():
    vars_ = _make_vars(["Spinach"])
    prefs = UserPrefs(whitelist=["Spinach"])
    apply_prefs(vars_, prefs, default_min_grams=30)
    assert vars_["Spinach"].lb == 0.30


def test_unknown_food_raises():
    vars_ = _make_vars(["Broccoli"])
    prefs = UserPrefs(blacklist=["Quinoa"])
    with pytest.raises(InvalidPrefsError, match="unknown foods"):
        apply_prefs(vars_, prefs)


def test_overlap_blacklist_whitelist_raises():
    with pytest.raises(InvalidPrefsError, match="both blacklist and whitelist"):
        UserPrefs.from_dict({"blacklist": ["X"], "whitelist": ["X"]})


def test_unknown_field_raises():
    with pytest.raises(InvalidPrefsError, match="unknown prefs fields"):
        UserPrefs.from_dict({"blacklist": [], "typo_field": []})


def test_from_dict_minimal():
    prefs = UserPrefs.from_dict({})
    assert prefs.blacklist == []
    assert prefs.whitelist == []


def test_apply_returns_report():
    vars_ = _make_vars(["Broccoli", "Carrots"])
    prefs = UserPrefs(blacklist=["Broccoli"], whitelist=["Carrots"])
    report = apply_prefs(vars_, prefs, default_min_grams=50)
    assert report["excluded"] == ["Broccoli"]
    assert report["required"] == [("Carrots", 50)]
