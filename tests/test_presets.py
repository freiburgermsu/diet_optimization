"""Tests for diet_opt.presets (dietary preset expansion)."""
from pathlib import Path

import pytest

from diet_opt.presets import (
    expand_preset,
    foods_excluded_by_presets,
    foods_matching_keywords,
    keywords_for_preset,
    list_presets,
    load_dietary_groups,
)


def test_load_dietary_groups_has_expected_keys():
    cfg = load_dietary_groups()
    assert "groups" in cfg
    assert "presets" in cfg
    assert "meat" in cfg["groups"]
    assert "vegan" in cfg["presets"]


def test_list_presets_includes_common_names():
    names = list_presets()
    for expected in ["vegan", "vegetarian", "pescatarian", "gluten_free", "nut_free"]:
        assert expected in names


def test_expand_preset_vegan():
    cfg = load_dietary_groups()
    groups = expand_preset("vegan", cfg)
    assert set(groups) == {"meat", "fish", "dairy", "egg"}


def test_expand_preset_unknown_raises():
    cfg = load_dietary_groups()
    with pytest.raises(KeyError):
        expand_preset("carnivore_supreme", cfg)


def test_keywords_for_preset_vegan_is_union():
    cfg = load_dietary_groups()
    keywords = keywords_for_preset("vegan", cfg)
    # Should include items from meat, fish, dairy, egg groups
    assert "beef" in keywords
    assert "salmon" in keywords
    assert "milk" in keywords
    assert "egg" in keywords


def test_foods_matching_keywords_case_insensitive():
    keywords = {"beef", "salmon"}
    foods = ["Ground Beef", "Wild Salmon", "Carrots", "CHICKEN"]
    matches = foods_matching_keywords(foods, keywords)
    assert "Ground Beef" in matches
    assert "Wild Salmon" in matches
    assert "Carrots" not in matches
    assert "CHICKEN" not in matches  # chicken not in our keyword set


def test_foods_excluded_by_vegetarian():
    foods = [
        "carrots", "ground beef", "atlantic salmon", "tofu",
        "chicken breast", "pinto beans", "kale"
    ]
    excluded = foods_excluded_by_presets(["vegetarian"], foods)
    assert "ground beef" in excluded
    assert "atlantic salmon" in excluded
    assert "chicken breast" in excluded
    assert "carrots" not in excluded
    assert "tofu" not in excluded


def test_foods_excluded_by_vegan():
    foods = ["milk", "egg", "yolk egg", "tofu", "ground beef", "carrots"]
    excluded = foods_excluded_by_presets(["vegan"], foods)
    assert "milk" in excluded
    assert "egg" in excluded
    assert "yolk egg" in excluded
    assert "ground beef" in excluded
    assert "tofu" not in excluded
    assert "carrots" not in excluded


def test_multiple_presets_union():
    foods = ["ground beef", "wheat bread", "tofu", "almonds nuts", "carrots"]
    excluded = foods_excluded_by_presets(["vegan", "gluten_free", "nut_free"], foods)
    assert "ground beef" in excluded
    assert "wheat bread" in excluded
    assert "almonds nuts" in excluded
    assert "tofu" not in excluded
    assert "carrots" not in excluded


def test_nut_preset_matches_coconut_not_peanut():
    foods = ["coconut milk nuts", "peanuts", "pistachio nuts", "walnut_halves"]
    excluded = foods_excluded_by_presets(["nut_free"], foods)
    assert "coconut milk nuts" in excluded
    assert "pistachio nuts" in excluded
    assert "walnut_halves" in excluded
    assert "peanuts" not in excluded   # peanuts are legumes, not nuts


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        foods_excluded_by_presets(["not_a_real_preset"], ["carrots"])


def test_live_data_vegan_excludes_reasonable_count(tmp_path: Path):
    """On the live 610-food table, vegan preset should exclude ~100-300 foods."""
    import json
    root = Path(__file__).resolve().parent.parent
    pf = root / "priced_foods.json"
    if not pf.exists():
        pytest.skip("priced_foods.json not present")
    foods = list(json.loads(pf.read_text()).keys())
    excluded = foods_excluded_by_presets(["vegan"], foods)
    # 610-food table has lots of meat/fish/dairy; expect substantial exclusion
    assert 50 < len(excluded) < 500
    # Sanity: common vegan foods should remain
    remaining = set(foods) - set(excluded)
    assert any("bean" in f for f in remaining)
    assert any("rice" in f for f in remaining)
