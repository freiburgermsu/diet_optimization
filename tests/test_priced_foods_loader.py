"""Tests for diet_opt.data.load_priced_foods()."""
import json
from pathlib import Path

import pytest

from diet_opt.data import DEFAULT_CUP_EQ, load_priced_foods


def test_load_priced_foods_formats_for_build_model(tmp_path: Path, monkeypatch):
    """The returned food_info must have the shape build_model() expects:
    price/yield/4.54 must equal the original price_per_100g.
    """
    # Build a synthetic priced_foods.json alongside nutrition.json
    from diet_opt import data as data_mod

    priced = tmp_path / "priced_foods.json"
    priced.write_text(json.dumps({
        "pinto beans": {
            "price_per_100g": 0.20,
            "price_source": "kroger",
            "nutrients": {"Protein": 22.0, "Iron": 4.5},
        },
        "carrots": {
            "price_per_100g": 0.16,
            "price_source": "kroger",
            "nutrients": {"Vitamin A": 835.0},
        },
    }))
    nutrition_file = tmp_path / "nutrition.json"
    nutrition_file.write_text(json.dumps({
        "Protein": {"low_bound": 50, "high_bound": 150, "units": "g"},
    }))

    monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)

    food_info, food_matches, nutrition = load_priced_foods("priced_foods.json")

    # build_model does: price / yield / 4.54 as the per-100g cost.
    # Verify this reproduces the original price_per_100g.
    for term, entry in food_info.items():
        reconstructed = entry["price"] / entry["yield"] / 4.54
        expected = {"pinto beans": 0.20, "carrots": 0.16}[term]
        assert abs(reconstructed - expected) < 1e-9

    # Nutrients come through in food_matches
    assert food_matches["pinto beans"]["Protein"] == 22.0
    assert food_matches["carrots"]["Vitamin A"] == 835.0

    # nutrition passthrough
    assert "Protein" in nutrition


def test_default_cup_eq_used_when_missing(tmp_path: Path, monkeypatch):
    from diet_opt import data as data_mod

    priced = tmp_path / "priced_foods.json"
    priced.write_text(json.dumps({
        "carrots": {"price_per_100g": 0.16, "price_source": "kroger", "nutrients": {}},
    }))
    (tmp_path / "nutrition.json").write_text("{}")
    monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)

    food_info, _, _ = load_priced_foods("priced_foods.json")
    assert food_info["carrots"]["cupEQ"] == DEFAULT_CUP_EQ


def test_explicit_cup_eq_preserved(tmp_path: Path, monkeypatch):
    from diet_opt import data as data_mod

    priced = tmp_path / "priced_foods.json"
    priced.write_text(json.dumps({
        "carrots": {
            "price_per_100g": 0.16, "price_source": "kroger", "nutrients": {},
            "cup_equivalent": 0.5,
        },
    }))
    (tmp_path / "nutrition.json").write_text("{}")
    monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)

    food_info, _, _ = load_priced_foods("priced_foods.json")
    assert food_info["carrots"]["cupEQ"] == 0.5


def test_live_priced_foods_file_loads():
    root = Path(__file__).resolve().parent.parent
    if not (root / "priced_foods.json").exists():
        pytest.skip("priced_foods.json not present — run scripts/build_priced_foods.py")
    food_info, food_matches, nutrition = load_priced_foods()
    # Current live data is 610 priced terms
    assert 550 < len(food_info) < 700
    assert len(food_matches) == len(food_info)
    # Common-sense foods should be present
    assert any("carrot" in t for t in food_info)
    assert any("bean" in t for t in food_info)
