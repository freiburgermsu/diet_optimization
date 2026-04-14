import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_food_table import build_unified, validate_unified  # noqa: E402


def test_build_uses_fdc_id_when_available():
    info = {"Carrots, raw": {"price": 1.28, "cupEQ": 1.0, "yield": 0.92}}
    nutrients = {"Carrots, raw": {"Iron": 0.3}}
    mapping = {"Carrots, raw": "Carrots"}
    fdc = {"carrots": 170393}
    unified = build_unified(info, nutrients, mapping, fdc)
    assert "170393" in unified
    assert unified["170393"]["name"] == "Carrots"
    assert unified["170393"]["nutrients"]["Iron"] == 0.3


def test_build_falls_back_to_synthetic_id():
    info = {"Unicorn fruit": {"price": 1.0, "cupEQ": 1.0, "yield": 1.0}}
    unified = build_unified(info, {}, {}, {})
    keys = list(unified.keys())
    assert len(keys) == 1
    assert int(keys[0]) < 0


def test_multiple_synthetic_ids_unique():
    info = {"A": {"price": 1, "cupEQ": 1, "yield": 1},
            "B": {"price": 2, "cupEQ": 1, "yield": 1}}
    unified = build_unified(info, {}, {}, {})
    assert len(set(unified.keys())) == 2


def test_price_normalized_to_per_100g():
    info = {"X": {"price": 4.54, "cupEQ": 1.0, "yield": 1.0}}   # $/lb, yield 1
    unified = build_unified(info, {}, {}, {})
    entry = next(iter(unified.values()))
    # price / yield / 4.54 = 1.00 ... in $/100g edible
    assert abs(entry["price_per_100g"] - 1.0) < 1e-9


def test_validate_unified_accepts_valid():
    unified = {
        "1": {
            "name": "X", "price_per_100g": 1.0, "cup_equivalent_g": 1.0,
            "yield": 1.0, "nutrients": {},
        }
    }
    assert validate_unified(unified) == []


def test_validate_unified_rejects_non_numeric_key():
    unified = {
        "abc": {
            "name": "X", "price_per_100g": 1.0, "cup_equivalent_g": 1.0,
            "yield": 1.0, "nutrients": {},
        }
    }
    errs = validate_unified(unified)
    assert any("not an fdc_id" in e for e in errs)


def test_validate_unified_detects_missing_fields():
    unified = {"1": {"name": "X"}}
    errs = validate_unified(unified)
    assert any("missing fields" in e for e in errs)


def test_validate_unified_detects_nutrients_type():
    unified = {
        "1": {
            "name": "X", "price_per_100g": 1.0, "cup_equivalent_g": 1.0,
            "yield": 1.0, "nutrients": "not a dict",
        }
    }
    errs = validate_unified(unified)
    assert any("nutrients must be dict" in e for e in errs)
