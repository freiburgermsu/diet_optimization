import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fetch_prices import is_stale, normalize_product, parse_size_to_grams  # noqa: E402
from normalize_prices import price_per_100g_edible  # noqa: E402


def test_parse_size_oz():
    assert abs(parse_size_to_grams("16 oz") - 453.592) < 0.1


def test_parse_size_lb():
    assert abs(parse_size_to_grams("2 lb") - 907.184) < 0.1


def test_parse_size_g():
    assert parse_size_to_grams("500 g") == 500.0


def test_parse_size_kg():
    assert parse_size_to_grams("1.5 kg") == 1500.0


def test_parse_size_invalid():
    assert parse_size_to_grams("some garbage") is None
    assert parse_size_to_grams("16") is None
    assert parse_size_to_grams("") is None


def test_normalize_product_picks_promo_over_regular():
    raw = {
        "upc": "123",
        "description": "Kroger Carrots",
        "items": [{
            "price": {"regular": 1.99, "promo": 1.29},
            "size": "16 oz",
        }],
    }
    out = normalize_product(raw, "01400943")
    assert out["price"] == 1.29


def test_normalize_product_falls_back_to_regular():
    raw = {
        "upc": "123", "description": "X",
        "items": [{"price": {"regular": 2.00}, "size": "1 lb"}],
    }
    assert normalize_product(raw, "X")["price"] == 2.00


def test_normalize_product_returns_none_when_no_items():
    assert normalize_product({"items": []}, "X") is None


def test_is_stale_threshold():
    recent = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat().replace("+00:00", "Z")
    assert not is_stale(recent)
    assert is_stale(old)


def test_price_per_100g_edible_basic():
    # $2 for 454g, yield 1.0  →  $2/454*100 = $0.44/100g
    val = price_per_100g_edible(2.00, 454, 1.0)
    assert abs(val - 0.440) < 0.005


def test_price_per_100g_edible_yield_adjustment():
    base = price_per_100g_edible(2.00, 454, 1.0)
    with_loss = price_per_100g_edible(2.00, 454, 0.80)
    assert with_loss > base  # 20% inedible → more expensive per edible gram


def test_price_per_100g_edible_rejects_bad_input():
    import pytest
    with pytest.raises(ValueError):
        price_per_100g_edible(1.0, 0, 1.0)
    with pytest.raises(ValueError):
        price_per_100g_edible(1.0, 100, 0)


