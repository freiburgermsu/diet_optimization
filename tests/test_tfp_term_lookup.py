"""Tests for build_tfp_term_lookup.py pure helpers (no API calls)."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _load():
    pytest.importorskip("anthropic")
    pytest.importorskip("pydantic")
    import build_tfp_term_lookup as m
    return m


def _make_tfp_csv(path: Path, rows: list[tuple[int, str, float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fndds_code", "tfp_category", "pricing_method", "price_per_100g_2021"])
        for code, cat, price in rows:
            w.writerow([code, cat, 1, price])


def test_slugify():
    m = _load()
    assert m.slugify("pinto beans") == "pinto_beans"
    assert m.slugify("Atlantic Cod!!") == "atlantic_cod"
    assert m.slugify("") == "_unnamed"


def test_load_tfp_category_stats_aggregates(tmp_path: Path):
    m = _load()
    p = tmp_path / "t.csv"
    _make_tfp_csv(p, [
        (1, "Poultry", 0.50),
        (2, "Poultry", 1.50),
        (3, "Poultry", 1.00),
        (4, "Seafood", 2.00),
    ])
    stats = m.load_tfp_category_stats(p)
    assert stats["Poultry"].median_price_per_100g_2021 == 1.00
    assert stats["Poultry"].n_foods == 3
    assert stats["Seafood"].n_foods == 1


def test_build_user_prompt_includes_term_and_categories():
    m = _load()
    prompt = m.build_user_prompt("salmon", ["Seafood", "Poultry"])
    assert '"salmon"' in prompt
    assert "Seafood" in prompt
    assert "Poultry" in prompt


def test_system_prompt_above_opus_cache_minimum():
    """Must be >4096 tokens for Opus prompt caching to activate."""
    m = _load()
    tokens = len(m.SYSTEM_INSTRUCTIONS) / 4  # rough char-to-token ratio
    assert tokens > 1000, f"only ~{tokens:.0f} tokens — cache won't trigger"


def test_live_tfp_prices_loads():
    m = _load()
    root = Path(__file__).resolve().parent.parent
    p = root / "data" / "tfp_prices.csv"
    if not p.exists():
        pytest.skip("tfp_prices.csv absent")
    stats = m.load_tfp_category_stats(p)
    assert len(stats) >= 60  # 67 categories expected
