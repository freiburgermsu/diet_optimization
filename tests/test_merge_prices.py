"""Tests for scripts/merge_prices.py.

Verifies precedence logic, CPI inflation, and provenance stamping
against synthetic inputs. No live-data dependencies.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from merge_prices import (  # noqa: E402
    build_final_prices,
    load_claude_prices,
    load_terms,
    load_tfp_lookup,
)


def test_claude_price_wins_over_tfp():
    terms = ["carrots", "adzuki beans"]
    claude = {
        "carrots": {
            "price_per_100g": 0.16,
            "price_source": "kroger",
            "raw_description": "Kroger Whole Carrots",
            "claude_confidence": "high",
        }
    }
    tfp = {
        "carrots": {
            "tfp_category": "Other vegetables",
            "price_per_100g_2021": 0.80,
            "confidence": "high",
            "reason": "r",
        },
        "adzuki beans": {
            "tfp_category": "Beans, peas, lentils",
            "price_per_100g_2021": 0.32,
            "confidence": "high",
            "reason": "r",
        },
    }
    final, counts = build_final_prices(terms, claude, tfp)

    # carrots → Claude (kroger price wins)
    assert final["carrots"]["price_source"] == "kroger"
    assert final["carrots"]["price_per_100g"] == 0.16
    # adzuki beans → TFP fallback
    assert final["adzuki beans"]["price_source"] == "tfp"
    # TFP entry is CPI-inflated
    assert final["adzuki beans"]["price_per_100g"] > 0.32

    assert counts["kroger"] == 1
    assert counts["tfp"] == 1
    assert counts["unpriced"] == 0


def test_unpriced_term_is_counted_not_included():
    terms = ["carrots", "bear game meat"]
    claude = {"carrots": {"price_per_100g": 0.16}}
    tfp = {}
    final, counts = build_final_prices(terms, claude, tfp)

    assert "bear game meat" not in final
    assert counts["unpriced"] == 1


def test_tfp_entry_has_provenance_fields():
    terms = ["adzuki beans"]
    tfp = {
        "adzuki beans": {
            "tfp_category": "Beans, peas, lentils",
            "price_per_100g_2021": 0.32,
            "confidence": "high",
            "reason": "Dry legumes",
        }
    }
    final, _ = build_final_prices(terms, {}, tfp)
    entry = final["adzuki beans"]
    assert entry["price_source"] == "tfp"
    assert entry["tfp_category"] == "Beans, peas, lentils"
    assert "inflation_factor" in entry
    assert entry["inflation_factor"] > 1.0
    assert entry["cpi_base"] == "2021-06"
    assert entry["confidence"] == "high"


def test_cpi_inflation_is_applied(tmp_path: Path):
    terms = ["agave"]
    tfp = {
        "agave": {
            "tfp_category": "Sugars",
            "price_per_100g_2021": 1.00,
            "confidence": "high",
            "reason": "r",
        }
    }
    final, _ = build_final_prices(terms, {}, tfp)
    price_current = final["agave"]["price_per_100g"]
    # June 2021 → current inflation factor is ~1.20 (BLS CPI food-at-home)
    assert 1.15 < price_current < 1.25


def test_load_claude_prices_missing_file_returns_empty(tmp_path: Path):
    assert load_claude_prices(tmp_path / "nope.json") == {}


def test_load_tfp_lookup_skips_null_rows(tmp_path: Path):
    p = tmp_path / "lookup.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["search_term", "tfp_category", "price_per_100g_2021", "confidence", "reason"])
        w.writerow(["carrots", "Other vegetables", "0.80", "high", "ok"])
        w.writerow(["bear game meat", "", "", "high", "no TFP coverage"])
    loaded = load_tfp_lookup(p)
    assert "carrots" in loaded
    assert "bear game meat" not in loaded


def test_load_terms_reads_prices_raw_shape(tmp_path: Path):
    p = tmp_path / "raw.json"
    p.write_text(json.dumps({
        "terms": ["carrots", "beef", "apple"],
        "products": [],
    }))
    assert load_terms(p) == ["carrots", "beef", "apple"]


def test_end_to_end_coverage_counting():
    """Simulate the real 708-term distribution at a smaller scale."""
    terms = ["a", "b", "c", "d", "e"]  # a,b → kroger; c → tfp; d,e → dropped
    claude = {
        "a": {"price_per_100g": 1.0},
        "b": {"price_per_100g": 2.0},
    }
    tfp = {
        "c": {
            "tfp_category": "X", "price_per_100g_2021": 0.5,
            "confidence": "high", "reason": "",
        },
    }
    final, counts = build_final_prices(terms, claude, tfp)
    assert len(final) == 3
    assert counts == {"kroger": 2, "tfp": 1, "unpriced": 2}
