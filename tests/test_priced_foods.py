"""Tests for scripts/build_priced_foods.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_priced_foods import (  # noqa: E402
    average_nutrients,
    build_priced_foods,
    group_fdc_by_search_term,
)


def test_average_nutrients_mean():
    entries = [
        ("Beans, pinto, raw", {"Protein": 20.0, "Iron": 5.0}),
        ("Beans, pinto, mature, raw", {"Protein": 22.0, "Iron": 4.0}),
    ]
    result = average_nutrients(entries)
    assert result["Protein"] == 21.0
    assert result["Iron"] == 4.5


def test_average_nutrients_skips_non_numeric():
    entries = [
        ("X", {"Protein": 10.0, "Junk": "N/A"}),
        ("Y", {"Protein": 20.0}),
    ]
    result = average_nutrients(entries)
    assert result["Protein"] == 15.0
    assert "Junk" not in result


def test_group_fdc_by_search_term_buckets_variants():
    fdc = {
        "Beans, pinto, mature seeds, raw": {"Protein": 21.0},
        "Beans, pinto, mature seeds, sprouted, raw": {"Protein": 23.0},
        "Beans, black, mature seeds, raw": {"Protein": 21.6},
        "Carrots, raw whole": {"Protein": 0.9},
    }
    buckets = group_fdc_by_search_term(fdc)
    assert "pinto beans" in buckets
    assert len(buckets["pinto beans"]) == 2
    assert "black beans" in buckets
    assert len(buckets["black beans"]) == 1


def test_build_priced_foods_joins_correctly():
    final_prices = {
        "pinto beans": {"price_per_100g": 0.20, "price_source": "kroger"},
        "carrots": {"price_per_100g": 0.16, "price_source": "kroger"},
    }
    fdc = {
        "Beans, pinto, mature seeds, raw": {"Protein": 21.0, "Iron": 5.0},
        "Beans, pinto, sprouted, raw": {"Protein": 23.0, "Iron": 4.0},
        "Carrots, raw whole": {"Protein": 0.9, "Iron": 0.3},
    }
    buckets = group_fdc_by_search_term(fdc)
    out, counts = build_priced_foods(final_prices, buckets)

    assert counts["matched"] == 2
    assert out["pinto beans"]["price_per_100g"] == 0.20
    assert out["pinto beans"]["nutrients"]["Protein"] == 22.0  # mean of 21, 23
    assert out["pinto beans"]["fdc_count"] == 2
    assert out["carrots"]["fdc_count"] == 1


def test_build_flags_price_without_fdc_bucket():
    final_prices = {
        "unicorn meat": {"price_per_100g": 99.99, "price_source": "kroger"},
    }
    out, counts = build_priced_foods(final_prices, {})
    assert counts["no_fdc_bucket"] == 1
    assert "unicorn meat" not in out


def test_build_preserves_provenance_fields():
    final_prices = {
        "adzuki beans": {
            "price_per_100g": 0.39,
            "price_source": "tfp",
            "tfp_category": "Beans, peas, lentils",
            "inflation_factor": 1.205,
            "cpi_current": "2026-03",
        },
    }
    # Create a matching bucket directly
    buckets = {"adzuki beans": [("Beans, adzuki, raw", {"Protein": 25.0})]}
    out, _ = build_priced_foods(final_prices, buckets)
    entry = out["adzuki beans"]
    assert entry["price_source"] == "tfp"
    assert entry["tfp_category"] == "Beans, peas, lentils"
    assert entry["inflation_factor"] == 1.205
