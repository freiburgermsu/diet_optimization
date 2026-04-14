import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tfp_pricing import (  # noqa: E402
    TFPEntry,
    inflate_cpi,
    load_tfp,
    merge_price_sources,
    tally_sources,
    tfp_per_100g_edible,
)


def test_load_tfp_skips_comments():
    entries = load_tfp(Path(__file__).resolve().parent.parent / "data" / "tfp_prices.csv")
    assert len(entries) >= 5
    assert all(isinstance(e.price_per_lb_2019, float) for e in entries)
    # Comment line starting with # is not parsed as an entry
    assert all(not e.category.startswith("#") for e in entries)


def test_inflate_cpi_scales_correctly():
    # With default snapshots, a $1 base -> current/base ratio
    val = inflate_cpi(1.0)
    assert 1.0 < val < 1.5  # reasonable 5yr food inflation


def test_inflate_identity():
    assert inflate_cpi(10.0, base="2019-06", current="2019-06") == pytest.approx(10.0)


def test_per_100g_edible_conversion():
    entry = TFPEntry("legumes_dry", "Dry beans", 4.54, "t", "p")  # $4.54/lb base
    # With yield=1.0 and no inflation, expect $1.00/100g
    # With inflation ~302/255 ≈ 1.186
    val = tfp_per_100g_edible(entry, yield_factor=1.0)
    assert 1.15 < val < 1.22


def test_per_100g_edible_yield_reduces_effective_price():
    entry = TFPEntry("fruit", "Banana", 1.0, "t", "p")
    val_no_peel = tfp_per_100g_edible(entry, yield_factor=1.0)
    val_with_peel = tfp_per_100g_edible(entry, yield_factor=0.64)  # banana edible
    assert val_with_peel > val_no_peel  # peel loss makes edible gram pricier


def test_merge_prefers_primary():
    primary = {"Rice": {"price_per_100g": 0.20, "price_source": "retailer"}}
    tfp = [TFPEntry("grains_whole", "Brown rice", 1.0, "t", "p")]
    mapping = {"grains_whole": "Rice"}
    merged = merge_price_sources(primary, tfp, mapping)
    assert merged["Rice"]["price_source"] == "retailer"


def test_merge_fills_gap_with_tfp():
    primary = {}
    tfp = [TFPEntry("grains_whole", "Brown rice", 1.0, "t", "p")]
    mapping = {"grains_whole": "Rice"}
    merged = merge_price_sources(primary, tfp, mapping)
    assert merged["Rice"]["price_source"] == "tfp"
    assert "TFP 2021" in merged["Rice"]["price_citation"]


def test_tally_sources_counts():
    prices = {
        "A": {"price_source": "retailer"},
        "B": {"price_source": "retailer"},
        "C": {"price_source": "tfp"},
    }
    assert tally_sources(prices) == {"retailer": 2, "tfp": 1}
