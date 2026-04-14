import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tfp_pricing import (  # noqa: E402
    TFPEntry,
    inflate_cpi,
    load_fndds_fdc_crosswalk,
    load_tfp,
    merge_price_sources,
    tally_sources,
    tfp_price_current,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_tfp_parses_all_rows():
    entries = load_tfp(DATA_DIR / "tfp_prices.csv")
    # USDA publishes ~3000 FNDDS entries in the Online Supplement
    assert len(entries) >= 3000
    assert all(isinstance(e.price_per_100g_2021, float) for e in entries)
    assert all(isinstance(e.fndds_code, int) for e in entries)


def test_load_tfp_categories_present():
    entries = load_tfp(DATA_DIR / "tfp_prices.csv")
    categories = {e.tfp_category for e in entries}
    # Should include major food groups
    assert any("Seafood" in c for c in categories)
    assert any("Milk" in c or "dairy" in c.lower() for c in categories)


def test_inflate_cpi_scales_forward():
    val = inflate_cpi(1.0)
    assert val > 1.0   # inflation is positive


def test_inflate_identity():
    assert inflate_cpi(10.0, base="2021-06", current="2021-06") == pytest.approx(10.0)


def test_tfp_price_current_applies_inflation():
    entry = TFPEntry(fndds_code=11100000, tfp_category="Milk", pricing_method=1,
                     price_per_100g_2021=1.00)
    cur = tfp_price_current(entry)
    assert cur > 1.00  # inflated
    assert cur < 2.00  # not absurd


def test_merge_prefers_primary():
    primary = {"170393": {"price_per_100g": 0.15, "price_source": "kroger"}}
    tfp = [TFPEntry(fndds_code=11111, tfp_category="Vegetables", pricing_method=1,
                    price_per_100g_2021=0.30)]
    crosswalk = {11111: 170393}
    merged = merge_price_sources(primary, tfp, crosswalk)
    assert merged["170393"]["price_source"] == "kroger"


def test_merge_fills_gap_with_tfp_via_crosswalk():
    primary = {}
    tfp = [TFPEntry(fndds_code=11111, tfp_category="Vegetables", pricing_method=1,
                    price_per_100g_2021=0.30)]
    crosswalk = {11111: 170393}
    merged = merge_price_sources(primary, tfp, crosswalk)
    assert "170393" in merged
    assert merged["170393"]["price_source"] == "tfp"
    assert "11111" in merged["170393"]["price_citation"]


def test_merge_keys_on_fndds_when_no_crosswalk():
    primary = {}
    tfp = [TFPEntry(fndds_code=11111, tfp_category="X", pricing_method=1,
                    price_per_100g_2021=0.30)]
    merged = merge_price_sources(primary, tfp, crosswalk={})
    assert "fndds:11111" in merged


def test_load_crosswalk_missing_file_empty(tmp_path: Path):
    assert load_fndds_fdc_crosswalk(tmp_path / "does-not-exist.csv") == {}


def test_load_crosswalk_parses(tmp_path: Path):
    p = tmp_path / "xw.csv"
    p.write_text("fndds_code,fdc_id\n11100000,170393\n11111000,170394\n")
    cw = load_fndds_fdc_crosswalk(p)
    assert cw == {11100000: 170393, 11111000: 170394}


def test_tally_sources_counts():
    prices = {
        "1": {"price_source": "kroger"},
        "2": {"price_source": "kroger"},
        "3": {"price_source": "tfp"},
    }
    assert tally_sources(prices) == {"kroger": 2, "tfp": 1}
