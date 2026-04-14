"""Tests for extracting Kroger-friendly search terms from FDC descriptions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fetch_prices import (  # noqa: E402
    extract_search_term,
    load_terms_from_fdc_descriptions,
)


def test_first_segment_taken():
    assert extract_search_term("Carrots, raw whole") == "carrots"
    assert extract_search_term("Pinto beans, mature seeds, raw") == "pinto beans"
    assert extract_search_term("Beef, round, eye of round") == "beef"


def test_long_fdc_research_description():
    desc = 'Beef, round, eye of round, roast, separable lean only, trimmed to 1/8" fat, all grades, raw'
    assert extract_search_term(desc) == "beef"


def test_empty_returns_none():
    assert extract_search_term("") is None
    assert extract_search_term(None) is None
    assert extract_search_term(",,,") is None


def test_amino_acid_rows_skipped():
    desc = "Amino Acids, Chicken, dark meat, lean only, drumstick, raw, non-enhanced (CA2,NC1)"
    assert extract_search_term(desc) is None


def test_b12_reference_skipped():
    assert extract_search_term("B12") is None
    assert extract_search_term("b-12, fortified") is None


def test_load_from_live_file_has_many_terms():
    root = Path(__file__).resolve().parent.parent
    p = root / "fresh_foods_nutrients_names_physiology.json"
    if not p.exists():
        import pytest
        pytest.skip("live FDC JSON absent")
    terms = load_terms_from_fdc_descriptions(p)
    assert 200 < len(terms) < 500   # currently 386, give room for future refinements
    assert "carrots" in terms or "carrot" in terms
    assert "beef" in terms
    # FDC describes pinto beans as "Beans, pinto, mature seeds, raw" so the
    # first-segment extraction captures "beans" (pinto is too specific for
    # a retailer search anyway — Kroger's own search will surface the most
    # relevant bean product).
    assert "beans" in terms


def test_load_normalizes_plural_duplicates(tmp_path: Path):
    p = tmp_path / "test.json"
    p.write_text(json.dumps({
        "Apple, raw": {},
        "Apples, Gala, raw": {},   # would extract "apples" (plural of apple)
        "Carrots, raw": {},
    }))
    terms = load_terms_from_fdc_descriptions(p)
    # "apple" wins over "apples" when both are present
    assert "apple" in terms
    assert "apples" not in terms
    assert "carrots" in terms  # no singular competitor, stays


def test_load_without_plural_normalization(tmp_path: Path):
    p = tmp_path / "test.json"
    p.write_text(json.dumps({"Apple, raw": {}, "Apples, Gala": {}}))
    terms = load_terms_from_fdc_descriptions(p, normalize_plurals=False)
    assert "apple" in terms
    assert "apples" in terms
