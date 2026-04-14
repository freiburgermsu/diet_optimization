"""Tests for extracting Kroger-friendly search terms from FDC descriptions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fetch_prices import (  # noqa: E402
    extract_search_term,
    load_terms_from_fdc_descriptions,
)


def test_plain_category_no_qualifier():
    # segment[1] = "raw" is a terminator → just "carrots"
    assert extract_search_term("Carrots, raw whole") == "carrots"


def test_prepends_variety_qualifier():
    # segment[1] = "pinto" is a variety → "pinto beans"
    assert extract_search_term("Beans, pinto, mature seeds, raw") == "pinto beans"
    assert extract_search_term("Rice, brown, long-grain, raw") == "brown rice"
    assert extract_search_term("Beef, ground, 85% lean, raw") == "ground beef"
    assert extract_search_term("Chicken, breast, raw") == "breast chicken"  # inversion artifact


def test_multi_word_qualifier_skipped_if_too_long():
    # "broilers or fryers" → 3 words, don't prepend → just "chicken"
    assert extract_search_term("Chicken, broilers or fryers, meat only") == "chicken"


def test_long_fdc_research_description():
    # segment[1] = "round" (cut name, 1 word, alpha) → "round beef"
    desc = 'Beef, round, eye of round, roast, separable lean only, trimmed to 1/8" fat, all grades, raw'
    assert extract_search_term(desc) == "round beef"


def test_nutrient_headed_rows_recover_the_food():
    """Previously these were dropped; now we shift past the nutrient
    header and keep the food."""
    assert extract_search_term("Total Fat, Ground turkey, 93% lean, raw (NY1)") == "ground turkey"
    assert extract_search_term("Niacin, Chicken breast, raw") == "chicken breast"
    # seg[1] = "belly" is a cut name → prepended
    assert extract_search_term("Cholesterol, Pork, belly, raw") == "belly pork"
    # seg[1] = "dark meat" (2 words, alphabetic) → prepended
    assert extract_search_term("Amino Acids, Chicken, dark meat, lean only") == "dark meat chicken"


def test_multiple_nutrient_headers_stripped():
    # Hypothetical double-header — loop must handle it
    assert extract_search_term("Total Fat, Protein, Chicken breast, raw") == "chicken breast"


def test_hyphenated_nutrient_header_stripped():
    # FDC research rows like "Cholesterol-WT, Chicken, skin, raw"
    assert extract_search_term("Cholesterol-WT, Chicken, skin, raw") == "skin chicken"
    # "Cholesterol - Beef, Top Loin, raw": we strip the whole first segment
    # (including "Beef"), landing on "top loin". Kroger search will surface
    # top loin products in practice.
    assert extract_search_term("Cholesterol - Beef, Top Loin, raw") == "top loin"


def test_empty_returns_none():
    assert extract_search_term("") is None
    assert extract_search_term(None) is None
    assert extract_search_term(",,,") is None


def test_amino_acid_rows_recover_food():
    # Old behavior returned None; now we shift past "Amino Acids"
    desc = "Amino Acids, Chicken, dark meat, lean only, drumstick, raw, non-enhanced (CA2,NC1)"
    assert extract_search_term(desc) == "dark meat chicken"


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
    # Enhanced extraction yields ~700-800 terms (vs 386 with first-segment-only)
    assert 500 < len(terms) < 1000
    assert "carrots" in terms or "carrot" in terms
    assert "beef" in terms
    # Variety qualifiers now land as precise search terms
    assert "pinto beans" in terms
    assert "brown rice" in terms
    assert "black beans" in terms
    assert "kidney beans" in terms
    # Nutrient-headed rows now resolve to their actual food
    assert "ground turkey" in terms
    # And the nutrient header itself never appears as a term
    assert not any(t.startswith("total fat") for t in terms)
    assert not any(t.startswith("niacin ") for t in terms)
    assert "amino acids" not in terms
    assert "cholesterol" not in terms


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
    # With enhanced extraction, "Apples, Gala" → "gala apples" (not "apples")
    p = tmp_path / "test.json"
    p.write_text(json.dumps({"Apples, raw": {}, "Apples, Gala": {}}))
    terms = load_terms_from_fdc_descriptions(p, normalize_plurals=False)
    assert "apples" in terms
    assert "gala apples" in terms
