"""Tests for the simple-product filter and winner-per-term selector."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from normalize_prices import (  # noqa: E402
    build_prices_by_term,
    description_simplicity_score,
    diagnose_dropped_terms,
    is_simple_product,
    select_winner_per_term,
)


# The exact noisy descriptions from the user's dry-run
NOISY_EXAMPLES = [
    "Green Giant Restaurant Style Honey Glazed Carrots with Sage Butter",
    "Kroger® Carrots Celery And Broccoli Snack Tray With Ranch Dip",
    "Carrots Sticks Cup",
    "Market Cuts Mix Carrots & Celery Cup",
]


def test_noisy_examples_rejected():
    for d in NOISY_EXAMPLES:
        assert not is_simple_product(d), f"should reject: {d}"


def test_plain_descriptions_accepted():
    assert is_simple_product("Organic Carrots")
    assert is_simple_product("Kroger Fresh Broccoli Crown")
    assert is_simple_product("Pinto Beans, dry")


def test_empty_description_rejected():
    assert not is_simple_product("")
    assert not is_simple_product(None)


def test_simplicity_score_favors_brevity():
    long_desc = "Organic Fresh Whole Raw Baby Carrots with Stems Attached From California"
    short_desc = "Organic Carrots"
    assert description_simplicity_score(short_desc) > description_simplicity_score(long_desc)


def test_simplicity_score_bonus_for_affirmatives():
    assert description_simplicity_score("Fresh Carrots") > description_simplicity_score("Carrots Packaged")


def test_winner_picks_cheapest_per_term():
    products = [
        {"search_term": "carrots", "description": "Organic Carrots", "price": 2.00, "package_size_g": 454},
        {"search_term": "carrots", "description": "Kroger Fresh Carrots", "price": 1.29, "package_size_g": 454},
        {"search_term": "carrots", "description": "Bulk Carrots", "price": 1.00, "package_size_g": 1000},
    ]
    winners = select_winner_per_term(products)
    w = winners["carrots"]
    # Bulk Carrots is cheapest per-100g ($0.10/100g vs $0.22 vs $0.28)
    assert w["description"] == "Bulk Carrots"


def test_winner_filters_prepared():
    products = [
        {"search_term": "carrots", "description": "Honey Glazed Carrots", "price": 0.50, "package_size_g": 454},
        {"search_term": "carrots", "description": "Kroger Fresh Carrots", "price": 1.29, "package_size_g": 454},
    ]
    winners = select_winner_per_term(products)
    assert winners["carrots"]["description"] == "Kroger Fresh Carrots"


def test_winner_skipped_when_all_prepared():
    products = [
        {"search_term": "carrots", "description": "Glazed Carrots", "price": 1, "package_size_g": 100},
        {"search_term": "carrots", "description": "Pickled Carrots", "price": 1, "package_size_g": 100},
    ]
    assert select_winner_per_term(products) == {}


def test_winner_no_filter_when_disabled():
    products = [
        {"search_term": "carrots", "description": "Glazed Carrots", "price": 0.50, "package_size_g": 454},
        {"search_term": "carrots", "description": "Fresh Carrots", "price": 1.00, "package_size_g": 454},
    ]
    winners = select_winner_per_term(products, filter_simple=False)
    # Without filter, cheapest wins
    assert winners["carrots"]["description"] == "Glazed Carrots"


def test_tiebreak_by_simplicity():
    products = [
        {"search_term": "carrots", "description": "Kroger Fresh Organic Whole Raw Baby Carrots with Ends",
         "price": 1.00, "package_size_g": 100},
        {"search_term": "carrots", "description": "Carrots",
         "price": 1.00, "package_size_g": 100},
    ]
    winners = select_winner_per_term(products)
    assert winners["carrots"]["description"] == "Carrots"


def test_build_prices_by_term_end_to_end():
    raw = {
        "fetched_at": "2026-04-14T09:00:00Z",
        "retailer": "kroger",
        "terms": ["carrots", "broccoli"],
        "products": [
            {"search_term": "carrots", "description": "Fresh Carrots",
             "price": 1.29, "package_size_g": 454},
            {"search_term": "carrots", "description": "Honey Glazed Carrots",
             "price": 0.50, "package_size_g": 454},
            {"search_term": "broccoli", "description": "Kroger Broccoli Crown",
             "price": 2.00, "package_size_g": 400},
        ],
    }
    out = build_prices_by_term(raw)
    assert set(out) == {"carrots", "broccoli"}
    assert out["carrots"]["raw_description"] == "Fresh Carrots"
    assert out["carrots"]["price_source"] == "kroger"
    assert abs(out["broccoli"]["price_per_100g"] - 0.50) < 0.01


def test_diagnose_flags_terms_with_results_but_no_winner():
    raw = {
        "terms": ["lentils", "quinoa"],
        "products": [
            {"search_term": "lentils", "description": "Lentil Soup Mix", "price": 1, "package_size_g": 100},
            {"search_term": "quinoa", "description": "Organic Quinoa", "price": 5, "package_size_g": 400},
        ],
    }
    winners = select_winner_per_term(raw["products"])
    dropped = diagnose_dropped_terms(raw, winners)
    assert "lentils" in dropped
    assert "quinoa" not in dropped
