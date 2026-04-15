"""Unit tests for the pure helpers in claude_rank_products.py.

The LLM call itself is not exercised in CI — that needs a live API key.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _load():
    # Deferred import because the script imports anthropic at module level.
    import pytest
    pytest.importorskip("anthropic")
    pytest.importorskip("pydantic")
    import claude_rank_products as m
    return m


def test_slugify_roundtrip():
    m = _load()
    assert m.slugify("pinto beans") == "pinto_beans"
    assert m.slugify("Kroger® Whole Carrots") == "kroger_whole_carrots"
    assert m.slugify("   ") == "_unnamed"


def test_group_by_term_partitions_products():
    m = _load()
    raw = {
        "products": [
            {"description": "A", "search_term": "carrots"},
            {"description": "B", "search_term": "carrots"},
            {"description": "C", "search_term": "broccoli"},
        ]
    }
    groups = m.group_by_term(raw)
    assert len(groups["carrots"]) == 2
    assert len(groups["broccoli"]) == 1


def test_build_user_prompt_formats_candidates():
    m = _load()
    products = [
        {"description": "Kroger Carrots", "price": 1.29, "package_size_g": 454},
        {"description": "Baby Carrots", "price": 1.99, "package_size_g": 227},
    ]
    prompt = m.build_user_prompt("carrots", products)
    assert 'Term: "carrots"' in prompt
    assert "[0]" in prompt
    assert "[1]" in prompt
    assert "Kroger Carrots" in prompt
    assert "454g" in prompt
    assert "$1.29" in prompt


def test_build_user_prompt_zero_size():
    m = _load()
    products = [{"description": "Odd", "price": 1.00, "package_size_g": 0}]
    prompt = m.build_user_prompt("x", products)
    # Should not crash on divide-by-zero
    assert "0.000/100g" in prompt


def test_to_price_entry_with_chosen_product():
    m = _load()
    result = {
        "term": "carrots",
        "choice": {"chosen_index": 0, "reason": "fresh whole carrots", "confidence": "high"},
        "chosen_product": {
            "description": "Kroger Whole Carrots",
            "price": 3.59,
            "package_size_g": 2267.96,
        },
    }
    entry = m.to_price_entry(result, "kroger", "2026-04-14T00:00:00Z")
    assert entry is not None
    assert abs(entry["price_per_100g"] - 0.158) < 0.005
    assert entry["price_source"] == "kroger"
    assert entry["raw_description"] == "Kroger Whole Carrots"
    assert entry["claude_reason"] == "fresh whole carrots"
    assert entry["claude_confidence"] == "high"


def test_to_price_entry_null_chosen():
    m = _load()
    result = {
        "term": "caribou",
        "choice": {"chosen_index": None, "reason": "only coffee brand", "confidence": "high"},
        "chosen_product": None,
    }
    assert m.to_price_entry(result, "kroger", "t") is None


def test_to_price_entry_zero_size_returns_none():
    m = _load()
    result = {
        "term": "x",
        "choice": {"chosen_index": 0, "reason": "r", "confidence": "low"},
        "chosen_product": {"description": "X", "price": 1, "package_size_g": 0},
    }
    assert m.to_price_entry(result, "kroger", "t") is None


def test_system_prompt_exceeds_opus_cache_minimum():
    """Opus 4.6 requires ≥4096 tokens for prompt caching to activate.
    Shorter prompts silently don't cache — the API accepts the marker
    but `cache_creation_input_tokens` stays at 0 across requests."""
    m = _load()
    approx_tokens = len(m.SYSTEM_PROMPT) / 4  # ~4 chars per token
    assert approx_tokens > 4500, (
        f"system prompt only {approx_tokens:.0f} approx tokens — "
        "Opus 4.6 cache won't trigger (needs 4096)"
    )
