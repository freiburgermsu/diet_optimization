"""Tests for the user-parameterization changes to fetch_prices.py."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fetch_prices import load_config, load_terms_from_food_info  # noqa: E402


def test_load_terms_from_food_info(tmp_path: Path):
    p = tmp_path / "food_info.json"
    p.write_text(json.dumps({"Carrots": {}, "Pinto beans": {}, "Blueberries": {}}))
    terms = load_terms_from_food_info(p)
    assert set(terms) == {"Carrots", "Pinto beans", "Blueberries"}


def test_load_terms_from_live_food_info():
    root = Path(__file__).resolve().parent.parent
    terms = load_terms_from_food_info(root / "food_info.json")
    # Sanity: the checked-in food_info has ~69 foods with nutrition data
    assert len(terms) >= 60
    assert "Carrots" in terms or "Broccoli" in terms


def test_load_config_location_arg_wins(monkeypatch):
    monkeypatch.setenv("KROGER_CLIENT_ID", "id")
    monkeypatch.setenv("KROGER_CLIENT_SECRET", "secret")
    monkeypatch.setenv("KROGER_LOCATION_ID", "envloc")
    cfg = load_config(location_id="arg_wins")
    assert cfg.location_id == "arg_wins"


def test_load_config_env_used_when_arg_absent(monkeypatch):
    monkeypatch.setenv("KROGER_CLIENT_ID", "id")
    monkeypatch.setenv("KROGER_CLIENT_SECRET", "secret")
    monkeypatch.setenv("KROGER_LOCATION_ID", "envloc")
    cfg = load_config()
    assert cfg.location_id == "envloc"


def test_load_config_errors_without_location(monkeypatch):
    monkeypatch.setenv("KROGER_CLIENT_ID", "id")
    monkeypatch.setenv("KROGER_CLIENT_SECRET", "secret")
    monkeypatch.delenv("KROGER_LOCATION_ID", raising=False)
    with pytest.raises(SystemExit, match="location_id"):
        load_config()


def test_load_config_errors_without_client_id(monkeypatch):
    monkeypatch.delenv("KROGER_CLIENT_ID", raising=False)
    monkeypatch.setenv("KROGER_CLIENT_SECRET", "s")
    with pytest.raises(SystemExit, match="KROGER_CLIENT_ID"):
        load_config(location_id="x")
