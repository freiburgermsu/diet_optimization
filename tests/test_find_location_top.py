"""Tests for --top on the find-location subcommand.

We don't hit the live API; we monkeypatch find_locations + get_access_token.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_prices  # noqa: E402


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("KROGER_CLIENT_ID", "id")
    monkeypatch.setenv("KROGER_CLIENT_SECRET", "secret")


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setattr(fetch_prices, "get_access_token", lambda cfg: "faketoken")
    store_list = [
        {
            "locationId": "01400943",
            "name": "Kroger - Downtown",
            "address": {
                "addressLine1": "123 Main St", "city": "Chicago",
                "state": "IL", "zipCode": "60601"
            },
        },
        {
            "locationId": "01400944",
            "name": "Mariano's - Uptown",
            "address": {
                "addressLine1": "456 Oak Ave", "city": "Chicago",
                "state": "IL", "zipCode": "60602"
            },
        },
    ]
    monkeypatch.setattr(
        fetch_prices, "find_locations",
        lambda token, zip_code, radius_miles=10, limit=20: store_list,
    )


def test_top_prints_single_location_id(fake_env, fake_api, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fetch_prices.py", "find-location", "--zip", "60601", "--top"])
    rc = fetch_prices.main()
    assert rc == 0
    captured = capsys.readouterr()
    # stdout is a single locationId (pipe-friendly)
    assert captured.out.strip() == "01400943"
    # a helpful comment goes to stderr
    assert "Kroger - Downtown" in captured.err


def test_without_top_prints_all(fake_env, fake_api, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fetch_prices.py", "find-location", "--zip", "60601"])
    fetch_prices.main()
    out = capsys.readouterr().out
    assert "01400943" in out
    assert "01400944" in out
    assert "Kroger - Downtown" in out


def test_top_no_stores_exits_nonzero(fake_env, monkeypatch, capsys):
    monkeypatch.setattr(fetch_prices, "get_access_token", lambda cfg: "t")
    monkeypatch.setattr(fetch_prices, "find_locations", lambda *a, **kw: [])
    monkeypatch.setattr(sys, "argv", ["fetch_prices.py", "find-location", "--zip", "99999", "--top"])
    assert fetch_prices.main() == 2
    assert "no Kroger stores" in capsys.readouterr().err
