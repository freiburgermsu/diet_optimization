"""HTTP-layer tests for the diet-opt FastAPI app."""
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from diet_opt.web.app import MealPlanRequest, OptimizeRequest, create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "priced_foods_present" in body
    assert "anthropic_configured" in body


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<form" in r.text.lower()
    assert "diet-opt" in r.text.lower()


def test_presets_endpoint_returns_list(client):
    r = client.get("/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    for expected in ("vegan", "vegetarian", "nut_free", "gluten_free"):
        assert expected in presets


def test_foods_endpoint(client):
    r = client.get("/foods")
    assert r.status_code == 200
    # Live priced_foods.json present → non-empty list; absent → empty
    assert "foods" in r.json()


def test_optimize_request_schema_defaults():
    req = OptimizeRequest()
    assert req.min_serving_grams == 0.0
    assert req.dietary_presets == []
    assert req.blacklist == []


def test_optimize_rejects_out_of_range_min_serving(client):
    r = client.post("/optimize", json={"min_serving_grams": 500})
    assert r.status_code == 422


def test_optimize_rejects_unknown_preset(client):
    r = client.post("/optimize", json={
        "min_serving_grams": 0, "dietary_presets": ["notarealpreset"],
    })
    # 400 from our validation; 500 if priced_foods missing
    assert r.status_code in (400, 500)


def test_meal_plan_request_defaults():
    req = MealPlanRequest(diet={"carrots": 100})
    assert req.days == 7
    assert req.max_prep_min == 30
    assert req.model == "claude-haiku-4-5"


def test_meal_plan_rejects_out_of_range_days(client):
    r = client.post("/meal-plan", json={"diet": {"carrots": 100}, "days": 99})
    assert r.status_code == 422


def test_meal_plan_rejects_missing_diet(client):
    r = client.post("/meal-plan", json={})
    assert r.status_code == 422


def test_meal_plan_empty_diet_returns_400_or_500(client, monkeypatch):
    # Ensure ANTHROPIC_API_KEY is set so we get past the 500 auth check
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-xxxx")
    r = client.post("/meal-plan", json={"diet": {}})
    assert r.status_code in (400, 500)   # 400 for empty diet
