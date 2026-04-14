import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from diet_opt.web.app import OptimizeRequest, create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<form" in r.text.lower()


def test_optimize_validates_input(client):
    # missing required fields
    r = client.post("/optimize", json={})
    assert r.status_code == 422


def test_optimize_rejects_bad_sex(client):
    r = client.post("/optimize", json={
        "age": 28, "sex": "unicorn", "weight_kg": 68, "activity_level": "active",
    })
    assert r.status_code == 422


def test_optimize_rejects_out_of_range_age(client):
    r = client.post("/optimize", json={
        "age": 200, "sex": "male", "weight_kg": 68, "activity_level": "active",
    })
    assert r.status_code == 422


def test_request_model_accepts_minimal():
    req = OptimizeRequest(age=28, sex="male", weight_kg=68, activity_level="moderate")
    assert req.blacklist == []
    assert req.whitelist == []
    assert req.budget_ceiling is None
