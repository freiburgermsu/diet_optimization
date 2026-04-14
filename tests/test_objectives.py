from dataclasses import dataclass

from diet_opt.objectives import ObjectiveConfig, build_secondary_term, sodium_contribution


@dataclass
class FakeVar:
    name: str


def test_sodium_contribution_present():
    matches = {"Celery": {"Sodium": 80.0, "Iron": 0.2}}
    assert sodium_contribution(matches, "Celery") == 80.0


def test_sodium_contribution_missing():
    assert sodium_contribution({}, "Unicorn") == 0.0
    assert sodium_contribution({"Apple": {"Iron": 0.1}}, "Apple") == 0.0


def test_no_terms_when_disabled():
    terms = build_secondary_term({}, {}, {}, ObjectiveConfig())
    assert terms == []


def test_sodium_term_emits_per_food():
    food_info = {"Celery": {}, "Apple": {}}
    matches = {"Celery": {"Sodium": 80.0}, "Apple": {"Sodium": 1.0}}
    variables = {"Celery": FakeVar("Celery"), "Apple": FakeVar("Apple")}
    cfg = ObjectiveConfig(minimize_sodium_weight=0.01)
    terms = build_secondary_term(food_info, matches, variables, cfg)
    assert len(terms) == 2
    # Verify the multiplier is weight * per-food-sodium
    mul_elements = [t["elements"][0]["elements"] for t in terms]
    coefs = sorted(m[1] for m in mul_elements)
    assert coefs == [0.01, 0.80]


def test_sodium_term_skips_zero_sodium_foods():
    food_info = {"Celery": {}, "Water": {}}
    matches = {"Celery": {"Sodium": 80.0}, "Water": {"Sodium": 0.0}}
    variables = {"Celery": FakeVar("Celery"), "Water": FakeVar("Water")}
    cfg = ObjectiveConfig(minimize_sodium_weight=0.5)
    terms = build_secondary_term(food_info, matches, variables, cfg)
    assert len(terms) == 1
