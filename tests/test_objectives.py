from dataclasses import dataclass

from diet_opt.objectives import (
    ObjectiveConfig,
    build_carbon_ceiling_constraint,
    build_secondary_term,
    load_carbon_footprint,
    load_polyphenol_content,
    pareto_sweep_points,
    sodium_contribution,
)


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
    coefs = sorted(t["elements"][0]["elements"][1] for t in terms)
    assert coefs == [0.01, 0.80]


def test_sodium_term_skips_zero_sodium_foods():
    food_info = {"Celery": {}, "Water": {}}
    matches = {"Celery": {"Sodium": 80.0}, "Water": {"Sodium": 0.0}}
    variables = {"Celery": FakeVar("Celery"), "Water": FakeVar("Water")}
    cfg = ObjectiveConfig(minimize_sodium_weight=0.5)
    terms = build_secondary_term(food_info, matches, variables, cfg)
    assert len(terms) == 1


# --- Polyphenols (ORAC replacement) ---

def test_load_polyphenol_content_has_blueberries():
    pp = load_polyphenol_content()
    assert pp["blueberries"] == 836


def test_polyphenol_csv_comments_ignored():
    pp = load_polyphenol_content()
    assert "# Polyphenol content" not in pp


def test_polyphenol_term_is_negative_coefficient():
    food_info = {"Blueberries": {}}
    variables = {"Blueberries": FakeVar("Blueberries")}
    pp = {"Blueberries": 836.0}
    cfg = ObjectiveConfig(maximize_polyphenols_weight=0.01)
    terms = build_secondary_term(food_info, {}, variables, cfg, polyphenols=pp)
    assert len(terms) == 1
    coef = terms[0]["elements"][0]["elements"][1]
    assert coef < 0   # maximize → negative in a minimize objective
    assert abs(coef - (-0.01 * 836.0)) < 1e-9


def test_sodium_and_polyphenol_terms_both_emit():
    food_info = {"Blueberries": {}, "Celery": {}}
    matches = {"Celery": {"Sodium": 80}}
    variables = {f.replace(" ", "_"): FakeVar(f) for f in food_info}
    pp = {"Blueberries": 836.0}
    cfg = ObjectiveConfig(minimize_sodium_weight=0.5, maximize_polyphenols_weight=0.01)
    terms = build_secondary_term(food_info, matches, variables, cfg, polyphenols=pp)
    # One sodium term (Celery) + one polyphenol term (Blueberries)
    assert len(terms) == 2


# --- Carbon footprint (ε-constraint) ---

def test_load_carbon_footprint_has_beef_and_pulses():
    cf = load_carbon_footprint()
    assert cf["Beef"] > 90
    assert cf["Lentils"] < 1.5


def test_build_carbon_ceiling_constraint_shape():
    food_info = {"Beef": {}, "Lentils": {}}
    variables = {"Beef": FakeVar("Beef"), "Lentils": FakeVar("Lentils")}
    carbon = {"Beef": 100.0, "Lentils": 1.0}
    c = build_carbon_ceiling_constraint(food_info, variables, carbon, ceiling_kg_co2e=2.5)
    assert c["name"] == "carbon_footprint_ceiling"
    assert c["ub"] == 2.5
    # Variable units 1.0 = 100g = 0.1kg, so coefficient is kg_co2e_per_kg * 0.1
    coefs = sorted(el["elements"][1] for el in c["expr"]["elements"])
    assert coefs == [0.1, 10.0]


def test_build_carbon_ceiling_skips_unknown_foods():
    food_info = {"Beef": {}, "Martian_Weed": {}}
    variables = {"Beef": FakeVar("Beef"), "Martian_Weed": FakeVar("Martian_Weed")}
    carbon = {"Beef": 100.0}   # Martian_Weed absent
    c = build_carbon_ceiling_constraint(food_info, variables, carbon, 10.0)
    assert len(c["expr"]["elements"]) == 1


# --- Pareto sweep ---

def test_pareto_sweep_evenly_spaced():
    points = pareto_sweep_points([0.0, 10.0], num_points=5)
    assert points == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_pareto_sweep_handles_single_point():
    assert pareto_sweep_points([5.0], num_points=1) == [5.0]
    assert pareto_sweep_points([], num_points=10) == []
