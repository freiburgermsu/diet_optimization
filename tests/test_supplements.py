from diet_opt.supplements import load_supplements, supplement_nutrient_contribution


def test_load_supplements_has_centrum():
    sups = load_supplements()
    keys = [s.key for s in sups]
    assert "centrum_adult" in keys


def test_centrum_nutrient_vector_nontrivial():
    sups = load_supplements()
    centrum = next(s for s in sups if s.key == "centrum_adult")
    assert centrum.nutrients["Vitamin B12"] == 25
    assert centrum.nutrients["Iron"] == 8
    assert centrum.price_per_dose == 0.10


def test_counts_against_volume_false():
    sups = load_supplements()
    assert all(s.counts_against_volume is False for s in sups)


def test_nutrient_contribution_present():
    sups = load_supplements()
    centrum = next(s for s in sups if s.key == "centrum_adult")
    assert supplement_nutrient_contribution(centrum, "Vitamin B12") == 25
    assert supplement_nutrient_contribution(centrum, "Kryptonite") == 0.0
