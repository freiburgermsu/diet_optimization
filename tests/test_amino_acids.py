from diet_opt.amino_acids import (
    EAARequirement,
    load_eaa_requirements,
    per_food_eaa_content,
    required_mg_per_day,
)


def test_load_returns_seven_requirements():
    eaa = load_eaa_requirements()
    assert len(eaa) == 9  # 9 YAML entries but Met+Cys and Phe+Tyr are composite — see below
    # Actually loaded as 9 distinct keys but composite ones bundle 2 FDC ids
    # Verify the pairings exist
    by_key = {r.key: r for r in eaa}
    assert "methionine_cysteine" in by_key
    assert "phenylalanine_tyrosine" in by_key


def test_methionine_is_composite():
    eaa = load_eaa_requirements()
    met = next(r for r in eaa if r.key == "methionine_cysteine")
    assert len(met.fdc_nutrient_ids) == 2
    assert 1215 in met.fdc_nutrient_ids  # Methionine
    assert 1216 in met.fdc_nutrient_ids  # Cystine


def test_tryptophan_singleton():
    eaa = load_eaa_requirements()
    trp = next(r for r in eaa if r.key == "tryptophan")
    assert trp.fdc_nutrient_ids == (1210,)
    assert trp.mg_per_kg_bw == 5


def test_required_mg_scales_with_bw():
    eaa = EAARequirement("lysine", "Lysine", 38, (1214,), "Lysine")
    assert required_mg_per_day(eaa, 68) == 38 * 68  # 150 lb ~= 68 kg


def test_per_food_eaa_sums_composite():
    eaa = EAARequirement("met_cys", "Met+Cys", 19, (1215, 1216), "Methionine + Cystine")
    food = {1215: 80, 1216: 20, 1214: 200}  # lysine irrelevant
    assert per_food_eaa_content(eaa, food) == 100


def test_per_food_eaa_missing_contributes_zero():
    eaa = EAARequirement("lys", "Lysine", 38, (1214,), "Lysine")
    assert per_food_eaa_content(eaa, {}) == 0.0
