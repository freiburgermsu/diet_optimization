from diet_opt.solve import ShadowPrice, explain_shadow_prices


def test_explain_handles_upper_bound():
    shadows = [ShadowPrice("Energy", "upper", 3200.0, 0.15)]
    nutrition = {"Energy": {"units": "kcal"}}
    out = explain_shadow_prices(shadows, nutrition)
    assert len(out) == 1
    assert "Energy" in out[0]
    assert "upper" in out[0]
    assert "0.150" in out[0]


def test_explain_handles_lower_bound():
    shadows = [ShadowPrice("Fat", "lower", 65.0, 0.04)]
    nutrition = {"Fat": {"units": "grams"}}
    out = explain_shadow_prices(shadows, nutrition)
    assert "lower" in out[0]
    assert "Lowering" in out[0]


def test_explain_top_k():
    shadows = [ShadowPrice(f"c{i}", "upper", 1.0, float(i)) for i in range(10)]
    out = explain_shadow_prices(shadows, {}, top_k=3)
    assert len(out) == 3


def test_explain_unknown_nutrient_no_crash():
    shadows = [ShadowPrice("Mystery", "upper", 1.0, 0.01)]
    out = explain_shadow_prices(shadows, {})
    assert len(out) == 1
