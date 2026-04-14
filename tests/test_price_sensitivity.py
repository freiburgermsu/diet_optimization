from diet_opt.price_sensitivity import (
    PriceTriplet,
    Scenario,
    format_cost_range,
    rank_spread_drivers,
    sweep_prices,
)


def test_sweep_runs_three_scenarios():
    food_info = {"Carrots": {}, "Beans": {}}
    triplets = {
        "Carrots": PriceTriplet(0.10, 0.14, 0.20),
        "Beans": PriceTriplet(0.30, 0.40, 0.55),
    }

    calls = []

    def fake_solve(prices):
        calls.append(prices.copy())
        return sum(prices.values()), {k: 1.0 for k in prices}

    scenarios = sweep_prices(food_info, triplets, fake_solve)
    assert [s.label for s in scenarios] == ["low", "median", "high"]
    assert calls[0]["Carrots"] == 0.10
    assert calls[2]["Beans"] == 0.55


def test_rank_spread_drivers_weights_by_grams():
    low = Scenario("low", 0, {}, {"Cheap": 1.00, "Pricey": 0.10})
    med = Scenario("median", 0, {"Cheap": 5, "Pricey": 200}, {"Cheap": 1.00, "Pricey": 0.15})
    high = Scenario("high", 0, {}, {"Cheap": 3.00, "Pricey": 0.20})
    drivers = rank_spread_drivers([low, med, high])
    # Cheap: spread=$2 * 5g = 10
    # Pricey: spread=$0.10 * 200g = 20  (should dominate)
    assert drivers[0][0] == "Pricey"


def test_rank_ignores_zero_spread():
    low = Scenario("low", 0, {}, {"Stable": 1.00})
    med = Scenario("median", 0, {"Stable": 100}, {"Stable": 1.00})
    high = Scenario("high", 0, {}, {"Stable": 1.00})
    assert rank_spread_drivers([low, med, high]) == []


def test_format_cost_range():
    scenarios = [
        Scenario("low", 7.50, {}, {}),
        Scenario("median", 9.68, {}, {}),
        Scenario("high", 12.30, {}, {}),
    ]
    out = format_cost_range(scenarios)
    assert "$7.50" in out
    assert "$12.30" in out
    assert "9.68" in out
