"""Price sensitivity: sweep low/median/high prices, report cost interval.

Per my comment on the issue, this runs three separate solves rather than
a parametric LP — the diet composition can change across the three, and
that composition change is the interesting finding ("when peppers spike,
solver drops them").

Spread-driver ranking: foods are ranked by
    (high_price - low_price) × grams_in_diet
not by price variance alone. A food with huge price variance that the
diet uses 5g of doesn't contribute to the spread.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceTriplet:
    low: float
    median: float
    high: float


@dataclass
class Scenario:
    label: str             # "low" / "median" / "high"
    cost: float
    diet: dict[str, float]  # food -> grams
    food_prices: dict[str, float]


def substitute_prices(food_info: dict, price_field: str = "price") -> dict[str, float]:
    """Return {food: price} from the current food_info dict."""
    return {f: v[price_field] for f, v in food_info.items()}


def sweep_prices(
    food_info: dict,
    price_triplets: dict[str, PriceTriplet],
    solve_fn,
) -> list[Scenario]:
    """Run `solve_fn` three times with low/median/high prices for each food.

    `solve_fn(prices: dict[str, float]) -> (cost, diet_grams: dict)` — the
    caller adapts their model builder + solver into this signature.
    """
    scenarios = []
    for label in ("low", "median", "high"):
        prices = {food: getattr(price_triplets[food], label) for food in food_info}
        cost, diet = solve_fn(prices)
        scenarios.append(Scenario(label, cost, diet, prices))
    return scenarios


def rank_spread_drivers(scenarios: list[Scenario], top_k: int = 5) -> list[tuple[str, float]]:
    """Rank foods by (high_price - low_price) * grams_in_median_diet.

    Returns [(food, driver_score_$_per_day), ...] sorted descending.
    """
    by_label = {s.label: s for s in scenarios}
    low, median, high = by_label["low"], by_label["median"], by_label["high"]
    drivers: list[tuple[str, float]] = []
    for food, grams in median.diet.items():
        spread = high.food_prices.get(food, 0) - low.food_prices.get(food, 0)
        if spread <= 0 or grams <= 0:
            continue
        drivers.append((food, spread * grams))
    drivers.sort(key=lambda t: -t[1])
    return drivers[:top_k]


def format_cost_range(scenarios: list[Scenario]) -> str:
    by_label = {s.label: s.cost for s in scenarios}
    return f"${by_label['low']:.2f}–${by_label['high']:.2f}/day (median ${by_label['median']:.2f})"
