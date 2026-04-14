"""TFP 2021 pricing fallback.

The TFP basket publishes $/lb at Jun-2019 dollars. Three conversions
land a value in the model's schema:

  1. Inflate 2019 → current via BLS CPI (series CUSR0000SAF11 = "Food at
     home"). Stored here as a static value; bump via data_sources.yaml
     when CPI updates.
  2. $/lb → $/100g: divide by 4.54 (1 lb = 453.59g ≈ 4.54 * 100g).
  3. $/100g gross → $/100g edible: divide by FDC yield (inedible peel,
     pit, core fraction).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# CPI-U series "Food at home" (CUSR0000SAF11). Snapshot from BLS — bump
# when you rebuild against current prices.
CPI_SNAPSHOTS = {
    "2019-06": 254.716,
    "2024-10": 302.401,   # placeholder — replace on next data refresh
}

DEFAULT_BASE = "2019-06"
DEFAULT_CURRENT = "2024-10"


@dataclass(frozen=True)
class TFPEntry:
    category: str
    representative_food: str
    price_per_lb_2019: float
    source_table: str
    source_page: str


def load_tfp(path: Path | str) -> list[TFPEntry]:
    out: list[TFPEntry] = []
    with open(path) as f:
        for row in csv.DictReader(filter(lambda l: not l.startswith("#"), f)):
            out.append(TFPEntry(
                category=row["category"],
                representative_food=row["representative_food"],
                price_per_lb_2019=float(row["$_per_lb_2019"]),
                source_table=row["source_table"],
                source_page=row["source_page"],
            ))
    return out


def inflate_cpi(
    value_base: float,
    base: str = DEFAULT_BASE,
    current: str = DEFAULT_CURRENT,
    cpi: dict[str, float] = CPI_SNAPSHOTS,
) -> float:
    return value_base * cpi[current] / cpi[base]


def tfp_per_100g_edible(entry: TFPEntry, yield_factor: float = 1.0) -> float:
    """Convert TFP $/lb (2019) to $/100g edible (current dollars)."""
    per_lb_current = inflate_cpi(entry.price_per_lb_2019)
    per_100g_gross = per_lb_current / 4.54
    return per_100g_gross / max(yield_factor, 0.01)


def merge_price_sources(
    primary_prices: dict[str, dict],
    tfp_entries: list[TFPEntry],
    category_to_food: dict[str, str],
    yield_factors: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Fill gaps in primary_prices with TFP entries, preserving provenance.

    Returns a copy — primary data wins when both sources have an entry.
    Each TFP-sourced entry gains `price_source: 'tfp'` and a citation.
    """
    yields = yield_factors or {}
    out = {k: dict(v) for k, v in primary_prices.items()}
    for entry in tfp_entries:
        food = category_to_food.get(entry.category)
        if food is None or food in out:
            continue
        out[food] = {
            "price_per_100g": tfp_per_100g_edible(entry, yields.get(food, 1.0)),
            "price_source": "tfp",
            "price_citation": f"TFP 2021 {entry.source_table}, p. {entry.source_page}",
        }
    return out


def tally_sources(prices: dict[str, dict]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for entry in prices.values():
        src = entry.get("price_source", "unknown")
        tally[src] = tally.get(src, 0) + 1
    return tally
