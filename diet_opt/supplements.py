"""Model supplements (multivitamin etc.) as additional LP variables.

Supplements have a nutrient vector (from product label) and a per-dose
price. They join the LP as one continuous variable each (dose count per
day) with a reasonable upper bound (5 doses/day — anything higher is
absurd). The `counts_against_volume=False` flag keeps them out of the
cup-volume constraint and #11 group caps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "supplements.yaml"
MAX_DOSES_PER_DAY = 5


@dataclass(frozen=True)
class Supplement:
    key: str
    product: str
    price_per_dose: float
    nutrients: dict[str, float]
    counts_against_volume: bool = False
    dose_unit: str = "tablet"


def load_supplements(path: Path | str = DEFAULT_PATH) -> list[Supplement]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return [
        Supplement(
            key=key,
            product=entry["product"],
            price_per_dose=float(entry["price_per_dose"]),
            nutrients=dict(entry["nutrients"]),
            counts_against_volume=bool(entry.get("counts_against_volume", False)),
            dose_unit=entry.get("dose_unit", "tablet"),
        )
        for key, entry in raw.items()
    ]


def supplement_nutrient_contribution(supplement: Supplement, nutrient: str) -> float:
    """Return this supplement's contribution per dose for the named nutrient."""
    return supplement.nutrients.get(nutrient, 0.0)
