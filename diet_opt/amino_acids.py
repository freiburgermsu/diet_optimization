"""Essential amino acid DRI constraints.

Replaces the single bulk-protein floor from the notebook with 7 EAA
constraints. Classic plant-diet failure modes surface: all-legume
infeasible on methionine, all-grain infeasible on lysine.

Met+Cys and Phe+Tyr are DRI-paired so this is 7 constraints, not 9.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "amino_acids.yaml"


@dataclass(frozen=True)
class EAARequirement:
    key: str
    display_name: str
    mg_per_kg_bw: float
    fdc_nutrient_ids: tuple[int, ...]
    fdc_name: str


def load_eaa_requirements(path: Path | str = DEFAULT_PATH) -> list[EAARequirement]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return [
        EAARequirement(
            key=key,
            display_name=key.replace("_", " ").title(),
            mg_per_kg_bw=float(e["mg_per_kg_bw"]),
            fdc_nutrient_ids=tuple(e["fdc_nutrient_ids"]),
            fdc_name=e["fdc_name"],
        )
        for key, e in raw.items()
    ]


def required_mg_per_day(eaa: EAARequirement, body_weight_kg: float) -> float:
    return eaa.mg_per_kg_bw * body_weight_kg


def per_food_eaa_content(
    eaa: EAARequirement,
    food_nutrients: dict[int, float],
) -> float:
    """Sum of component nutrients (e.g. Met + Cys) per 100g edible.

    `food_nutrients` is a dict keyed on FDC nutrient ID. Missing IDs
    contribute zero (FDC sparsity is common for minor amino acids).
    """
    return sum(food_nutrients.get(nid, 0.0) for nid in eaa.fdc_nutrient_ids)
