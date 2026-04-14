"""Handle nutrients with sparse FDC coverage.

Replaces the blanket `<6 foods → drop` rule in notebook cell 21 with a
per-nutrient policy driven by `data/sparse_nutrient_triage.yaml`. Four
tiers:

    supplement  → solver covers from supplements.yaml (#9)
    soft        → slack var + penalty in objective; solver can underfulfill
    impute      → fill missing values from food-group medians
    drop        → no constraint (upper-bound-only / non-required)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "sparse_nutrient_triage.yaml"


@dataclass(frozen=True)
class Triage:
    tier: str
    rationale: str
    soft_penalty: float | None


def load_triage(path: Path | str = DEFAULT_PATH) -> dict[str, Triage]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {
        name: Triage(tier=e["tier"], rationale=e["rationale"], soft_penalty=e.get("soft_penalty"))
        for name, e in raw.items()
    }


def categorize_nutrient(
    nutrient: str,
    foods_supporting: int,
    triage: dict[str, Triage],
    hard_threshold: int = 6,
) -> str:
    """Return one of {'hard', 'soft', 'impute', 'supplement', 'drop'}.

    A nutrient with sufficient support becomes 'hard' regardless of its
    triage entry (i.e. triage only kicks in when coverage is insufficient).
    """
    if foods_supporting >= hard_threshold:
        return "hard"
    if nutrient in triage:
        return triage[nutrient].tier
    return "drop"


def impute_from_group_medians(
    food_matches: dict[str, dict[str, float]],
    food_to_group: dict[str, str],
    nutrient: str,
) -> dict[str, float]:
    """Return {food: imputed_value} for foods missing `nutrient`.

    For each food lacking the nutrient, pick the median reported value
    from other foods in the same group. Foods whose group has no reported
    values are skipped.
    """
    from statistics import median

    group_values: dict[str, list[float]] = {}
    for food, nutrients in food_matches.items():
        if nutrient in nutrients:
            group = food_to_group.get(food)
            if group:
                group_values.setdefault(group, []).append(nutrients[nutrient])

    imputed: dict[str, float] = {}
    for food, nutrients in food_matches.items():
        if nutrient in nutrients:
            continue
        group = food_to_group.get(food)
        if group and group in group_values:
            imputed[food] = median(group_values[group])
    return imputed
