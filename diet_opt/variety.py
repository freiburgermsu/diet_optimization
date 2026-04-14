"""Variety constraints — per-food mass caps and food-group tagging.

Per-food absolute caps prevent the degenerate 500g-carrots + 500g-beans
outcome. Food-group diversity (≥N groups per week) is deferred until the
weekly model (#12) since a daily "all groups every day" constraint
forces unrealistic menus.
"""
from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_GROUPS_PATH = Path(__file__).resolve().parent.parent / "data" / "food_groups.yaml"

# Sensible per-group caps in grams/day — vegetables can go higher than nuts.
DEFAULT_CAPS_G = {
    "leafy_green": 300,
    "root_veg": 350,
    "cruciferous": 300,
    "legume": 250,
    "grain": 250,
    "nut_seed": 60,
    "fruit": 300,
    "other_veg": 300,
    "_default": 300,
}


def load_food_groups(path: Path | str = DEFAULT_GROUPS_PATH) -> dict[str, str]:
    """Return {food_name: group} — inverted from the YAML's group→[foods]."""
    with open(path) as f:
        groups = yaml.safe_load(f) or {}
    inverted: dict[str, str] = {}
    for group, foods in groups.items():
        for food in foods:
            if food in inverted:
                raise ValueError(f"{food} assigned to both {inverted[food]} and {group}")
            inverted[food] = group
    return inverted


def apply_caps(
    variables: dict,
    food_to_group: dict[str, str],
    caps_g: dict[str, float] = DEFAULT_CAPS_G,
) -> dict[str, int]:
    """Set per-food upper bounds from the group cap in grams.

    Variable units: 1.0 = 100 g. Foods missing from the group table take
    the `_default` cap.

    Returns a count of foods tagged per group.
    """
    counts: dict[str, int] = {}
    for food_key, var in variables.items():
        name = food_key.replace("_", " ")
        group = food_to_group.get(name, "_default")
        cap_g = caps_g.get(group, caps_g["_default"])
        var.ub = min(var.ub, cap_g / 100.0)
        counts[group] = counts.get(group, 0) + 1
    return counts
