"""Dietary preset expansion: preset name → excluded food list.

Presets (vegan, vegetarian, no-nuts, ...) are combinations of one or
more KEYWORD GROUPS. Each group has a list of lowercase substrings;
a food is excluded if any substring appears in its name.

Usage:
    preset_names = ["vegan", "gluten_free"]
    excluded = foods_excluded_by_presets(preset_names, food_names)
    # excluded is a list of foods to blacklist from the LP
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "dietary_groups.yaml"


def load_dietary_groups(path: Path | str = DEFAULT_PATH) -> dict:
    """Return the parsed dietary_groups.yaml as {groups: {...}, presets: {...}}."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def list_presets(cfg: dict | None = None) -> list[str]:
    """Return available preset names."""
    cfg = cfg or load_dietary_groups()
    return sorted(cfg.get("presets", {}))


def expand_preset(preset_name: str, cfg: dict) -> list[str]:
    """Return the list of group names a preset expands to.

    Raises KeyError if preset_name isn't defined.
    """
    return list(cfg["presets"][preset_name])


def keywords_for_preset(preset_name: str, cfg: dict) -> set[str]:
    """Return the union of keywords from all groups in a preset."""
    keywords: set[str] = set()
    for group_name in expand_preset(preset_name, cfg):
        keywords.update(cfg["groups"].get(group_name, []))
    return keywords


def foods_matching_keywords(
    food_names: list[str], keywords: set[str]
) -> list[str]:
    """Case-insensitive substring match; return matching food names."""
    out = []
    for food in food_names:
        lower = food.lower()
        if any(k in lower for k in keywords):
            out.append(food)
    return out


def foods_excluded_by_presets(
    preset_names: list[str],
    food_names: list[str],
    cfg: dict | None = None,
) -> list[str]:
    """Return the deduped list of foods excluded by the union of preset keywords.

    Raises ValueError if any preset name is unknown.
    """
    cfg = cfg or load_dietary_groups()
    known = set(cfg.get("presets", {}))
    unknown = [p for p in preset_names if p not in known]
    if unknown:
        raise ValueError(
            f"unknown preset(s): {unknown}. Available: {sorted(known)}"
        )
    all_keywords: set[str] = set()
    for p in preset_names:
        all_keywords.update(keywords_for_preset(p, cfg))
    return foods_matching_keywords(food_names, all_keywords)
