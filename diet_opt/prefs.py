"""User preferences: blacklists, whitelists, allergen tags.

The prefs file is an untrusted input (will eventually come from the web
form in #16). Schema is validated and any unknown food identifier raises
— silent-drop here would translate to "you asked to exclude X but it was
ignored" on the UI, which is a correctness bug.

Once #4 (unified FDC-keyed schema) lands, `identifiers` should be fdc_id
integers. For now we match against the string keys used in food_info.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_MIN_WHITELIST_GRAMS = 30  # ~1/3 cup floor when a user demands a food


class InvalidPrefsError(ValueError):
    pass


@dataclass
class UserPrefs:
    blacklist: list[str] = field(default_factory=list)
    whitelist: list[str] = field(default_factory=list)
    whitelist_min_grams: dict[str, float] = field(default_factory=dict)
    allergen_tags: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path | str) -> "UserPrefs":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "UserPrefs":
        allowed = {"blacklist", "whitelist", "whitelist_min_grams", "allergen_tags"}
        unknown = set(data) - allowed
        if unknown:
            raise InvalidPrefsError(f"unknown prefs fields: {unknown}")
        bl = list(data.get("blacklist", []))
        wl = list(data.get("whitelist", []))
        overlap = set(bl) & set(wl)
        if overlap:
            raise InvalidPrefsError(f"foods in both blacklist and whitelist: {overlap}")
        return cls(
            blacklist=bl,
            whitelist=wl,
            whitelist_min_grams=dict(data.get("whitelist_min_grams", {})),
            allergen_tags=list(data.get("allergen_tags", [])),
        )


def apply_prefs(
    variables: dict,
    prefs: UserPrefs,
    default_min_grams: float = DEFAULT_MIN_WHITELIST_GRAMS,
) -> dict:
    """Mutate `variables` (as built by model.build_model) to apply prefs.

    Returns a report dict describing what changed. Raises InvalidPrefsError
    if any blacklist/whitelist entry doesn't match a known variable.
    """
    # variables are keyed by `food.replace(" ", "_")`
    def _key(food: str) -> str:
        return food.replace(" ", "_")

    known = set(variables)
    missing = [f for f in prefs.blacklist + prefs.whitelist if _key(f) not in known]
    if missing:
        raise InvalidPrefsError(f"prefs reference unknown foods: {missing}")

    applied = {"excluded": [], "required": []}
    for food in prefs.blacklist:
        variables[_key(food)].ub = 0
        applied["excluded"].append(food)
    for food in prefs.whitelist:
        # grams in notebook are stored as f (scalar 0..5, each unit = 100g)
        min_grams = prefs.whitelist_min_grams.get(food, default_min_grams)
        variables[_key(food)].lb = min_grams / 100.0
        applied["required"].append((food, min_grams))
    return applied
