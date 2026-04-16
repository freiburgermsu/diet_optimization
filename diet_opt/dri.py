"""Compute personalized DRI bounds from a user profile.

Scales the baseline nutrition.json (28yo active 150lb male) to the user's
demographics. Two patch sources:

1. **Closed-form formulas** for nutrients that scale smoothly with
   body mass / energy intake:
     - Energy:       Mifflin-St Jeor BMR × physical activity factor
     - Protein:      RDA 0.8 g/kg (up to 1.6 g/kg for very active)
     - Fiber:        14 g per 1000 kcal (IOM AI)
     - Total Water:  3.0 L/day + body-weight and activity scaling
     - Linoleic:     17 g/day (men) / 12 g/day (women) AI
     - Linolenic:    1.6 / 1.1 g/day

2. **IOM-bracketed lookups** for nutrients where sex and/or age cross
   discrete thresholds (iron drops for post-menopausal women, calcium
   rises at 51/71, etc.). Defined in data/dri_overrides_by_profile.yaml.

For nutrients not covered by either, the baseline nutrition.json value
is retained unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "data" / "dri_overrides_by_profile.yaml"

# IOM/WHO Physical Activity Level multipliers applied to BMR.
ACTIVITY_PAL = {
    "sedentary": 1.2,   # desk job, minimal exercise
    "light": 1.375,     # some daily walking
    "moderate": 1.55,   # 3-5 days/week moderate exercise
    "active": 1.725,    # 6-7 days/week strenuous
    "very_active": 1.9, # daily intense training + physical job
}

# Protein RDA range by activity.
PROTEIN_G_PER_KG = {
    "sedentary": 0.8,
    "light": 0.8,
    "moderate": 1.0,
    "active": 1.2,
    "very_active": 1.6,
}


@dataclass(frozen=True)
class UserProfile:
    sex: Literal["male", "female", "nonbinary"]
    age: int
    weight_kg: float
    height_cm: float
    activity: Literal["sedentary", "light", "moderate", "active", "very_active"]

    def __post_init__(self):
        if self.sex not in ("male", "female", "nonbinary"):
            raise ValueError(f"sex must be male/female/nonbinary, got {self.sex!r}")
        if not 1 <= self.age <= 120:
            raise ValueError(f"age out of range: {self.age}")
        if not 20 <= self.weight_kg <= 300:
            raise ValueError(f"weight_kg out of range: {self.weight_kg}")
        if not 100 <= self.height_cm <= 230:
            raise ValueError(f"height_cm out of range: {self.height_cm}")
        if self.activity not in ACTIVITY_PAL:
            raise ValueError(f"activity must be one of {list(ACTIVITY_PAL)}, got {self.activity!r}")


def mifflin_st_jeor_bmr(profile: UserProfile) -> float:
    """Basal metabolic rate in kcal/day (Mifflin-St Jeor equation).

    Sex offset: men +5, women −161. Nonbinary defaults to midpoint (−78).
    """
    base = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age
    offset = {"male": 5, "female": -161, "nonbinary": -78}[profile.sex]
    return base + offset


def energy_kcal(profile: UserProfile) -> float:
    """Total daily energy expenditure (TDEE) = BMR × PAL."""
    return mifflin_st_jeor_bmr(profile) * ACTIVITY_PAL[profile.activity]


def protein_rda_g(profile: UserProfile) -> float:
    """Protein requirement in g/day: weight × g/kg factor from activity."""
    return profile.weight_kg * PROTEIN_G_PER_KG[profile.activity]


def fiber_rda_g(energy: float) -> float:
    """Fiber AI: 14 g per 1000 kcal."""
    return energy / 1000.0 * 14.0


def water_rda_L(profile: UserProfile) -> float:
    """Total water intake AI in L/day.

    IOM: 3.7 L men, 2.7 L women for 19+ adults. Scale modestly with
    weight for large bodies and with activity for athletes.
    """
    base = {"male": 3.7, "female": 2.7, "nonbinary": 3.2}[profile.sex]
    # +0.5 L for very active / +0.25 L for active
    activity_adj = {"very_active": 0.5, "active": 0.25}.get(profile.activity, 0.0)
    return base + activity_adj


def linoleic_g(profile: UserProfile) -> float:
    """Linoleic acid (omega-6) AI: 17 g men / 12 g women."""
    return {"male": 17, "female": 12, "nonbinary": 14}[profile.sex]


def linolenic_g(profile: UserProfile) -> float:
    """α-Linolenic acid (omega-3) AI: 1.6 g men / 1.1 g women."""
    return {"male": 1.6, "female": 1.1, "nonbinary": 1.35}[profile.sex]


def load_profile_overrides(path: Path | str = DEFAULT_OVERRIDES_PATH) -> dict:
    """Load the bracketed profile-dependent overrides YAML."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def pick_bracket(brackets: list[dict], age: int) -> dict | None:
    """Given a list of age-keyed bracket dicts, return the one matching `age`.

    Brackets are sorted by min_age ascending; we pick the last one whose
    min_age ≤ age. Returns None if age falls below the earliest bracket.
    """
    matches = [b for b in brackets if b.get("min_age", 0) <= age]
    if not matches:
        return None
    matches.sort(key=lambda b: b["min_age"])
    return matches[-1]


def apply_profile(
    baseline_nutrition: dict,
    profile: UserProfile,
    overrides_path: Path | str = DEFAULT_OVERRIDES_PATH,
) -> dict:
    """Return a new nutrition dict with profile-specific DRI bounds applied.

    Patches in order:
      1. Energy, Protein, Total Fiber, Total Water, Linoleic, Linolenic
         from closed-form formulas.
      2. Bracketed overrides from YAML (iron/calcium/vit-D/zinc/etc.)
         keyed by sex + age bracket.

    Nutrients not mentioned by either path keep their baseline values.
    """
    result = {k: dict(v) for k, v in baseline_nutrition.items()}

    # --- Formula-scaled patches ---
    energy = energy_kcal(profile)
    _patch(result, "Energy", lb=int(energy * 0.85), ub=int(energy * 1.10), units="kcal")
    _patch(result, "Protein",
           lb=round(protein_rda_g(profile), 1),
           ub=round(protein_rda_g(profile) * 2.5, 1),   # safety tolerance
           units="grams")
    _patch(result, "Total Fiber", lb=round(fiber_rda_g(energy), 1),
           ub=100, units="grams")
    _patch(result, "Total Water", lb=0.37, ub=water_rda_L(profile),
           units="L")
    _patch(result, "Linoleic Acid",
           lb=linoleic_g(profile), ub=linoleic_g(profile) * 3, units="grams")
    _patch(result, "Linolenic Acid",
           lb=linolenic_g(profile), ub=linolenic_g(profile) * 3, units="grams")

    # --- Bracketed overrides ---
    overrides = load_profile_overrides(overrides_path)
    for nutrient, sex_map in overrides.items():
        brackets = sex_map.get(profile.sex) or sex_map.get("nonbinary") or []
        picked = pick_bracket(brackets, profile.age)
        if picked is None:
            continue
        _patch(
            result, nutrient,
            lb=picked["low_bound"], ub=picked["high_bound"],
            units=picked.get("units"),
        )

    return result


def _patch(nutrition: dict, name: str, *, lb: float, ub: float, units: str | None) -> None:
    """Upsert a nutrient's bounds. Adds a new entry if absent."""
    entry = nutrition.setdefault(name, {})
    entry["low_bound"] = lb
    entry["high_bound"] = ub
    if units is not None:
        entry["units"] = units
