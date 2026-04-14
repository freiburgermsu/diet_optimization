"""Apply hand-vetted overrides to the raw DRI table.

Migrates the inline patches from notebook cell 13 into a diffable YAML
with per-entry citations. Override layering:
    raw DRI  ->  apply_overrides()  ->  validate_bounds()  ->  use
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "nutrition_overrides.yaml"


def load_overrides(path: Path | str = DEFAULT_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_overrides(nutrition: dict, overrides: dict) -> dict:
    """Return a copy of `nutrition` with override fields merged in.

    Overrides that name a nutrient not in `nutrition` are added as new
    entries (e.g. Carotenoids), which preserves cell-13 behavior.
    """
    merged = {k: dict(v) for k, v in nutrition.items()}
    applied = []
    for nutrient, override in overrides.items():
        entry = merged.setdefault(nutrient, {})
        changed = []
        for field, value in override.items():
            if field == "citation":
                continue
            if entry.get(field) != value:
                entry[field] = value
                changed.append(field)
        if changed:
            applied.append(f"{nutrient}: {','.join(changed)}")
    log.info("Applied %d overrides: %s", len(applied), applied)
    return merged
