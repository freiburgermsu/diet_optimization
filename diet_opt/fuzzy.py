"""Fuzzy matcher for USDA variant food names.

Replaces the 27-entry `food_simplification` dict hardcoded in notebook
cell 2. New USDA releases or new variant strings route to
`unresolved_matches.csv` for manual review — they are never silently
dropped.

Confirmed mappings live in `data/food_name_mapping.csv` with a
`confirmed_by` / `confirmed_at` audit trail; this module never mutates
that file.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

try:
    from rapidfuzz import fuzz, process
except ImportError as e:
    raise ImportError(
        "rapidfuzz required; install with `pip install rapidfuzz` or "
        "`pip install -e .[dev]`"
    ) from e


@dataclass(frozen=True)
class Match:
    variant: str
    canonical: str | None
    score: float


def match_food_names(
    variants: list[str],
    canonical: list[str],
    threshold: float = 85.0,
) -> tuple[dict[str, str], list[Match]]:
    """Match each variant to its best canonical name.

    Returns:
        (matched, unresolved) where `matched` is {variant: canonical} for
        scores >= threshold and `unresolved` is a list of Match objects
        (best candidate + score) for scores below threshold.
    """
    matched: dict[str, str] = {}
    unresolved: list[Match] = []
    for variant in variants:
        hit = process.extractOne(variant, canonical, scorer=fuzz.WRatio)
        if hit is None:
            unresolved.append(Match(variant, None, 0.0))
            continue
        cand, score, _ = hit
        if score >= threshold:
            matched[variant] = cand
        else:
            unresolved.append(Match(variant, cand, score))
    return matched, unresolved


def write_unresolved(unresolved: list[Match], path: Path | str = "unresolved_matches.csv") -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "best_candidate", "score"])
        for m in unresolved:
            w.writerow([m.variant, m.canonical or "", f"{m.score:.1f}"])


def load_confirmed_mapping(path: Path | str) -> dict[str, str]:
    """Load the human-confirmed mapping CSV.

    Expected columns: variant, canonical, confirmed_by, confirmed_at.
    Only `variant` and `canonical` are consumed; the audit columns exist
    so git blame can answer "who approved this mapping".
    """
    mapping: dict[str, str] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["variant"]] = row["canonical"]
    return mapping
