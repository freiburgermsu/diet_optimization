"""TFP 2021 pricing fallback.

USDA's TFP 2021 Online Supplement gives per-FNDDS-food prices at
$/100g as-consumed in June 2021 dollars — no lb→100g conversion
needed, no gross→edible conversion (the "as-consumed" framing already
factors waste in). One job remains: inflate June 2021 → current via
BLS CPI "Food at home" (CUSR0000SAF11).

TFP data flow:
    TFP_2021_Online_Supplement.xlsx
        ↓  parse_tfp_xlsx.py
    data/tfp_prices.csv   [fndds_code, tfp_category, pricing_method,
                           price_per_100g_2021]
        ↓  this module
    {fdc_id: {price_per_100g, price_source: 'tfp', ...}}

FNDDS codes are not FDC IDs. The FNDDS→FDC crosswalk lives in
`data/fndds_fdc_crosswalk.csv` and is populated by follow-up work
(see #4 for the unified food table). Without the crosswalk, TFP
entries pass through but are keyed by FNDDS code — usable for
aggregate tallies but not for LP integration.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# CPI-U series "Food at home" (CUSR0000SAF11, BLS).
# Snapshot indexed values; bump the "current" entry when rebuilding prices.
CPI_SNAPSHOTS = {
    "2021-06": 268.473,
    "2024-10": 302.401,
    "2026-03": 314.200,   # placeholder — confirm and bump
}

DEFAULT_BASE = "2021-06"
DEFAULT_CURRENT = "2026-03"


@dataclass(frozen=True)
class TFPEntry:
    fndds_code: int
    tfp_category: str
    pricing_method: int
    price_per_100g_2021: float


def load_tfp(path: Path | str = "data/tfp_prices.csv") -> list[TFPEntry]:
    out: list[TFPEntry] = []
    with open(path) as f:
        for row in csv.DictReader(filter(lambda l: not l.startswith("#"), f)):
            out.append(TFPEntry(
                fndds_code=int(row["fndds_code"]),
                tfp_category=row["tfp_category"],
                pricing_method=int(row["pricing_method"]),
                price_per_100g_2021=float(row["price_per_100g_2021"]),
            ))
    return out


def inflate_cpi(
    value_base: float,
    base: str = DEFAULT_BASE,
    current: str = DEFAULT_CURRENT,
    cpi: dict[str, float] = CPI_SNAPSHOTS,
) -> float:
    return value_base * cpi[current] / cpi[base]


def tfp_price_current(entry: TFPEntry) -> float:
    """Return TFP's $/100g as-consumed inflated to current dollars."""
    return inflate_cpi(entry.price_per_100g_2021)


def load_fndds_fdc_crosswalk(path: Path | str) -> dict[int, int]:
    """Load FNDDS → FDC ID mapping. Returns {} if file absent."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        reader = csv.DictReader(filter(lambda l: not l.startswith("#"), f))
        return {int(r["fndds_code"]): int(r["fdc_id"]) for r in reader}


def merge_price_sources(
    primary_prices: dict[str, dict],
    tfp_entries: list[TFPEntry],
    crosswalk: dict[int, int] | None = None,
) -> dict[str, dict]:
    """Fill gaps in primary_prices with TFP entries, preserving provenance.

    Primary data wins; TFP entries only land when the target key
    (fdc_id as str) isn't already present. Each TFP record gets
    `price_source='tfp'` and a citation with its FNDDS code.

    Without a crosswalk, TFP entries key on FNDDS code with an `fndds:`
    prefix — still useful for aggregate tallies, not for LP joins.
    """
    crosswalk = crosswalk or {}
    out = {k: dict(v) for k, v in primary_prices.items()}
    for entry in tfp_entries:
        fdc_id = crosswalk.get(entry.fndds_code)
        key = str(fdc_id) if fdc_id is not None else f"fndds:{entry.fndds_code}"
        if key in out:
            continue
        out[key] = {
            "price_per_100g": tfp_price_current(entry),
            "price_source": "tfp",
            "price_citation": f"TFP 2021 Online Supplement, FNDDS code {entry.fndds_code}",
            "tfp_category": entry.tfp_category,
        }
    return out


def tally_sources(prices: dict[str, dict]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for entry in prices.values():
        src = entry.get("price_source", "unknown")
        tally[src] = tally.get(src, 0) + 1
    return tally
