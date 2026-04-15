#!/usr/bin/env python3
"""Categorize Claude-ranker null matches by why they failed.

Buckets:
  wrong_foods     — Kroger returned wrong foods (needs TFP fallback)
  only_processed  — Food exists only in processed form at retailer
  bad_extraction  — FDC extractor produced a malformed search term
  zero_candidates — Search returned nothing matching (rare here —
                    zero-candidate terms don't enter the ranker at all)
  other           — Uncategorized; review manually

Usage:
    python scripts/bucket_claude_nulls.py [--cache-dir cache/claude]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WRONG_FOOD_MARKERS = (
    "different", "other varieties", "other species", "wrong food",
    "not the same", "unrelated",
)
PROCESSED_MARKERS = (
    "nectar", "syrup", "sausage", "seasoned", "snack", "prepared",
    "processed", "flavored", "canned in", "sweetened", "dried", "flakes",
    "paste", "powder", "juice", "extract", "roasted", "candied",
)
BAD_EXTRACTION_MARKERS = (
    "personal care", "coffee brand", "corn-based", "body wash", "hairspray",
    "brand name", "brand collision", "not the",
)


def classify(reason: str) -> str:
    r = (reason or "").lower()
    if any(m in r for m in BAD_EXTRACTION_MARKERS):
        return "bad_extraction"
    if any(m in r for m in WRONG_FOOD_MARKERS):
        return "wrong_foods"
    if any(m in r for m in PROCESSED_MARKERS):
        return "only_processed"
    if "no candidate" in r or "empty list" in r:
        return "zero_candidates"
    return "other"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="cache/claude")
    p.add_argument(
        "--show", type=int, default=5,
        help="how many examples to print per bucket (default: 5)",
    )
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"cache directory not found: {cache_dir}", file=sys.stderr)
        return 1

    buckets: dict[str, list[tuple[str, str]]] = {
        "wrong_foods": [],
        "only_processed": [],
        "bad_extraction": [],
        "zero_candidates": [],
        "other": [],
    }

    total_files = 0
    null_count = 0
    for cache_file in sorted(cache_dir.glob("*.json")):
        total_files += 1
        result = json.load(open(cache_file))
        choice = result.get("choice", {})
        if choice.get("chosen_index") is not None:
            continue
        null_count += 1
        term = result.get("term", cache_file.stem)
        reason = choice.get("reason", "")
        buckets[classify(reason)].append((term, reason))

    print(f"{null_count} null matches across {total_files} cached terms\n")
    for name, items in buckets.items():
        print(f"=== {name}: {len(items)} ===")
        for term, reason in items[: args.show]:
            print(f"  {term}: {reason[:80]}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
