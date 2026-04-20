#!/usr/bin/env python3
"""Validate and expand diet-opt citations via Claude API + CrossRef.

Three modes:
  validate   — check existing citations in app.py against CrossRef DOI registry
  expand     — ask Claude to suggest new citations for disease presets / nutrients
  both       — validate existing, then expand gaps

CrossRef API (https://api.crossref.org) is free, no auth, rate-limited to
50 req/s with a polite User-Agent.

Usage:
  # Validate existing hard-coded citations
  uv run python scripts/validate_citations.py validate

  # Expand: generate new citations for contexts missing coverage
  uv run python scripts/validate_citations.py expand --output data/expanded_citations.json

  # Both: validate + expand, write a full validated set
  uv run python scripts/validate_citations.py both --output data/validated_citations.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CROSSREF_API = "https://api.crossref.org/works"
USER_AGENT = "diet-opt/1.0 (https://github.com/freiburgermsu/diet_optimization; mailto:afreiburger@anl.gov)"
RATE_LIMIT_SEC = 0.1  # 10 req/s (well under CrossRef's 50/s polite limit)


@dataclass
class CitationResult:
    cite: str
    doi: str | None
    contexts: list[str]
    crossref_valid: bool | None = None     # None = not checked
    crossref_title: str | None = None
    title_match: bool | None = None        # does CrossRef title match cite text?
    notes: str = ""


# ---------------------------------------------------------------------------
# CrossRef validation
# ---------------------------------------------------------------------------

def validate_doi(doi: str) -> dict | None:
    """Query CrossRef for a DOI. Returns metadata dict or None if not found."""
    import requests

    if not doi:
        return None
    url = f"{CROSSREF_API}/{doi}"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("message", {})
        return None
    except Exception:
        return None


def title_similarity(a: str, b: str) -> float:
    """Simple word-overlap Jaccard similarity between two titles."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    union = wa | wb
    if not union:
        return 0.0
    return len(wa & wb) / len(union)


def validate_citation(cit: dict) -> CitationResult:
    """Validate a single citation dict against CrossRef."""
    result = CitationResult(
        cite=cit["cite"],
        doi=cit.get("doi"),
        contexts=cit.get("contexts", []),
    )
    if not result.doi:
        result.crossref_valid = None
        result.notes = "no DOI to validate"
        return result

    meta = validate_doi(result.doi)
    time.sleep(RATE_LIMIT_SEC)

    if meta is None:
        result.crossref_valid = False
        result.notes = "DOI not found in CrossRef"
        return result

    result.crossref_valid = True
    # Extract title from CrossRef
    titles = meta.get("title", [])
    result.crossref_title = titles[0] if titles else None

    if result.crossref_title:
        sim = title_similarity(result.cite, result.crossref_title)
        result.title_match = sim > 0.3  # loose threshold
        if not result.title_match:
            result.notes = f"title mismatch (similarity={sim:.2f}): CrossRef says '{result.crossref_title}'"
    else:
        result.title_match = None
        result.notes = "CrossRef returned no title"

    return result


# ---------------------------------------------------------------------------
# Claude expansion
# ---------------------------------------------------------------------------

EXPAND_SYSTEM = """You are a biomedical nutrition researcher. Given a disease
state or nutritional context, suggest 3-5 peer-reviewed papers that provide
the strongest evidence for specific dietary nutrient modifications.

For each paper, return a JSON array of objects with these fields:
- "cite": full citation in AMA format (authors, title, journal, year, volume, pages)
- "doi": the DOI string (e.g. "10.1234/example") or null if unknown
- "contexts": list of context tags this paper supports (e.g. ["diabetes", "low_carb"])
- "rationale": one sentence explaining what nutrient constraint it supports

Return ONLY the JSON array, no other text."""

EXPAND_PROMPT = """Suggest 3-5 peer-reviewed citations for the following nutritional context:

Context: {context}
Current constraints being applied: {constraints}

Focus on:
- Meta-analyses and systematic reviews when available
- Society guidelines (AHA, ADA, KDOQI, ESPEN, IOM, WHO)
- Papers published in high-impact journals (NEJM, Lancet, JAMA, BMJ, Circulation)
- Papers that specifically justify the NUMERICAL BOUNDS used (e.g. "sodium < 1500mg" or "protein 1.2g/kg")

Do NOT suggest papers I already have: {existing_dois}"""


def expand_citations_for_context(
    client,
    context: str,
    constraints: str,
    existing_dois: list[str],
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """Ask Claude to suggest citations for a context."""
    prompt = EXPAND_PROMPT.format(
        context=context,
        constraints=constraints,
        existing_dois=", ".join(existing_dois) if existing_dois else "none",
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            system=EXPAND_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Extract JSON array
        if text.startswith("["):
            return json.loads(text)
        # Try to find JSON in the response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return []
    except Exception as e:
        print(f"  Claude API error for {context}: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_existing_citations() -> list[dict]:
    """Import the hard-coded citations from app.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # Can't import directly due to FastAPI dependency; parse the source
    app_path = Path(__file__).resolve().parent.parent / "diet_opt" / "web" / "app.py"
    source = app_path.read_text()
    # Find CITATIONS list
    start = source.find("CITATIONS: list[dict] = [")
    if start < 0:
        print("Could not find CITATIONS in app.py", file=sys.stderr)
        return []
    # Find the matching closing bracket
    depth = 0
    for i, ch in enumerate(source[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                break
    block = source[start:i + 1]
    # Extract just the list part
    eq_pos = block.find("= [")
    list_str = block[eq_pos + 2:]
    # Evaluate safely — these are simple dicts with string values
    return eval(list_str)  # noqa: S307 — trusted source (our own app.py)


def get_disease_presets() -> dict:
    """Import DISEASE_PRESETS from app.py."""
    app_path = Path(__file__).resolve().parent.parent / "diet_opt" / "web" / "app.py"
    source = app_path.read_text()
    start = source.find("DISEASE_PRESETS: dict[str, dict] = {")
    if start < 0:
        return {}
    depth = 0
    for i, ch in enumerate(source[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
    block = source[start:i + 1]
    eq_pos = block.find("= {")
    dict_str = block[eq_pos + 2:]
    return eval(dict_str)  # noqa: S307


def cmd_validate(args) -> int:
    """Validate existing citations against CrossRef."""
    citations = get_existing_citations()
    print(f"Validating {len(citations)} citations against CrossRef...\n")

    results: list[CitationResult] = []
    for cit in citations:
        r = validate_citation(cit)
        results.append(r)
        status = "OK" if r.crossref_valid else ("SKIP" if r.crossref_valid is None else "FAIL")
        match = ""
        if r.title_match is False:
            match = " [TITLE MISMATCH]"
        elif r.title_match is True:
            match = " [title OK]"
        print(f"  [{status}]{match} {r.cite[:80]}...")
        if r.notes:
            print(f"         {r.notes}")

    # Summary
    valid = sum(1 for r in results if r.crossref_valid is True)
    invalid = sum(1 for r in results if r.crossref_valid is False)
    skipped = sum(1 for r in results if r.crossref_valid is None)
    mismatch = sum(1 for r in results if r.title_match is False)
    print(f"\n{valid} valid, {invalid} invalid, {skipped} no DOI, {mismatch} title mismatches")

    if args.output:
        Path(args.output).write_text(json.dumps(
            [asdict(r) for r in results], indent=2
        ))
        print(f"Wrote results to {args.output}")

    return 1 if invalid > 0 else 0


def cmd_expand(args) -> int:
    """Use Claude to suggest new citations for disease presets."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY to use expand mode.", file=sys.stderr)
        return 1

    import anthropic
    client = anthropic.Anthropic()

    existing = get_existing_citations()
    existing_dois = [c.get("doi") for c in existing if c.get("doi")]
    existing_contexts = set()
    for c in existing:
        existing_contexts.update(c.get("contexts", []))

    presets = get_disease_presets()
    model = args.model or "claude-sonnet-4-6"

    all_new: list[dict] = []
    for disease, preset in presets.items():
        overrides = preset.get("overrides", [])
        constraints_str = ", ".join(
            f"{o['nutrient']} {o['bound']} {o['value']}" for o in overrides
        )
        flags = preset.get("flags", set())
        if flags:
            constraints_str += f" + flags: {', '.join(flags)}"

        print(f"\nExpanding: {disease} ({constraints_str})")
        suggestions = expand_citations_for_context(
            client, disease, constraints_str, existing_dois, model=model,
        )
        for s in suggestions:
            s.setdefault("contexts", [disease])
            all_new.append(s)
            print(f"  + {s['cite'][:80]}...")

    # Validate new citations against CrossRef
    print(f"\nValidating {len(all_new)} new citations...")
    validated: list[CitationResult] = []
    for cit in all_new:
        r = validate_citation(cit)
        validated.append(r)
        status = "OK" if r.crossref_valid else ("SKIP" if r.crossref_valid is None else "FAIL")
        symbol = "+" if r.crossref_valid else "?"
        print(f"  [{status}] {symbol} {r.cite[:80]}...")

    valid = [r for r in validated if r.crossref_valid is True]
    invalid = [r for r in validated if r.crossref_valid is False]
    print(f"\n{len(valid)} validated, {len(invalid)} failed CrossRef check")

    if args.output:
        out = {
            "existing": existing,
            "new_validated": [asdict(r) for r in valid],
            "new_unvalidated": [asdict(r) for r in invalid],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Wrote to {args.output}")

    return 0


def cmd_both(args) -> int:
    """Validate existing + expand + validate new."""
    print("=== Phase 1: Validate existing ===")
    cmd_validate(args)
    print("\n=== Phase 2: Expand with Claude ===")
    return cmd_expand(args)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate and expand diet-opt citations")
    sub = p.add_subparsers(dest="cmd", required=True)

    val = sub.add_parser("validate", help="Validate existing citations against CrossRef")
    val.add_argument("--output", help="Write results JSON to this path")

    exp = sub.add_parser("expand", help="Use Claude to suggest new citations")
    exp.add_argument("--output", default="data/expanded_citations.json")
    exp.add_argument("--model", default=None, help="Claude model ID")

    both = sub.add_parser("both", help="Validate existing + expand + validate new")
    both.add_argument("--output", default="data/validated_citations.json")
    both.add_argument("--model", default=None, help="Claude model ID")

    args = p.parse_args()
    if args.cmd == "validate":
        return cmd_validate(args)
    elif args.cmd == "expand":
        return cmd_expand(args)
    elif args.cmd == "both":
        return cmd_both(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
