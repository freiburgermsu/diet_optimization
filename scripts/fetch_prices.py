#!/usr/bin/env python3
"""Retailer price scraper — Kroger Developer API.

Kroger runs an official OAuth2-gated public API (api.kroger.com) with a
free tier covering product search, which keeps us out of ToS gray zone
that Walmart/Trader Joe's scraping would create. Register an app at
https://developer.kroger.com/ to obtain CLIENT_ID / CLIENT_SECRET.

Output schema (`prices_raw.json`):

    {
      "fetched_at": "2026-04-14T09:00:00Z",
      "products": [
        {
          "upc": "0001111042850",
          "description": "Kroger Carrots",
          "price": 1.29,
          "package_size_g": 454,
          "location_id": "01400943"
        }, ...
      ]
    }

Normalization to $/100g edible (via FDC yields) happens in a second
pass, not here — keeps this script a thin I/O layer so it's cheap to
re-run without re-hitting FDC joins.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STALENESS_THRESHOLD_DAYS = 30

# Kroger API endpoints (public, documented).
OAUTH_URL = "https://api.kroger.com/v1/connect/oauth2/token"
PRODUCT_SEARCH_URL = "https://api.kroger.com/v1/products"


@dataclass(frozen=True)
class RetailerConfig:
    client_id: str
    client_secret: str
    location_id: str   # Kroger store ID; affects pricing since prices are location-specific


def load_config(location_id: str | None = None) -> RetailerConfig:
    """Load credentials from env. `location_id` precedence: arg > env > prompt."""
    try:
        client_id = os.environ["KROGER_CLIENT_ID"]
        client_secret = os.environ["KROGER_CLIENT_SECRET"]
    except KeyError as e:
        raise SystemExit(
            f"Missing env var: {e}. Register at https://developer.kroger.com/ "
            "and export KROGER_CLIENT_ID / KROGER_CLIENT_SECRET."
        )
    resolved = location_id or os.environ.get("KROGER_LOCATION_ID")
    if not resolved:
        raise SystemExit(
            "No location_id provided. Pass --location-id or set KROGER_LOCATION_ID. "
            "Find your nearest Kroger store ID via "
            "`scripts/fetch_prices.py find-location --zip 60601 --radius 10` "
            "or at https://developer.kroger.com/documentation/api-products/locations-public"
        )
    return RetailerConfig(
        client_id=client_id, client_secret=client_secret, location_id=resolved
    )


def find_locations(token: str, zip_code: str, radius_miles: int = 10, limit: int = 20) -> list[dict]:
    """Look up Kroger stores near a zip code. Use to pick a location_id."""
    import requests

    resp = requests.get(
        "https://api.kroger.com/v1/locations",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "filter.zipCode.near": zip_code,
            "filter.radiusInMiles": str(radius_miles),
            "filter.limit": str(limit),
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def load_terms_from_food_info(path: str) -> list[str]:
    """Pull search terms from a food_info.json-style dict's keys.

    Kroger's search accepts plain English — no tokenization needed. We
    pass each food's display name verbatim.
    """
    with open(path) as f:
        return list(json.load(f).keys())


# FDC research sub-sample rows start with a nutrient name rather than the
# food category. These aren't foods — they're lab-analysis panels we need
# to skip.
_NUTRIENT_HEADERS = {
    "amino acids",
    "total fat", "saturated fat", "trans fat",
    "cholesterol", "sugars", "total sugars", "added sugars",
    "carbohydrate", "protein", "fiber",
    "calcium", "iron", "magnesium", "phosphorus", "potassium",
    "sodium", "zinc", "copper", "manganese", "selenium", "fluoride",
    "vitamin a", "vitamin c", "vitamin d", "vitamin e", "vitamin k",
    "vitamin b-6", "vitamin b6", "vitamin b-12", "vitamin b12",
    "thiamin", "riboflavin", "niacin", "folate", "pantothenic acid",
    "choline", "water", "ash", "energy", "carbohydrates",
    "fatty acids",
}
_SKIP_EXACT = {"b12", "b-12", "b6", "b-6"}

# Terminators: if segment[1] is one of these, don't prepend it to segment[0].
# These are processing/state descriptors, not variety qualifiers.
_SEGMENT_TERMINATORS = {
    "raw", "cooked", "boiled", "roasted", "grilled", "fried", "baked",
    "steamed", "stewed", "broiled", "sauteed",
    "mature", "dry", "dried", "fresh", "frozen", "canned",
    "whole", "sliced", "chopped", "diced",
    "prepared", "unprepared",
}


def _slugify(term: str) -> str:
    """Filesystem-safe slug: lowercase, spaces→'_', drops non [a-z0-9_-]."""
    import re
    s = term.lower().strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_-]+", "", s)
    return s or "_unnamed"


def _is_nutrient_header(segment: str) -> bool:
    """True if this segment looks like a nutrient heading.

    Handles exact match ("cholesterol") and common suffix patterns
    ("cholesterol-wt", "cholesterol - beef").
    """
    if segment in _NUTRIENT_HEADERS:
        return True
    for h in _NUTRIENT_HEADERS:
        if segment.startswith(h):
            rest = segment[len(h):]
            if rest and not rest[0].isalpha():  # hyphen, space, dash, etc.
                return True
    return False


def extract_search_term(fdc_description: str) -> str | None:
    """Turn an FDC description into a retailer-friendly search term.

    Rules:
      1. If segment[0] is a nutrient header (Total Fat, Niacin, Amino Acids,
         ...), drop it and re-process starting from segment[1]. Loops to
         handle multi-nutrient headers.
      2. Invert FDC's "category, qualifier" ordering: if segment[1] is a
         short alphabetic variety word, prepend it.
      3. Skip terminators (raw, cooked, frozen, ...) as qualifiers.
      4. Skip ambiguous tokens like "b12" that aren't foods.

    Examples:
      "Beans, pinto, mature seeds, raw"          → "pinto beans"
      "Rice, brown, long-grain, raw"             → "brown rice"
      "Carrots, raw whole"                       → "carrots"
      "Total Fat, Ground turkey, 93% lean, raw"  → "ground turkey"
      "Niacin, Chicken breast, raw"              → "chicken breast"
      "Amino Acids, Chicken, dark meat, ..."     → "dark meat chicken"
    """
    if not fdc_description:
        return None
    segments = [s.strip().lower() for s in fdc_description.split(",")]

    # Rule 1: shift past nutrient headers (handles exact + suffixed forms
    # like "cholesterol-wt" or "cholesterol - beef").
    while segments and _is_nutrient_header(segments[0]):
        segments = segments[1:]

    if not segments or not segments[0]:
        return None
    head = segments[0]
    if head in _SKIP_EXACT:
        return None

    # Rule 2: try prepending a variety qualifier from segment[1].
    if len(segments) >= 2:
        q = segments[1]
        words = q.split()
        if (1 <= len(words) <= 2
                and all(w.isalpha() and 3 <= len(w) <= 15 for w in words)
                and q not in _SEGMENT_TERMINATORS
                and words[0] not in _SEGMENT_TERMINATORS):
            return f"{q} {head}"

    return head


def load_terms_from_fdc_descriptions(
    path: str,
    normalize_plurals: bool = True,
) -> list[str]:
    """Extract unique Kroger-friendly search terms from a JSON file whose
    keys are FDC descriptions (e.g. fresh_foods_nutrients_names_physiology.json).

    When `normalize_plurals=True`, collapses "apple" / "apples" by preferring
    the singular form (simpler for Kroger's search to match).
    """
    with open(path) as f:
        keys = list(json.load(f).keys())
    terms: set[str] = set()
    for k in keys:
        t = extract_search_term(k)
        if t is not None:
            terms.add(t)
    if normalize_plurals:
        # If both "apple" and "apples" present, keep singular.
        drop = {t for t in terms if t.endswith("s") and t[:-1] in terms}
        terms -= drop
    return sorted(terms)


def get_access_token(cfg: RetailerConfig) -> str:
    """OAuth2 client-credentials flow with the `product.compact` scope."""
    import base64

    import requests

    creds = base64.b64encode(f"{cfg.client_id}:{cfg.client_secret}".encode()).decode()
    resp = requests.post(
        OAUTH_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {creds}",
        },
        data={"grant_type": "client_credentials", "scope": "product.compact"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_products(
    token: str,
    term: str,
    location_id: str,
    limit: int = 50,
    max_retries: int = 3,
    backoff_base_sec: float = 2.0,
) -> list[dict]:
    """Search products, retrying transient 5xx / timeout errors.

    Returns [] on persistent failure after `max_retries` attempts, with
    a warning to stderr — the caller can continue with the next term
    rather than losing a whole scrape to one bad response.
    """
    import requests
    import time

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                PRODUCT_SEARCH_URL,
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "filter.term": term,
                    "filter.locationId": location_id,
                    "filter.limit": str(limit),
                },
                timeout=30,
            )
            # Retry on 5xx; raise on 4xx (won't fix with retry)
            if 500 <= resp.status_code < 600:
                raise requests.HTTPError(f"{resp.status_code} from Kroger", response=resp)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = backoff_base_sec * (2 ** attempt)
                print(
                    f"  [retry {attempt + 1}/{max_retries}] {term!r}: {e}; "
                    f"sleeping {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue
    print(f"  [skip] {term!r}: {last_exc}", file=sys.stderr)
    return []


def normalize_product(raw: dict, location_id: str) -> dict | None:
    """Extract price + package size from Kroger's nested response."""
    items = raw.get("items") or []
    if not items:
        return None
    item = items[0]
    price_info = item.get("price") or {}
    price = price_info.get("promo") or price_info.get("regular")
    if price is None:
        return None

    size_raw = item.get("size", "")
    package_g = parse_size_to_grams(size_raw)
    if package_g is None:
        return None

    return {
        "upc": raw.get("upc"),
        "description": raw.get("description"),
        "price": float(price),
        "package_size_g": package_g,
        "size_raw": size_raw,
        "location_id": location_id,
    }


UNIT_TO_GRAMS = {
    "oz": 28.3495,
    "lb": 453.592,
    "g": 1.0,
    "kg": 1000.0,
    "gram": 1.0,
    "grams": 1.0,
    "pound": 453.592,
    "pounds": 453.592,
    "ounce": 28.3495,
    "ounces": 28.3495,
}


def parse_size_to_grams(size: str) -> float | None:
    """Parse strings like '16 oz', '2 lb', '500 g' into grams."""
    parts = size.lower().strip().split()
    if len(parts) != 2:
        return None
    try:
        value = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].rstrip(".").strip()
    factor = UNIT_TO_GRAMS.get(unit)
    return value * factor if factor else None


def is_stale(fetched_at: str, threshold_days: int = STALENESS_THRESHOLD_DAYS) -> bool:
    dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    return age_days > threshold_days


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Kroger prices")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="fetch prices for a list of search terms")
    group = fetch.add_mutually_exclusive_group(required=True)
    group.add_argument("--terms", help="comma-separated search terms")
    group.add_argument(
        "--terms-from-food-info",
        help="path to food_info.json; uses its keys as search terms (covers the "
             "69 foods with ERS prices; small set)",
    )
    group.add_argument(
        "--terms-from-fdc-descriptions",
        help="path to a JSON whose keys are FDC descriptions "
             "(e.g. fresh_foods_nutrients_names_physiology.json — 2,254 foods). "
             "Extracts the first-comma segment, dedupes, and normalizes plurals "
             "to ~300-400 retailer-friendly terms.",
    )
    fetch.add_argument(
        "--location-id",
        help="Kroger store ID. Precedence: this flag > --zip auto-resolve > "
             "KROGER_LOCATION_ID env > error.",
    )
    fetch.add_argument(
        "--zip",
        help="US zip code. Auto-resolves to the nearest Kroger store's "
             "location ID (saved in output metadata). Overridden by "
             "--location-id if both are given.",
    )
    fetch.add_argument("--output", default="prices_raw.json")
    fetch.add_argument("--rate-limit-sec", type=float, default=1.0)
    fetch.add_argument(
        "--cache-dir", default=None,
        help="directory to cache per-term results. Default: "
             "cache/kroger-<location_id>/. A cached term is skipped on "
             "re-run — so an interrupted scrape resumes by just re-running "
             "the same command.",
    )
    fetch.add_argument(
        "--no-cache",
        action="store_true",
        help="disable per-term caching (forces every term to hit the API)",
    )

    find = sub.add_parser("find-location", help="look up store IDs near a zip code")
    find.add_argument("--zip", required=True, help="5-digit US zip code")
    find.add_argument("--radius", type=int, default=10, help="miles")
    find.add_argument(
        "--top",
        action="store_true",
        help="print only the nearest store's locationId (one line, no other output). "
             "Useful for scripting: --location-id \"$(... find-location --zip X --top)\"",
    )

    args = parser.parse_args()

    if args.cmd == "find-location":
        cfg = load_config(location_id="_ignored_for_location_lookup_")
        token = get_access_token(cfg)
        locations = find_locations(token, args.zip, args.radius)
        if not locations:
            print(f"no Kroger stores within {args.radius} miles of {args.zip}", file=sys.stderr)
            return 2
        if args.top:
            top = locations[0]
            addr = top.get("address", {})
            print(top.get("locationId"))
            print(
                f"# picked: {top.get('name')} at {addr.get('addressLine1')}, "
                f"{addr.get('city')} {addr.get('state')} {addr.get('zipCode')}",
                file=sys.stderr,
            )
            return 0
        for loc in locations:
            addr = loc.get("address", {})
            print(
                f"  {loc.get('locationId')}  {loc.get('name')}  "
                f"{addr.get('addressLine1')}, {addr.get('city')} {addr.get('state')} {addr.get('zipCode')}"
            )
        return 0

    # --- Resolve --zip to a location ID if needed ---
    location_id = args.location_id
    zip_code = getattr(args, "zip", None)
    if not location_id and zip_code:
        # Need a temporary token to look up the store; use a partial config.
        _tmp_cfg = load_config(location_id="_zip_lookup_")
        _tmp_token = get_access_token(_tmp_cfg)
        locations = find_locations(_tmp_token, zip_code, radius_miles=15)
        if not locations:
            print(f"no Kroger stores within 15 miles of {zip_code}", file=sys.stderr)
            return 2
        location_id = locations[0].get("locationId")
        addr = locations[0].get("address", {})
        print(
            f"resolved zip {zip_code} → store {location_id} "
            f"({locations[0].get('name')}, {addr.get('city')} {addr.get('state')})",
            file=sys.stderr,
        )

    cfg = load_config(location_id=location_id)
    token = get_access_token(cfg)

    if args.terms_from_food_info:
        terms = load_terms_from_food_info(args.terms_from_food_info)
        print(f"loaded {len(terms)} terms from {args.terms_from_food_info}", file=sys.stderr)
    elif args.terms_from_fdc_descriptions:
        terms = load_terms_from_fdc_descriptions(args.terms_from_fdc_descriptions)
        print(
            f"extracted {len(terms)} unique search terms from "
            f"{args.terms_from_fdc_descriptions}",
            file=sys.stderr,
        )
    else:
        terms = [t.strip() for t in args.terms.split(",") if t.strip()]

    # Per-term cache: each search term writes its own JSON file. Re-runs
    # skip terms that already have a cache hit. At the end, we merge all
    # cache files into the single --output file.
    if args.no_cache:
        cache_dir = None
    else:
        cache_dir = Path(args.cache_dir) if args.cache_dir else Path(
            f"cache/kroger-{cfg.location_id}"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, term in enumerate(terms, start=1):
            cache_file = cache_dir / f"{_slugify(term)}.json" if cache_dir else None
            if cache_file and cache_file.exists():
                # Cached — skip fetch
                continue
            print(f"[{i}/{len(terms)}] searching: {term}", file=sys.stderr)
            hits = search_products(token, term, cfg.location_id)
            term_products = []
            for raw in hits:
                normalized = normalize_product(raw, cfg.location_id)
                if normalized:
                    normalized["search_term"] = term
                    term_products.append(normalized)
            if cache_file:
                cache_file.write_text(json.dumps({
                    "search_term": term,
                    "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "location_id": cfg.location_id,
                    "products": term_products,
                }, indent=2))
            time.sleep(args.rate_limit_sec)
    except KeyboardInterrupt:
        print("\n[interrupted] per-term caches preserved; re-run to resume", file=sys.stderr)
        return 130

    # Merge all cache files (or in-memory results if --no-cache) into --output.
    products: list[dict] = []
    if cache_dir:
        for term in terms:
            cache_file = cache_dir / f"{_slugify(term)}.json"
            if cache_file.exists():
                products.extend(json.loads(cache_file.read_text()).get("products", []))

    out_path = Path(args.output)
    out: dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "retailer": "kroger",
        "location_id": cfg.location_id,
        "zip_code": zip_code,
        "terms": terms,
        "products": products,
    }
    out_path.write_text(json.dumps(out, indent=2))
    cached_terms = sum(
        1 for t in terms
        if cache_dir and (cache_dir / f"{_slugify(t)}.json").exists()
    ) if cache_dir else 0
    print(
        f"merged {cached_terms}/{len(terms)} cached terms → {len(products)} "
        f"products → {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
