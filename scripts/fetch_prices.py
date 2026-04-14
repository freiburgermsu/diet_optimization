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


def search_products(token: str, term: str, location_id: str, limit: int = 50) -> list[dict]:
    import requests

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
    resp.raise_for_status()
    return resp.json().get("data", [])


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
        help="Kroger store ID. Precedence: this flag > KROGER_LOCATION_ID env > error. "
             "Use `find-location` subcommand to discover by zip.",
    )
    fetch.add_argument("--output", default="prices_raw.json")
    fetch.add_argument("--rate-limit-sec", type=float, default=1.0)

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

    cfg = load_config(location_id=args.location_id)
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

    products: list[dict] = []
    for term in terms:
        print(f"searching: {term}", file=sys.stderr)
        for raw in search_products(token, term, cfg.location_id):
            normalized = normalize_product(raw, cfg.location_id)
            if normalized:
                normalized["search_term"] = term  # provenance for downstream join
                products.append(normalized)
        time.sleep(args.rate_limit_sec)

    out: dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "retailer": "kroger",
        "location_id": cfg.location_id,
        "terms": terms,
        "products": products,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(products)} products to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
