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


def load_config() -> RetailerConfig:
    try:
        return RetailerConfig(
            client_id=os.environ["KROGER_CLIENT_ID"],
            client_secret=os.environ["KROGER_CLIENT_SECRET"],
            location_id=os.environ.get("KROGER_LOCATION_ID", "01400943"),
        )
    except KeyError as e:
        raise SystemExit(
            f"Missing env var: {e}. Register at https://developer.kroger.com/ "
            "and export KROGER_CLIENT_ID / KROGER_CLIENT_SECRET."
        )


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
    parser.add_argument("--terms", required=True, help="comma-separated search terms")
    parser.add_argument("--output", default="prices_raw.json")
    parser.add_argument("--rate-limit-sec", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config()
    token = get_access_token(cfg)
    products: list[dict] = []
    for term in [t.strip() for t in args.terms.split(",") if t.strip()]:
        print(f"searching: {term}", file=sys.stderr)
        for raw in search_products(token, term, cfg.location_id):
            normalized = normalize_product(raw, cfg.location_id)
            if normalized:
                products.append(normalized)
        time.sleep(args.rate_limit_sec)

    out: dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "retailer": "kroger",
        "location_id": cfg.location_id,
        "products": products,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(products)} products to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
