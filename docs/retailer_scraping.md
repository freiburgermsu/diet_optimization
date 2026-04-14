# Retailer price scraping

## Current implementation

`scripts/fetch_prices.py` uses **Kroger's Developer API** (api.kroger.com) — an official OAuth2-gated public API, not web scraping. This is the lowest-risk category: a documented developer program, free tier, and an explicit product-search endpoint. Terms of service allow automated access within the rate limits.

## Terms of service review (Kroger)

Per https://developer.kroger.com/terms (confirm before shipping):

- ✅ Automated access allowed for registered apps
- ✅ Product search is in the free tier
- ⚠️ Rate limits: 10,000 calls/day per client — well above our needs (~300 foods, monthly refresh)
- ⚠️ Pricing is **location-specific**; document the `location_id` used
- ⚠️ No redistribution of pricing data without attribution; our use case (internal LP input) is fine, but a public dataset dump would not be

## Setup

1. Register at https://developer.kroger.com/
2. Create an app with the `product.compact` scope
3. Export credentials:

   ```sh
   export KROGER_CLIENT_ID=...
   export KROGER_CLIENT_SECRET=...
   export KROGER_LOCATION_ID=01400943   # default: a Midwest store
   ```

4. Run:

   ```sh
   python scripts/fetch_prices.py \
       --terms "carrots,broccoli,rice,pinto beans,chicken breast" \
       --output prices_raw.json
   python scripts/normalize_prices.py \
       --raw prices_raw.json \
       --fdc-lookup ../food.csv \
       --output prices.json
   ```

## Alternative retailers evaluated

| Retailer | Status | Why not |
|----------|--------|---------|
| Walmart | Aggressive anti-bot WAF | Scraping likely violates ToS, high ban risk |
| Trader Joe's | No API, bot detection | Not sustainable |
| Target | Partner API gated | Requires B2B paperwork |
| Instacart | Widely scraped but ambiguous ToS | Last-resort option |
| Open Food Facts | CC0 licensed, crowd-sourced | Good supplement (#2 alternative), sparse density per food category |

## Supplementing with Open Food Facts

OFF can fill gaps. It's CC0 licensed (unambiguously legal), but density is uneven — some categories are well-covered, others are sparse. Treat as a secondary source behind Kroger, not a primary.
