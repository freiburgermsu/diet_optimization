#!/usr/bin/env python3
"""Rank Kroger products per search term using the Claude API.

Our rule-based filter (normalize_prices.py) has clear gaps — it picks
jelly for "grapes", dog food for "ground chicken", denture adhesive
for "seal". A fast classifier task is a natural fit for Claude.

For each search term + candidate product list, ask Claude to pick the
best match for "raw/unprocessed <term> suitable for nutritional
modeling", or return null if no product fits. Uses:

- Prompt caching on the system prompt (big enough to clear the 4K cache
  minimum for Opus 4.6)
- Pydantic-validated structured outputs
- Per-term cache file at cache/claude/<slug>.json — re-running skips
  already-decided terms
- Live progress counter

Output: prices_claude.json (same shape as normalize_prices.py output).

Usage:
    export ANTHROPIC_API_KEY=...
    uv run --with "anthropic>=0.40" --with pydantic \\
      python scripts/claude_rank_products.py \\
      --raw prices_raw.json --output prices_claude.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import anthropic
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}. Install with:\n"
        f"  pip install 'anthropic>=0.40' pydantic"
    )


DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_CACHE_DIR = Path("cache/claude")
SYSTEM_PROMPT = """You are a food product classifier for a nutritional diet optimization research project.

## Task

For each Kroger search term and its candidate products, pick the ONE product that best represents a raw, unprocessed, single-ingredient version of the search term — suitable for pairing with USDA FoodData Central nutrition data.

Return `null` for the chosen product if NO candidate is a genuine raw-ingredient match.

## Rules for picking a winner

1. **Prefer raw/plain over prepared.** Fresh whole carrots beat carrot snack trays with ranch dip. Plain brown rice beats cilantro lime rice. Plain almonds beat honey-roasted almonds.

2. **Exclude these categories entirely** (return null if they're all that's available):
   - **Pet food**: Anything for dogs, cats, or other pets.
   - **Condiments/spreads made FROM the ingredient**: Jelly, jam, preserves, paste, syrup, nectar, butter (except plain dairy butter when searching for butter), sauce, dressing, dip.
   - **Snack/dessert forms**: Bars, biscuits, cookies, candies, chips, crackers, trail mix, snack packs, gummies.
   - **Beverages made from the ingredient**: Juice (unless the search term specifically names a juice), smoothie, tea.
   - **Oils and extracts**: "Avocado Oil" is not avocado. "Vanilla extract" is not vanilla.
   - **Prepared/ready-to-eat meals**: Stir fries, soups, stews, rice dishes, pasta dishes.
   - **Flavored/seasoned versions**: Honey-glazed carrots, wasabi almonds.
   - **Canned in syrup/sauce**: Peaches in heavy syrup, fruit cocktail.
   - **Flakes/powders/dried herbs** when a fresh version is available (if the search is for a dried spice like "cumin", dried is fine).
   - **Brand-name collisions**: "Caribou Coffee" is not the caribou animal. "Poligrip" denture adhesive contains "seal" in the name but isn't seal meat.
   - **Completely different foods**: A search for "plum" that only returns "plum tomatoes" is no match — plum tomatoes are tomatoes, not plums. Return null.

3. **Acceptable forms:**
   - Plain canned versions of beans (Kroger Pinto Beans, Kroger Black Beans) are OK — canning preserves raw-bean nutrition reasonably well.
   - Frozen plain vegetables (frozen peas, frozen broccoli) are OK.
   - Dry grains and legumes (dry lentils, dry beans) are OK.
   - Fresh produce, raw meat cuts, whole nuts/seeds are ideal.
   - Minimally processed forms (bagged spinach, pre-cut pineapple) are OK.

4. **When multiple acceptable products exist**, pick the cheapest per unit (total_price / package_size_g × 100). If prices are similar, pick the simplest description (shortest, with fewest adjectives).

5. **Brand context matters.** "Simple Truth", "Kroger", and "Private Selection" are Kroger's store brands and typically indicate plain versions. Third-party brand products are fine when the description is clean.

## Output format

Return a JSON object matching this schema:
```json
{
  "chosen_index": 0,
  "reason": "Plain raw carrots, cheapest per-100g of the candidates.",
  "confidence": "high"
}
```

- `chosen_index`: integer index (0-based) into the candidate products list, OR `null` if none qualify.
- `reason`: one-sentence explanation. If `chosen_index` is null, explain what all candidates were (e.g., "All 14 candidates are prepared bean products or bean-based snacks; no raw-ingredient match available").
- `confidence`: "high" (obvious match), "medium" (acceptable but not ideal), or "low" (best available but questionable).

## Worked examples

**Term: "carrots"**
Candidates:
[0] "Kroger® Whole Carrots" — 2267g, $3.59
[1] "Green Giant Restaurant Style Honey Glazed Carrots" — 340g, $4.99
[2] "Baby Cut 'n Peel Carrots" — 454g, $1.99
Answer: `{"chosen_index": 0, "reason": "Plain whole carrots, largest package, cheapest per-100g at \\$0.16", "confidence": "high"}`

**Term: "grapes"**
Candidates:
[0] "Smart Way™ Grape Jelly" — 510g, $1.99
[1] "Welch's 100% Grape Juice" — 1.89L, $4.49
Answer: `{"chosen_index": null, "reason": "Both candidates are grape-derived products (jelly, juice), not raw grapes", "confidence": "high"}`

**Term: "broccoli"**
Candidates:
[0] "Cauliflower" — 454g, $1.99
[1] "Broccoli Crown" — 454g, $1.99
[2] "Birds Eye Broccoli Florets (frozen)" — 340g, $2.49
Answer: `{"chosen_index": 1, "reason": "Plain fresh broccoli crown; cauliflower is a different food", "confidence": "high"}`

**Term: "caribou"**
Candidates:
[0] "Caribou Coffee Daybreak Light Roast Ground" — 340g, $12.99
Answer: `{"chosen_index": null, "reason": "Only match is a coffee brand name collision, not caribou meat", "confidence": "high"}`

**Term: "ground chicken"**
Candidates:
[0] "Pet Pride® Adult Wet Dog Food Chopped Chicken Ground" — 375g, $1.29
[1] "Kroger® Fresh Ground Chicken" — 454g, $4.99
Answer: `{"chosen_index": 1, "reason": "Pet food excluded; Kroger fresh ground chicken is the correct raw ingredient", "confidence": "high"}`

**Term: "avocado"**
Candidates:
[0] "Chosen Foods 100% Pure Avocado Oil Spray" — 142g, $7.99
[1] "Kroger® Fresh Mini Hass Avocados Bag - 2 Pound" — 907g, $5.99
[2] "Wholly Guacamole® 100% Natural Dip" — 227g, $4.49
Answer: `{"chosen_index": 1, "reason": "Fresh Hass avocados in bag; oil spray and guacamole dip are both processed derivatives", "confidence": "high"}`

**Term: "apricot"**
Candidates:
[0] "Smucker's Apricot Preserves" — 510g, $3.49
[1] "Kroger® Dried Apricots" — 170g, $4.99
[2] "Hunt's® 100% Natural Apricot Halves in Juice" — 411g, $2.49
Answer: `{"chosen_index": 1, "reason": "Dried apricots are minimally processed single-ingredient; preserves and canned-in-juice are excluded", "confidence": "medium"}`
Note: If fresh apricots were available they would win; dried is acceptable when no fresh option exists.

**Term: "agave"**
Candidates:
[0] "Simple Truth Organic® 100% Blue Agave Nectar Syrup" — 340g, $4.99
[1] "Tequila Patrón Silver" — 750ml, $49.99
Answer: `{"chosen_index": null, "reason": "Only candidates are agave nectar (syrup, excluded) and tequila (beverage, excluded); no raw agave plant form available", "confidence": "high"}`

**Term: "soybeans"**
Candidates:
[0] "Kroger® Wild Caught Sardines in Soybean Oil" — 106g, $1.99
[1] "Simple Truth Organic® Dry Edamame Soybeans" — 454g, $3.99
[2] "La Choy® Soy Sauce" — 295ml, $2.99
Answer: `{"chosen_index": 1, "reason": "Dry edamame soybeans are the raw ingredient; sardines and soy sauce are both wrong foods that happen to contain soy", "confidence": "high"}`

**Term: "plum"**
Candidates:
[0] "Hunt's® Whole Peeled Plum Tomatoes" — 411g, $1.49
[1] "Sunsweet® Pitted Dried Plums (Prunes)" — 255g, $4.49
Answer: `{"chosen_index": 1, "reason": "Dried plums (prunes) are single-ingredient plum products; plum tomatoes are a tomato variety, not plums", "confidence": "medium"}`

**Term: "beef"**
Candidates:
[0] "Kroger® 80/20 Ground Beef" — 454g, $4.99
[1] "Beef Jerky Original Flavor" — 85g, $6.99
[2] "Campbell's® Beef Broth" — 414g, $2.49
[3] "Private Selection® Beef Tenderloin Steak" — 340g, $22.99
Answer: `{"chosen_index": 0, "reason": "Ground beef is a plain raw meat product; tenderloin is also valid but much more expensive per-100g. Jerky (processed) and broth (extract) are excluded", "confidence": "high"}`

**Term: "arrowroot"**
Candidates:
[0] "Gerber Snacks for Baby Biscuits Arrowroot Bag" — 43g, $3.49
[1] "Bob's Red Mill® Arrowroot Starch/Flour" — 454g, $5.99
Answer: `{"chosen_index": 1, "reason": "Pure arrowroot starch is the raw ingredient form; baby biscuits are a prepared snack", "confidence": "high"}`

**Term: "almonds"**
Candidates:
[0] "Blue Diamond® Honey Roasted Almonds" — 170g, $4.99
[1] "Blue Diamond® Wasabi & Soy Sauce Almonds" — 170g, $4.99
[2] "KIND Trail Mix" — 142g, $3.99
[3] "Simple Truth® Raw Almonds" — 454g, $12.99
Answer: `{"chosen_index": 3, "reason": "Simple Truth raw almonds are unflavored; others are seasoned or mixed with other ingredients", "confidence": "high"}`

**Term: "flax seeds"**
Candidates:
[0] "Bob's Red Mill® Whole Golden Flax Seed" — 340g, $4.99
[1] "Spectrum® Premium Ground Flaxseed" — 397g, $6.99
Answer: `{"chosen_index": 0, "reason": "Whole flax seeds preferred over pre-ground (ground oxidizes faster and is a more processed form); either would be acceptable", "confidence": "high"}`

**Term: "abalone"**
Candidates: (empty list)
Answer: `{"chosen_index": null, "reason": "No candidates returned from retailer", "confidence": "high"}`

## Edge cases and judgment notes

- **Canned vs fresh.** When Kroger only stocks canned versions of a common ingredient (e.g. tomatoes, beans, pumpkin), canned is an acceptable v1 match. Nutrition data for canned vs raw differs slightly but not enough to reject when no fresh option exists. Note this in the reason ("canned; no fresh alternative").

- **Multi-ingredient staples.** Bread, pasta, sausage, cheese, and other inherently multi-ingredient foods: pick the plainest, least-flavored version available. A "Kroger® Plain Whole Wheat Bread" beats "Dave's Killer Bread Organic 21 Whole Grains and Seeds".

- **Organic vs conventional.** Prefer whichever is cheaper per-100g — the nutrition profile is effectively identical for LP purposes. Don't pick organic just because it sounds healthier.

- **Pre-cut / pre-washed.** Acceptable — bagged spinach, pre-cut pineapple chunks, baby carrots. These are minimally processed and the nutrition matches raw.

- **Dried herbs vs fresh herbs.** If the term specifically says "dried" (e.g. "dried oregano"), dried is correct. For plain herb names ("parsley", "cilantro"), prefer fresh over flakes, but flakes are acceptable if fresh is absent — note the trade-off in the reason.

- **Single-ingredient frozen.** Plain frozen peas, broccoli, strawberries are equivalent to fresh for LP purposes. Frozen prepared dinners (stir-fry kits, pot pies) are excluded.

- **Low confidence scenarios.** If all candidates are marginal (e.g., only dried/canned versions when fresh is expected, or only non-ideal brands), pick the best available and mark `confidence: "low"`. Downstream code will filter low-confidence choices if needed.

- **Error on the side of null over a bad match.** A null entry tells the pipeline to fall back to another price source (TFP, manual). A bad match silently corrupts the diet LP with wrong nutrition-to-price pairing.

## Additional worked examples

**Term: "chicken breast"**
Candidates:
[0] "Tyson® Thin Sliced Boneless Skinless Chicken Breasts" — 794g, $9.99
[1] "PERDUE® Fresh Boneless Skinless Chicken Breasts Value Pack" — 1361g, $12.99
[2] "Kroger® Breaded Chicken Breast Patties" — 680g, $6.99
[3] "Chicken Breast Lunchmeat Deli Sliced" — 227g, $4.99
Answer: `{"chosen_index": 1, "reason": "Plain fresh chicken breasts, cheapest per-100g at \\$0.95; breaded patties and lunchmeat are processed", "confidence": "high"}`

**Term: "oats"**
Candidates:
[0] "Kroger® 100% Whole Grain 1 Minute Quick Oats" — 510g, $1.49
[1] "Quaker® Instant Oatmeal Original" — 340g, $3.49
[2] "Kind® Oats & Honey Granola Bar 12-pack" — 336g, $10.99
[3] "Cheerios® Original Breakfast Cereal" — 510g, $5.49
Answer: `{"chosen_index": 0, "reason": "Plain whole-grain quick oats are the raw ingredient; instant oatmeal has added sugar/salt, granola bars are prepared snacks, Cheerios is cereal", "confidence": "high"}`

**Term: "yogurt"**
Candidates:
[0] "Chobani® Greek Strawberry Yogurt" — 150g, $1.29
[1] "Fage® Total 0% Plain Greek Yogurt" — 454g, $5.49
[2] "Yoplait® Original Vanilla" — 170g, $0.99
[3] "Stonyfield® Organic Whole Milk Plain Yogurt" — 907g, $6.99
Answer: `{"chosen_index": 3, "reason": "Plain organic whole milk yogurt is unsweetened/unflavored; plain Greek yogurt (Fage) is also acceptable; flavored variants excluded", "confidence": "high"}`

**Term: "salmon"**
Candidates:
[0] "Kroger® Wild Caught Sockeye Salmon Portions" — 340g, $14.99
[1] "Acme Smoked Fish Nova Smoked Salmon" — 113g, $8.99
[2] "StarKist® Salmon Creations Lemon Pepper Pouch" — 74g, $2.49
[3] "Bumble Bee® Pink Salmon in Water" — 418g, $4.49
Answer: `{"chosen_index": 0, "reason": "Raw wild salmon portions; smoked salmon and flavored pouches are processed; canned in water is acceptable but fresh wild-caught is better", "confidence": "high"}`

**Term: "honey"**
Candidates:
[0] "Kroger® Pure Clover Honey" — 340g, $4.99
[1] "Nature Nate's® 100% Pure Raw & Unfiltered Honey" — 907g, $11.99
[2] "Honey Nut Cheerios®" — 595g, $6.49
[3] "Local Hive® Raw & Unfiltered Honey" — 454g, $8.99
Answer: `{"chosen_index": 1, "reason": "Raw unfiltered honey at cheapest per-100g (\\$1.32); clover honey is also fine; Honey Nut Cheerios is cereal with honey flavor, not honey", "confidence": "high"}`

**Term: "tofu"**
Candidates:
[0] "House Foods® Tofu Shirataki Noodles" — 227g, $2.49
[1] "Simple Truth Organic® Soft Silken Tofu" — 397g, $2.29
[2] "Nasoya® Plantspired™ Teriyaki Tofu" — 227g, $4.49
[3] "Simple Truth Organic® Extra Firm Tofu" — 397g, $2.29
Answer: `{"chosen_index": 3, "reason": "Plain organic extra firm tofu; silken is also valid; teriyaki tofu is pre-marinated (excluded); tofu noodles are a different product", "confidence": "high"}`

**Term: "peanut butter"**
Candidates:
[0] "Simple Truth Organic® Creamy Peanut Butter (only peanuts)" — 454g, $3.99
[1] "Jif® Creamy Peanut Butter" — 454g, $3.49
[2] "Reese's® Peanut Butter Cups" — 42g, $1.29
[3] "Peanut Butter Cap'n Crunch® Cereal" — 335g, $4.49
Answer: `{"chosen_index": 0, "reason": "Single-ingredient organic peanut butter (just peanuts) is the best raw-ingredient match; Jif has added sugar/oils but is acceptable; candy and cereal excluded", "confidence": "high"}`

**Term: "eggs"**
Candidates:
[0] "Kroger® Grade A Large White Eggs - 12 Count" — 680g, $2.99
[1] "Kroger® Cage-Free Liquid Egg Whites" — 473g, $3.99
[2] "Just Egg® Plant-Based Scramble" — 340g, $5.99
[3] "Eggland's Best® Large Brown Eggs - 12 Count" — 680g, $4.49
Answer: `{"chosen_index": 0, "reason": "Whole large eggs at cheapest per-100g; liquid egg whites are a derivative; Just Egg is a plant-based substitute, not eggs", "confidence": "high"}`

**Term: "cheese"**
Candidates:
[0] "Kraft® Singles American Pasteurized Cheese Product" — 340g, $4.99
[1] "Kroger® Shredded Sharp Cheddar Cheese" — 227g, $3.99
[2] "Cabot® Seriously Sharp Cheddar Block" — 227g, $5.49
[3] "Velveeta® Original Cheese Sauce" — 227g, $4.49
Answer: `{"chosen_index": 2, "reason": "Real cheddar cheese block; shredded is acceptable too; Kraft Singles is a \\"pasteurized cheese product\\" (not real cheese); Velveeta is cheese sauce", "confidence": "high"}`

**Term: "orange"**
Candidates:
[0] "Kroger® Fresh Navel Oranges Bag" — 1814g, $3.99
[1] "Simply Orange® Pulp Free Orange Juice" — 2.36L, $4.99
[2] "Kroger® Mandarin Oranges in Juice" — 312g, $1.49
[3] "Tropicana® Orange Juice Calcium + Vitamin D" — 1.89L, $4.49
Answer: `{"chosen_index": 0, "reason": "Fresh whole oranges are the raw ingredient; juice is a beverage derivative; canned mandarins are acceptable but fresh is preferred", "confidence": "high"}`

**Term: "butter"**
Candidates:
[0] "Kroger® Salted Butter Quarters" — 454g, $4.49
[1] "Land O Lakes® Unsalted Butter Sticks" — 454g, $5.49
[2] "Peanut Butter Cap'n Crunch® Cereal" — 335g, $4.49
[3] "I Can't Believe It's Not Butter® Spread" — 425g, $4.99
Answer: `{"chosen_index": 0, "reason": "Plain salted butter (dairy) at cheapest per-100g; unsalted is equally valid; cereal and margarine spreads are not butter", "confidence": "high"}`

**Term: "milk"**
Candidates:
[0] "Kroger® 2% Reduced Fat Milk Gallon" — 3780g, $3.49
[1] "Silk® Unsweetened Almond Milk" — 1.89L, $4.49
[2] "Kroger® Whole Milk Gallon" — 3780g, $3.49
[3] "Nestlé® Condensed Sweetened Milk" — 397g, $2.49
Answer: `{"chosen_index": 2, "reason": "Plain whole cow's milk; 2% is also fine; almond milk is a different food (plant-based); condensed milk is processed", "confidence": "high"}`

**Term: "bread"**
Candidates:
[0] "Wonder® Classic White Bread" — 567g, $3.49
[1] "Dave's Killer Bread® 21 Whole Grains & Seeds" — 765g, $5.49
[2] "Kroger® 100% Whole Wheat Bread" — 680g, $2.99
[3] "Pepperidge Farm® Cinnamon Raisin Swirl" — 454g, $4.99
Answer: `{"chosen_index": 2, "reason": "Plain whole wheat bread; 21-grain is acceptable but more adjuncts; white bread is valid if plain-white requested; cinnamon raisin is a sweetened variant", "confidence": "medium"}`

**Term: "sugar"**
Candidates:
[0] "Kroger® Pure Cane Sugar" — 1814g, $3.49
[1] "Splenda® No Calorie Sweetener" — 227g, $9.99
[2] "Domino® Light Brown Sugar" — 907g, $3.99
[3] "Karo® Light Corn Syrup" — 454g, $3.49
Answer: `{"chosen_index": 0, "reason": "Plain pure cane sugar is the standard raw ingredient; brown sugar is acceptable but less \\"plain\\"; Splenda is a sweetener substitute; corn syrup is different", "confidence": "high"}`

**Term: "pork"**
Candidates:
[0] "Kroger® Ground Pork" — 454g, $3.99
[1] "Kroger® Fresh Natural Pork Shoulder Butt Bone In" — 1814g, $7.99
[2] "Hormel® Black Label Bacon" — 454g, $7.99
[3] "Jimmy Dean® Premium Pork Sausage Roll" — 454g, $3.99
Answer: `{"chosen_index": 1, "reason": "Fresh pork shoulder is a plain raw cut at cheapest per-100g (\\$0.44); ground pork is also plain; bacon is cured/smoked; sausage has seasoning/fillers", "confidence": "high"}`

**Term: "potato"** (note: both singular and plural — FDC may produce either)
Candidates:
[0] "Kroger® Russet Potatoes" — 2268g, $2.99
[1] "Lay's® Classic Potato Chips" — 226g, $4.29
[2] "Ore-Ida® Frozen Tater Tots" — 907g, $4.99
[3] "Idahoan® Instant Mashed Potatoes" — 113g, $1.49
Answer: `{"chosen_index": 0, "reason": "Plain fresh russet potatoes at cheapest per-100g (\\$0.13); chips, tots, and instant mash are all processed potato products", "confidence": "high"}`

**Term: "lamb"**
Candidates:
[0] "Lamb Loin Chops" — 340g, $10.49
[1] "Kroger® Ground Lamb" — 454g, $7.99
[2] "Catelli Veal Leg Cutlets" — 454g, $18.99
Answer: `{"chosen_index": 1, "reason": "Ground lamb is a plain raw cut at cheapest per-100g; loin chops are also valid but more expensive; veal is a different animal (cow)", "confidence": "high"}`

**Term: "garlic"**
Candidates:
[0] "Fresh Whole Garlic Bulb" — 85g, $0.79
[1] "Spice World® Minced Garlic 32 oz" — 907g, $6.99
[2] "McCormick® Garlic Powder" — 85g, $4.49
[3] "Kroger® Garlic Butter Sauce" — 280g, $2.49
Answer: `{"chosen_index": 0, "reason": "Fresh whole garlic bulb is the raw ingredient; minced garlic in jar is acceptable but has preservatives; garlic powder is dried/ground; garlic butter sauce is a prepared sauce", "confidence": "high"}`

**Term: "mushrooms"**
Candidates:
[0] "Kroger® Whole White Mushrooms" — 227g, $1.99
[1] "Campbell's® Cream of Mushroom Soup" — 298g, $1.49
[2] "Kroger® Sliced Baby Bella Mushrooms" — 227g, $2.49
[3] "B&B® Whole Mushrooms Pieces & Stems Canned" — 113g, $1.49
Answer: `{"chosen_index": 0, "reason": "Fresh whole mushrooms at cheapest per-100g; sliced baby bellas also fine; canned is acceptable if fresh absent; soup is a prepared product", "confidence": "high"}`

**Term: "walnuts"**
Candidates:
[0] "Simple Truth® Pieces Walnuts" — 283g, $6.49
[1] "Diamond® Walnut Halves & Pieces" — 454g, $9.99
[2] "Planters® NUT-rition Heart Healthy Mix" — 244g, $5.99
[3] "Walnut Creek Foods® Amish Country Pancake Syrup" — 510g, $4.99
Answer: `{"chosen_index": 1, "reason": "Plain walnut halves at cheapest per-100g (\\$2.20); Simple Truth pieces also valid; trail mix is multi-ingredient; Walnut Creek is a brand name, not walnuts", "confidence": "high"}`

**Term: "seaweed"**
Candidates:
[0] "Annie Chun's® Roasted Seaweed Snacks" — 14g, $1.99
[1] "Eden® Sushi Nori Sheets" — 17g, $4.99
[2] "Sea Tangle® Kelp Noodles" — 340g, $4.49
Answer: `{"chosen_index": 1, "reason": "Plain dried nori seaweed sheets are the closest raw-ingredient form; seaweed snacks are seasoned; kelp noodles are a processed pasta substitute", "confidence": "medium"}`

**Term: "cucumbers"**
Candidates:
[0] "Private Selection® Seedless Mini Cucumbers" — 454g, $3.99
[1] "Vlasic® Kosher Dill Pickles" — 907g, $4.49
[2] "Bubbies® Pure Kosher Dill Pickles" — 907g, $6.99
[3] "English Seedless Cucumber" — 340g, $1.99
Answer: `{"chosen_index": 3, "reason": "English seedless cucumber at cheapest per-100g; mini cucumbers also plain; pickles are cucumbers preserved in brine and excluded", "confidence": "high"}`

**Term: "celery"**
Candidates:
[0] "Private Selection® Long Celery Sticks" — 454g, $2.99
[1] "Celery Seed Spice 1.5 oz" — 43g, $4.49
[2] "Kroger® Celery Salt" — 170g, $1.99
[3] "Celery Bunch — sold each" — 680g, $2.49
Answer: `{"chosen_index": 3, "reason": "Whole celery bunch at cheapest per-100g; pre-cut sticks also valid; celery seed and celery salt are spices, not the vegetable", "confidence": "high"}`

**Term: "strawberries"**
Candidates:
[0] "Fresh Strawberries - 1lb Clamshell" — 454g, $3.99
[1] "Kroger® Sliced Strawberries in Light Syrup Frozen" — 454g, $3.49
[2] "Smucker's® Strawberry Preserves" — 510g, $3.49
[3] "Kroger® Plain Frozen Strawberries" — 340g, $3.49
Answer: `{"chosen_index": 0, "reason": "Fresh strawberries are ideal; plain frozen is also acceptable (just fruit); strawberries in light syrup are sweetened; preserves are jam", "confidence": "high"}`

**Term: "green peas"**
Candidates:
[0] "Birds Eye® Frozen Sweet Peas" — 340g, $2.49
[1] "Kroger® Sweet Peas Canned" — 425g, $1.29
[2] "Goya® Green Split Peas Dry" — 454g, $1.99
[3] "Lay's® Wasabi Peas Snack" — 142g, $3.49
Answer: `{"chosen_index": 0, "reason": "Plain frozen sweet peas; canned sweet peas also fine; dry split peas are a different legume form; wasabi peas are flavored snack", "confidence": "high"}`

**Term: "coconut"**
Candidates:
[0] "Caribbean Dreams Coconut Milk" — 400g, $2.49
[1] "Bob's Red Mill® Unsweetened Shredded Coconut" — 340g, $4.99
[2] "La Costena® Coconut Cream" — 425g, $2.99
[3] "Fresh Whole Coconut - Each" — 680g, $5.99
Answer: `{"chosen_index": 3, "reason": "Fresh whole coconut is the raw ingredient; unsweetened shredded is acceptable (just dried coconut); coconut milk and cream are derivatives", "confidence": "medium"}`

**Term: "spinach"**
Candidates:
[0] "Kroger® Fresh Baby Spinach Clamshell" — 142g, $3.49
[1] "Popeye® Canned Spinach" — 397g, $1.79
[2] "Kroger® Chopped Frozen Spinach" — 454g, $2.49
[3] "Kroger® Leaf Spinach — Fresh Bunch" — 227g, $1.99
Answer: `{"chosen_index": 3, "reason": "Fresh leaf spinach bunch at cheapest per-100g (\\$0.88); baby spinach clamshell also fine; canned and frozen are acceptable fallbacks", "confidence": "high"}`
"""


class ProductChoice(BaseModel):
    chosen_index: int | None = Field(
        default=None,
        description="0-based index into the candidate products list, or null if no candidate qualifies",
    )
    reason: str = Field(description="One-sentence explanation for the choice")
    confidence: str = Field(description="high / medium / low")


def slugify(term: str) -> str:
    s = term.lower().strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_-]+", "", s)
    return s or "_unnamed"


def group_by_term(raw: dict) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for p in raw.get("products", []):
        term = p.get("search_term", "")
        groups.setdefault(term, []).append(p)
    return groups


def build_user_prompt(term: str, products: list[dict]) -> str:
    lines = [f'Term: "{term}"', "", f"Candidates ({len(products)}):"]
    for i, p in enumerate(products):
        desc = p.get("description", "?")
        price = p.get("price", 0)
        size_g = p.get("package_size_g", 0)
        per_100g = (price / size_g * 100) if size_g > 0 else 0
        lines.append(
            f"[{i}] {desc!r} — {size_g:.0f}g, ${price:.2f} ({per_100g:.3f}/100g)"
        )
    return "\n".join(lines)


def rank_term(
    client: anthropic.Anthropic, model: str, term: str, products: list[dict]
) -> dict:
    """Call Claude to pick the best match for one term. Returns a dict with:
    {chosen_product: dict|None, choice: {...}, usage: {...}}
    """
    user_msg = build_user_prompt(term, products)

    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
        output_format=ProductChoice,
    )

    choice: ProductChoice = response.parsed_output
    chosen = (
        products[choice.chosen_index]
        if choice.chosen_index is not None and 0 <= choice.chosen_index < len(products)
        else None
    )

    return {
        "term": term,
        "choice": choice.model_dump(),
        "chosen_product": chosen,
        "usage": {
            "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
            "cache_read_input_tokens": response.usage.cache_read_input_tokens,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def to_price_entry(result: dict, retailer: str, fetched_at: str) -> dict | None:
    """Convert a ranker result to the same shape normalize_prices.py emits."""
    p = result.get("chosen_product")
    if p is None:
        return None
    size_g = p.get("package_size_g", 0)
    if size_g <= 0:
        return None
    return {
        "price_per_100g": p["price"] / size_g * 100,
        "price_source": retailer,
        "fetched_at": fetched_at,
        "raw_description": p.get("description"),
        "package_size_g": size_g,
        "claude_reason": result["choice"]["reason"],
        "claude_confidence": result["choice"]["confidence"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="prices_raw.json")
    parser.add_argument("--output", default="prices_claude.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N terms (for sanity-checking)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(Path(args.raw).read_text())
    fetched_at = raw.get("fetched_at", "")
    retailer = raw.get("retailer", "kroger")
    groups = group_by_term(raw)
    terms = list(groups.keys())
    if args.limit:
        terms = terms[: args.limit]

    # Rough cost estimate assuming Opus 4.6 pricing with cache hits after #1
    approx_input_tokens = sum(len(build_user_prompt(t, groups[t])) for t in terms) // 4
    system_tokens = len(SYSTEM_PROMPT) // 4
    est_input_cost = (
        system_tokens * 1.25 * 5 / 1_000_000  # first write
        + (len(terms) - 1) * system_tokens * 0.1 * 5 / 1_000_000  # subsequent reads
        + approx_input_tokens * 5 / 1_000_000  # per-term input
    )
    est_output_cost = len(terms) * 100 * 25 / 1_000_000
    print(
        f"[estimate] {len(terms)} terms, model {args.model}, "
        f"~${est_input_cost + est_output_cost:.2f} (Opus pricing)",
        file=sys.stderr,
    )

    client = anthropic.Anthropic()
    results: dict[str, dict] = {}

    for i, term in enumerate(terms, start=1):
        cache_file = cache_dir / f"{slugify(term)}.json"
        if cache_file.exists():
            results[term] = json.loads(cache_file.read_text())
            continue
        products = groups[term]
        if not products:
            continue

        print(f"[{i}/{len(terms)}] ranking: {term}  ({len(products)} candidates)",
              file=sys.stderr)
        try:
            result = rank_term(client, args.model, term, products)
        except anthropic.APIError as e:
            print(f"  [skip] {term}: {e}", file=sys.stderr)
            continue

        cache_file.write_text(json.dumps(result, indent=2))
        results[term] = result
        cr = result["usage"]["cache_read_input_tokens"]
        print(
            f"  → {result['choice'].get('confidence')} "
            f"(cache_read={cr})",
            file=sys.stderr,
        )

    # Merge to price-entry shape
    prices: dict[str, dict] = {}
    dropped = []
    for term, result in results.items():
        entry = to_price_entry(result, retailer, fetched_at)
        if entry is not None:
            prices[term] = entry
        else:
            dropped.append((term, result["choice"].get("reason", "")))

    Path(args.output).write_text(json.dumps(prices, indent=2))
    print(
        f"\nwrote {len(prices)}/{len(terms)} priced terms to {args.output}",
        file=sys.stderr,
    )
    if dropped:
        print(f"{len(dropped)} terms returned null (Claude found no match):",
              file=sys.stderr)
        for term, reason in dropped[:10]:
            print(f"  {term}: {reason[:80]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
