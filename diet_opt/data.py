"""Load and normalize USDA + DRI inputs.

Pulled from `optimization_diet.ipynb` cells 13–19. The heavy FDC CSV
preprocessing (cells 2–11) still lives in the notebook — extracting it
requires the >1.7 GB raw CSVs which are gitignored (see #6 for version
pinning, #4 for unified schema).
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import inf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent


def load_json(name: str) -> dict:
    with open(DATA_DIR / name) as f:
        return json.load(f)


def load_pipeline_inputs() -> tuple[dict, dict, dict]:
    """Load the three JSON inputs the LP consumes.

    Returns: (food_info, food_matches, nutrition)
    """
    food_info = load_json("food_info.json")
    food_matches = load_json("food_matches.json")
    nutrition = load_json("nutrition.json")
    return food_info, food_matches, nutrition


DEFAULT_CUP_EQ = 1.0   # cups per 100g-edible; used when priced_foods.json
                       # has no per-food value (most of the 610-food table).

ESSENTIAL_AMINO_ACIDS = [
    "Histidine", "Isoleucine", "Leucine", "Lysine", "Methionine",
    "Phenylalanine", "Threonine", "Tryptophan", "Valine", "Tyrosine",
]
_KNN_K = 5  # number of nearest neighbors for imputation


def impute_amino_acids(
    food_matches: dict,
    categories: dict,
) -> tuple[dict, dict]:
    """Fill in missing amino-acid values using KNN on the nutrient fingerprint.

    USDA only annotates amino acids for ~half the 610-food catalog. Without
    imputation the LP treats the other ~227 foods (chicken breast, ribeye,
    peanuts, ...) as having 0 of every EAA, which implicitly penalises them
    in any LP with AA constraints.

    Approach — K-nearest-neighbor imputation (K=5):

    1. Build a feature vector for each food from all non-AA nutrients that
       are broadly reported (Energy, Protein, Fat, Carb, Fiber, Iron, ...).
       Normalize each dimension by its standard deviation across the catalog.

    2. Partition foods into "donors" (all 9 EAAs present) and "recipients"
       (at least one EAA missing, Protein > 0).

    3. For each recipient, compute Euclidean distance to every donor using
       only nutrient dimensions that *both* foods report (partial matching).
       Select the K=5 nearest donors.

    4. For each missing AA: imputed value = food.Protein × weighted mean of
       (neighbor.AA / neighbor.Protein) across the K neighbors, weighted by
       1/distance. The protein-ratio anchor keeps the scale proportional to
       the food's own protein content, while the neighbor weighting captures
       the food's nutrient fingerprint (legumes match legumes, meats match
       meats, etc.).

    Foods without Protein get no imputation (AA = 0).

    Returns: (enriched food_matches, stats) where stats is a per-AA count
    of imputed entries. The input dict is not mutated.
    """
    from math import sqrt

    # Identify non-AA nutrient dimensions for the feature vector.
    aa_set = set(ESSENTIAL_AMINO_ACIDS)
    all_nutrients: set[str] = set()
    for nut in food_matches.values():
        all_nutrients.update(nut.keys())
    feature_nutrients = sorted(all_nutrients - aa_set)

    # Per-dimension mean and std for normalization (computed over all foods).
    dim_vals: dict[str, list[float]] = {fn: [] for fn in feature_nutrients}
    for nut in food_matches.values():
        for fn in feature_nutrients:
            if fn in nut:
                dim_vals[fn].append(nut[fn])

    dim_mean: dict[str, float] = {}
    dim_std: dict[str, float] = {}
    for fn, vals in dim_vals.items():
        if not vals:
            dim_mean[fn] = 0.0
            dim_std[fn] = 1.0
            continue
        m = sum(vals) / len(vals)
        dim_mean[fn] = m
        var = sum((v - m) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 1.0
        dim_std[fn] = sqrt(var) if var > 0 else 1.0

    def _normalize(food_nut: dict) -> dict[str, float]:
        return {
            fn: (food_nut[fn] - dim_mean[fn]) / dim_std[fn]
            for fn in feature_nutrients
            if fn in food_nut
        }

    # Partition into donors (have all 9 EAAs + Protein > 0) and recipients.
    donors: list[tuple[str, dict, dict[str, float]]] = []   # (name, raw_nut, norm_vec)
    recipients: list[tuple[str, dict]] = []                  # (name, raw_nut copy)

    for food, nut in food_matches.items():
        protein = nut.get("Protein", 0) or 0
        has_all = protein > 0 and all(nut.get(aa, 0) > 0 for aa in ESSENTIAL_AMINO_ACIDS)
        if has_all:
            donors.append((food, nut, _normalize(nut)))
        elif protein > 0 and any(aa not in nut or nut.get(aa, 0) <= 0 for aa in ESSENTIAL_AMINO_ACIDS):
            recipients.append((food, nut))

    def _distance(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        """Euclidean distance over shared dimensions only."""
        shared = set(vec_a) & set(vec_b)
        if not shared:
            return float("inf")
        return sqrt(sum((vec_a[fn] - vec_b[fn]) ** 2 for fn in shared) / len(shared))

    # Impute
    enriched = {food: dict(nut) for food, nut in food_matches.items()}
    stats = {aa: 0 for aa in ESSENTIAL_AMINO_ACIDS}

    for food, _nut in recipients:
        rec_vec = _normalize(_nut)
        protein = _nut.get("Protein", 0) or 0
        if protein <= 0:
            continue

        # Find K nearest donors
        dists = [(d_name, d_nut, _distance(rec_vec, d_vec))
                 for d_name, d_nut, d_vec in donors]
        dists.sort(key=lambda t: t[2])
        top_k = dists[:_KNN_K]
        if not top_k:
            continue

        # Inverse-distance weights (epsilon guard for zero distance)
        eps = 1e-8
        weights = [1.0 / (d + eps) for _, _, d in top_k]
        w_sum = sum(weights)

        for aa in ESSENTIAL_AMINO_ACIDS:
            if enriched[food].get(aa, 0) > 0:
                continue
            # Weighted mean of neighbor (AA / Protein) ratios
            ratio_sum = 0.0
            for (_, d_nut, _), w in zip(top_k, weights):
                d_prot = d_nut.get("Protein", 0) or 0
                if d_prot > 0:
                    ratio_sum += w * (d_nut[aa] / d_prot)
            ratio = ratio_sum / w_sum if w_sum > 0 else 0.0
            if ratio > 0:
                enriched[food][aa] = protein * ratio
                stats[aa] += 1

    return enriched, stats


def load_priced_foods(
    name: str = "priced_foods.json",
    default_cup_eq: float = DEFAULT_CUP_EQ,
    impute_aa: bool = True,
) -> tuple[dict, dict, dict]:
    """Load priced_foods.json and split it into the (food_info, food_matches,
    nutrition) triple the existing model.build_model() expects.

    Shape of priced_foods.json (per entry):
      {price_per_100g, price_source, tfp_category, nutrients: {...}, ...}

    When `impute_aa` is True (default), foods missing amino-acid values have
    them filled in via `impute_amino_acids` (protein × per-category median
    ratio). Pass False to keep original raw values for auditing.

    Returns:
      food_info    → {term: {price, yield, cupEQ}} where price is back-solved
                     so that `price/yield/4.54` equals the original
                     price_per_100g (preserves the existing objective formula).
      food_matches → {term: nutrients-dict} (amino acids imputed if enabled)
      nutrition    → loaded from nutrition.json (unchanged)
    """
    priced = load_json(name)
    food_info: dict[str, dict] = {}
    food_matches: dict[str, dict] = {}
    categories: dict[str, str] = {}
    for term, entry in priced.items():
        if term == "_metadata":
            continue
        ppg = entry["price_per_100g"]
        info: dict = {
            "price": ppg * 4.54,
            "yield": 1.0,
            "cupEQ": entry.get("cup_equivalent", default_cup_eq),
        }
        pkg = entry.get("package_size_g")
        if pkg:
            info["package_size_g"] = float(pkg)
        cat = entry.get("tfp_category")
        if cat:
            categories[term] = cat
            info["tfp_category"] = cat
        info["perishable"] = is_perishable(term, cat)
        food_info[term] = info
        food_matches[term] = entry.get("nutrients", {})
    nutrition = load_json("nutrition.json")
    if impute_aa:
        food_matches, _stats = impute_amino_acids(food_matches, categories)
    return food_info, food_matches, nutrition


# Keywords / categories for perishability classification.
# Perishable = spoils within ~1 week at fridge temp. Buy whole packages.
# Durable = weeks to months (roots, hard squash, dried, canned, frozen, nuts, grains).
_PERISHABLE_CATEGORIES = {
    "Dark green vegetables",
    "Fruit, higher nutrient density",
    "Fruit, lower nutrient density",
}

_PERISHABLE_KEYWORDS = {
    # Leafy greens
    "spinach", "kale", "lettuce", "arugula", "chard", "collard",
    "dandelion", "mustard greens", "watercress", "endive", "radicchio",
    "cabbage", "bok choy", "mesclun",
    # Soft fruit
    "berry", "berries", "strawberr", "blueberr", "raspberr", "blackberr",
    "grape", "mango", "papaya", "banana", "peach", "plum", "nectarine",
    "cherry", "melon", "watermelon", "cantaloupe", "honeydew", "kiwi",
    "pear", "fig", "apricot", "avocado", "pineapple", "lychee",
    "passion fruit", "guava", "persimmon", "starfruit",
    # Soft vegetables
    "tomato", "cucumber", "zucchini", "squash summer", "bell pepper",
    "hot pepper", "jalapeno", "serrano", "celery", "asparagus",
    "green bean", "snap pea", "snow pea", "okra", "eggplant",
    "mushroom", "corn",  # fresh corn, not dried
    "yogurt",  # yogurt spoils faster than hard dairy
    # Herbs
    "basil", "cilantro", "parsley", "dill", "mint", "chive",
    # Fresh proteins (if not frozen)
    "fresh fish", "fresh chicken",
}

_DURABLE_KEYWORDS = {
    # Roots and hard vegetables that last weeks+
    "carrot", "potato", "sweet potato", "yam", "beet", "turnip",
    "rutabaga", "parsnip", "radish", "onion", "garlic", "ginger",
    "butternut", "acorn squash", "pumpkin", "winter squash",
    # All shelf-stable
    "rice", "bean", "lentil", "chickpea", "pea dried", "grain",
    "oat", "wheat", "flour", "pasta", "bread", "cereal",
    "nut", "seed", "peanut", "almond", "walnut", "cashew", "pistachio",
    "canned", "frozen", "dried", "jerky", "oil", "vinegar",
    "sugar", "honey", "molasses", "syrup", "salt",
    "egg",  # eggs last 3-5 weeks refrigerated
    "apple",  # apples last weeks refrigerated
    # Dairy (except yogurt) lasts weeks refrigerated
    "milk", "cheese", "butter", "cream", "parmesan", "ricotta",
    "cottage", "mozzarella", "cheddar", "sour cream",
}


def is_perishable(food_name: str, tfp_category: str | None = None) -> bool:
    """Classify whether a food spoils within ~1 week (buy whole packages)."""
    name_lower = food_name.lower()

    # Durable keywords override — check first
    if any(kw in name_lower for kw in _DURABLE_KEYWORDS):
        return False

    # Perishable keywords
    if any(kw in name_lower for kw in _PERISHABLE_KEYWORDS):
        return True

    # Category-based fallback
    if tfp_category and tfp_category in _PERISHABLE_CATEGORIES:
        return True

    return False


def parse_bound(raw: str | float | int) -> float:
    """Parse a DRI bound like '100', '1,200', or 'inf' into a float."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace(",", "")
    if s.lower() in {"inf", "nd", ""}:
        return inf
    return float(s.split()[0])


def validate_bounds(nutrition: dict) -> list[str]:
    """Return a list of nutrients whose lower bound exceeds upper.

    Empty list = valid. See #7 for override handling and citations.
    """
    violations = []
    for nutrient, content in nutrition.items():
        lb = parse_bound(content.get("low_bound", 0))
        ub = parse_bound(content.get("high_bound", inf))
        if lb > ub:
            violations.append(f"{nutrient}: lb={lb} > ub={ub}")
    return violations


def average_dict_values(dicts: list[dict]) -> dict:
    """Mean-across-sources for nested nutrient dicts.

    Extracted from notebook cell 16. Used to collapse multiple FDC rows
    for the same canonical food into one nutrient vector.
    """
    sum_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for d in dicts:
        for _food, nutrients in d.items():
            for nutrient, value in nutrients.items():
                sum_counts[nutrient][0] += value
                sum_counts[nutrient][1] += 1
    return {k: s / n for k, (s, n) in sum_counts.items()}
