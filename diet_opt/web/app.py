"""FastAPI UI over the diet LP + Claude meal-plan pipeline.

Exposes every CLI flag as a form input:
  - LP solver: --min-serving-grams, --dietary-preset, --blacklist, --whitelist
  - Meal plan: --max-prep-min, --cuisine-style, --days, --model

Endpoints:
  GET  /                  the HTML form
  GET  /presets           list available dietary presets (populates multi-select)
  GET  /foods             list foods in priced_foods.json (populates autocomplete)
  POST /optimize          solve the LP → {cost, foods}
  POST /meal-plan         run Claude on a previously-solved diet → {plan, markdown}
  GET  /health            liveness

Rate-limited via SlowAPI when available. Anthropic API key is read from
the server's environment (ANTHROPIC_API_KEY) — not accepted in requests.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..data import load_priced_foods
from ..dri import ACTIVITY_PAL, UserProfile, apply_profile
from ..presets import foods_excluded_by_presets, list_presets

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PRICED_FOODS = REPO_ROOT / "priced_foods.json"


# Nutrient classification — macros on top, micros on bottom, following the
# conventional nutrition-label ordering. Tuple: (category_rank, category_label).
NUTRIENT_CATEGORIES: dict[str, tuple[int, str]] = {
    "Energy":                 (0, "Energy"),
    "Protein":                (1, "Macronutrients"),
    "Carbohydrate":           (1, "Macronutrients"),
    "Total Fiber":            (1, "Macronutrients"),
    "Fat":                    (1, "Macronutrients"),
    "Saturated fatty acids":  (1, "Macronutrients"),
    "Linoleic Acid":          (1, "Macronutrients"),
    "Linolenic Acid":         (1, "Macronutrients"),
    "Dietary Cholesterol":    (1, "Macronutrients"),
    "Total Water":            (2, "Water"),
    "Vitamin A":              (3, "Fat-soluble vitamins"),
    "Vitamin D":              (3, "Fat-soluble vitamins"),
    "Vitamin E":              (3, "Fat-soluble vitamins"),
    "Vitamin K":              (3, "Fat-soluble vitamins"),
    "Carotenoids":            (3, "Fat-soluble vitamins"),
    "Vitamin C":              (4, "Water-soluble vitamins"),
    "Thiamin":                (4, "Water-soluble vitamins"),
    "Riboflavin":             (4, "Water-soluble vitamins"),
    "Niacin":                 (4, "Water-soluble vitamins"),
    "Pantothenic Acid":       (4, "Water-soluble vitamins"),
    "Vitamin B6":             (4, "Water-soluble vitamins"),
    "Biotin":                 (4, "Water-soluble vitamins"),
    "Folate":                 (4, "Water-soluble vitamins"),
    "Vitamin B12":            (4, "Water-soluble vitamins"),
    "Choline":                (4, "Water-soluble vitamins"),
    "Calcium":                (5, "Major minerals"),
    "Phosphorus":             (5, "Major minerals"),
    "Magnesium":              (5, "Major minerals"),
    "Sodium":                 (5, "Major minerals"),
    "Potassium":              (5, "Major minerals"),
    "Chloride":               (5, "Major minerals"),
    "Iron":                   (6, "Trace minerals"),
    "Zinc":                   (6, "Trace minerals"),
    "Copper":                 (6, "Trace minerals"),
    "Manganese":              (6, "Trace minerals"),
    "Iodine":                 (6, "Trace minerals"),
    "Selenium":               (6, "Trace minerals"),
    "Molybdenum":             (6, "Trace minerals"),
    "Chromium":               (6, "Trace minerals"),
    "Fluoride":               (6, "Trace minerals"),
    "Histidine":              (7, "Essential amino acids"),
    "Isoleucine":             (7, "Essential amino acids"),
    "Leucine":                (7, "Essential amino acids"),
    "Lysine":                 (7, "Essential amino acids"),
    "Methionine":             (7, "Essential amino acids"),
    "Phenylalanine":          (7, "Essential amino acids"),
    "Threonine":              (7, "Essential amino acids"),
    "Tryptophan":             (7, "Essential amino acids"),
    "Valine":                 (7, "Essential amino acids"),
    "Tyrosine":               (7, "Essential amino acids"),
    "Total Polyphenols":      (8, "Bioactives"),
    "Effective Omega-3":      (8, "Bioactives"),
}
_NUTRIENT_ORDER = {name: i for i, name in enumerate(NUTRIENT_CATEGORIES)}


def _category_for(nutrient: str) -> tuple[int, str]:
    return NUTRIENT_CATEGORIES.get(nutrient, (99, "Other"))


# Peer-reviewed citations supporting the nutritional guidance.
# Each citation is tagged with contexts (profile/disease/flag) that trigger it.
CITATIONS: list[dict] = [
    # Baseline DRI
    {"contexts": ["baseline"], "cite": "Institute of Medicine. Dietary Reference Intakes for Energy, Carbohydrate, Fiber, Fat, Fatty Acids, Cholesterol, Protein, and Amino Acids. National Academies Press, 2005.", "doi": "10.17226/10490"},
    {"contexts": ["baseline"], "cite": "Institute of Medicine. Dietary Reference Intakes for Calcium and Vitamin D. National Academies Press, 2011.", "doi": "10.17226/13050"},
    {"contexts": ["baseline"], "cite": "Institute of Medicine. Dietary Reference Intakes for Vitamin A, Vitamin K, Arsenic, Boron, Chromium, Copper, Iodine, Iron, Manganese, Molybdenum, Nickel, Silicon, Vanadium, and Zinc. National Academies Press, 2001.", "doi": "10.17226/10026"},
    # Energy / BMR
    {"contexts": ["profile"], "cite": "Mifflin MD, St Jeor ST, et al. A new predictive equation for resting energy expenditure in healthy individuals. Am J Clin Nutr. 1990;51(2):241-247.", "doi": "10.1093/ajcn/51.2.241"},
    # Protein / activity
    {"contexts": ["weight_lifting", "active", "very_active"], "cite": "Jager R, Kerksick CM, et al. International Society of Sports Nutrition position stand: protein and exercise. J Int Soc Sports Nutr. 2017;14:20.", "doi": "10.1186/s12970-017-0177-8"},
    {"contexts": ["weight_lifting", "active", "very_active"], "cite": "Phillips SM, Van Loon LJ. Dietary protein for athletes: from requirements to optimum adaptation. J Sports Sci. 2011;29(sup1):S29-S38.", "doi": "10.1080/02640414.2011.619204"},
    # Amino acid requirements
    {"contexts": ["baseline"], "cite": "WHO/FAO/UNU. Protein and Amino Acid Requirements in Human Nutrition. WHO Technical Report Series 935, 2007."},
    # Omega-3 bioconversion
    {"contexts": ["anti_inflammatory"], "cite": "Burdge GC, Calder PC. Conversion of alpha-linolenic acid to longer-chain polyunsaturated fatty acids in human adults. Reprod Nutr Dev. 2005;45(5):581-597.", "doi": "10.1051/rnd:2005047"},
    {"contexts": ["anti_inflammatory"], "cite": "Simopoulos AP. The importance of the ratio of omega-6/omega-3 essential fatty acids. Biomed Pharmacother. 2002;56(8):365-379.", "doi": "10.1016/S0753-3322(02)00253-6"},
    # Polyphenols
    {"contexts": ["baseline"], "cite": "Rothwell JA, Perez-Jimenez J, et al. Phenol-Explorer 3.0: a major update of the Phenol-Explorer database to incorporate data on the effects of food processing on polyphenol content. Database. 2013;2013:bat070.", "doi": "10.1093/database/bat070"},
    # Cardiovascular
    {"contexts": ["cardiovascular"], "cite": "Sacks FM, Lichtenstein AH, et al. Dietary Fats and Cardiovascular Disease: A Presidential Advisory From the American Heart Association. Circulation. 2017;136(3):e1-e23.", "doi": "10.1161/CIR.0000000000000510"},
    # Diabetes
    {"contexts": ["diabetes"], "cite": "Evert AB, Dennison M, et al. Nutrition Therapy for Adults With Diabetes or Prediabetes: A Consensus Report. Diabetes Care. 2019;42(5):731-754.", "doi": "10.2337/dci19-0014"},
    # Hypertension
    {"contexts": ["hypertension", "preeclampsia_risk"], "cite": "Whelton PK, Carey RM, et al. 2017 ACC/AHA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults. J Am Coll Cardiol. 2018;71(19):e127-e248.", "doi": "10.1016/j.jacc.2017.11.006"},
    # Kidney
    {"contexts": ["kidney_disease"], "cite": "Ikizler TA, Burrowes JD, et al. KDOQI Clinical Practice Guideline for Nutrition in CKD: 2020 Update. Am J Kidney Dis. 2020;76(3 Suppl 1):S1-S107.", "doi": "10.1053/j.ajkd.2020.05.006"},
    # Osteoporosis
    {"contexts": ["osteoporosis"], "cite": "Weaver CM, Gordon CM, et al. The National Osteoporosis Foundation's position statement on peak bone mass development and lifestyle factors: a systematic review and implementation recommendations. Osteoporos Int. 2016;27(4):1281-1386.", "doi": "10.1007/s00198-015-3440-3"},
    # Iron deficiency
    {"contexts": ["iron_deficiency_anemia"], "cite": "Camaschella C. Iron-deficiency anemia. N Engl J Med. 2015;372(19):1832-1843.", "doi": "10.1056/NEJMra1401038"},
    # NAFLD
    {"contexts": ["nafld"], "cite": "Chalasani N, Younossi Z, et al. The diagnosis and management of nonalcoholic fatty liver disease: Practice guidance from the AASLD. Hepatology. 2018;67(1):328-357.", "doi": "10.1002/hep.29367"},
    # Hypothyroidism
    {"contexts": ["hypothyroidism"], "cite": "Zimmermann MB, Boelaert K. Iodine deficiency and thyroid disorders. Lancet Diabetes Endocrinol. 2015;3(4):286-295.", "doi": "10.1016/S2213-8587(14)70225-6"},
    # IBD
    {"contexts": ["ibd_crohns"], "cite": "Forbes A, Escher J, et al. ESPEN guideline: Clinical nutrition in inflammatory bowel disease. Clin Nutr. 2017;36(2):321-347.", "doi": "10.1016/j.clnu.2016.12.027"},
    # Celiac
    {"contexts": ["celiac"], "cite": "Rubio-Tapia A, Hill ID, et al. ACG clinical guidelines: diagnosis and management of celiac disease. Am J Gastroenterol. 2013;108(5):656-676.", "doi": "10.1038/ajg.2013.79"},
    # Pregnancy
    {"contexts": ["pregnancy", "preeclampsia_risk"], "cite": "Koletzko B, Godfrey KM, et al. Nutrition During Pregnancy, Lactation and Early Childhood and its Implications for Maternal and Long-Term Child Health: The EarlyNutrition Project Recommendations. Ann Nutr Metab. 2019;74(2):93-106.", "doi": "10.1159/000496471"},
    # Sarcopenia
    {"contexts": ["sarcopenia"], "cite": "Bauer J, Biolo G, et al. Evidence-based recommendations for optimal dietary protein intake in older people: a position paper from the PROT-AGE Study Group. J Am Med Dir Assoc. 2013;14(8):542-559.", "doi": "10.1016/j.jamda.2013.05.021"},
    # Migraine
    {"contexts": ["migraine"], "cite": "Mauskop A, Varughese J. Why all migraine patients should be treated with magnesium. J Neural Transm. 2012;119(5):575-579.", "doi": "10.1007/s00702-012-0790-2"},
    {"contexts": ["migraine"], "cite": "Schoenen J, Jacquy J, Lenaerts M. Effectiveness of high-dose riboflavin in migraine prophylaxis. A randomized controlled trial. Neurology. 1998;50(2):466-470.", "doi": "10.1212/WNL.50.2.466"},
    # Gout
    {"contexts": ["gout"], "cite": "FitzGerald JD, Dalbeth N, et al. 2020 American College of Rheumatology Guideline for Management of Gout. Arthritis Care Res. 2020;72(6):744-760.", "doi": "10.1002/acr.24180"},
    # CPI / pricing
    {"contexts": ["baseline"], "cite": "U.S. Bureau of Labor Statistics. Consumer Price Index for All Urban Consumers: Food at Home (CUUR0000SAF11). https://data.bls.gov/"},
    # Food composition
    {"contexts": ["baseline"], "cite": "U.S. Department of Agriculture, Agricultural Research Service. FoodData Central. https://fdc.nal.usda.gov/"},
]


def _relevant_citations(req: OptimizeRequest) -> list[dict]:
    """Return citations relevant to the user's profile and disease selections."""
    active_contexts = {"baseline"}
    if req.activity:
        active_contexts.add(req.activity)
    if req.age and req.age >= 60:
        active_contexts.add("sarcopenia")
    if any(f is not None for f in [req.age, req.sex, req.weight_kg]):
        active_contexts.add("profile")
    if req.anti_inflammatory:
        active_contexts.add("anti_inflammatory")
    for dp in req.disease_presets:
        active_contexts.add(dp)
    if req.low_fat:
        active_contexts.add("cardiovascular")
    if req.low_carb:
        active_contexts.add("diabetes")

    seen = set()
    out = []
    for c in CITATIONS:
        if any(ctx in active_contexts for ctx in c["contexts"]):
            key = c["cite"][:80]
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out


# --- Request / response schemas ---

class OptimizeRequest(BaseModel):
    min_serving_grams: float = Field(0.0, ge=0, le=200)
    dietary_presets: list[str] = Field(default_factory=list)
    blacklist: list[str] = Field(default_factory=list)
    whitelist: list[str] = Field(default_factory=list)
    # Optional demographics — when all present, DRI scales to this profile.
    age: int | None = Field(None, ge=1, le=120)
    sex: str | None = Field(None, pattern="^(male|female|nonbinary)$")
    weight_kg: float | None = Field(None, gt=20, lt=300)
    height_cm: float | None = Field(None, gt=100, lt=230)
    activity: str | None = Field(None, pattern="^(sedentary|light|moderate|weight_lifting|active|very_active)$")
    b12_supplemented: bool = Field(True)
    vitd_supplemented: bool = Field(True)
    iodine_supplemented: bool = Field(True)
    low_fat: bool = Field(False)
    low_carb: bool = Field(False)
    anti_inflammatory: bool = Field(False)
    # Disease presets (each activates a combination of dietary constraints)
    disease_presets: list[str] = Field(default_factory=list)
    # Manual nutrient overrides: [{nutrient, bound, value}]
    nutrient_overrides: list[dict] = Field(default_factory=list)


class FoodEntry(BaseModel):
    food: str
    grams: int
    price_per_100g: float
    price_source: str


class NutrientStatus(BaseModel):
    nutrient: str
    value: float
    low_bound: float | None
    high_bound: float | None
    units: str
    pct_of_lower: float | None   # value / low_bound * 100 (None if lb=0/inf)
    binding: str | None          # "lower" / "upper" / None — tells UI which bound pins it
    category: str                # classification label, e.g. "Macronutrients"


class ShadowPriceOut(BaseModel):
    nutrient: str
    bound: str             # "lower" / "upper"
    bound_value: float
    savings_per_unit: float  # $ saved per 1 unit of bound relaxation
    explanation: str


class OptimizeResponse(BaseModel):
    cost_per_day: float
    foods: list[FoodEntry]
    excluded_by_preset: int
    warnings: list[str] = Field(default_factory=list)
    nutrients: list[NutrientStatus] = Field(default_factory=list)
    shadow_prices: list[ShadowPriceOut] = Field(default_factory=list)


class MealPlanRequest(BaseModel):
    diet: dict[str, int]          # {food_name: grams/day}
    days: int = Field(7, ge=1, le=14)
    max_prep_min: int = Field(30, ge=5, le=120)
    cuisine_style: str | None = None
    blacklist: list[str] = Field(default_factory=list)
    whitelist: list[str] = Field(default_factory=list)
    model: str = "claude-haiku-4-5"


class MealPlanResponse(BaseModel):
    plan: dict                  # WeeklyMealPlan as dict
    markdown: str
    violations: list[str]


# Disease preset definitions: each maps to a set of flags + bound overrides.
DISEASE_PRESETS: dict[str, dict] = {
    "cardiovascular": {
        "flags": {"low_fat", "anti_inflammatory"},
        "overrides": [
            {"nutrient": "Sodium", "bound": "upper", "value": 1500},
            {"nutrient": "Dietary Cholesterol", "bound": "upper", "value": 300},
        ],
    },
    "diabetes": {
        "flags": {"low_carb"},
        "overrides": [
            {"nutrient": "Total Fiber", "bound": "lower", "value": 30},
        ],
    },
    "hypertension": {
        "flags": set(),
        "overrides": [
            {"nutrient": "Sodium", "bound": "upper", "value": 1500},
            {"nutrient": "Potassium", "bound": "lower", "value": 4700},
        ],
    },
    "kidney_disease": {
        "flags": set(),
        "overrides": [
            {"nutrient": "Protein", "bound": "upper", "value": 60},
            {"nutrient": "Sodium", "bound": "upper", "value": 2000},
            {"nutrient": "Potassium", "bound": "upper", "value": 2500},
            {"nutrient": "Phosphorus", "bound": "upper", "value": 1000},
        ],
    },
    "gout": {
        "flags": set(),
        "overrides": [
            {"nutrient": "Protein", "bound": "upper", "value": 80},
        ],
    },
    "osteoporosis": {
        "flags": set(),
        "overrides": [
            {"nutrient": "Calcium", "bound": "lower", "value": 1200},
            {"nutrient": "Vitamin K", "bound": "lower", "value": 200},
            {"nutrient": "Phosphorus", "bound": "lower", "value": 700},
        ],
    },
    "iron_deficiency_anemia": {
        "flags": set(),
        "overrides": [
            {"nutrient": "Iron", "bound": "lower", "value": 18},
            {"nutrient": "Vitamin C", "bound": "lower", "value": 200},
            {"nutrient": "Folate", "bound": "lower", "value": 600},
        ],
    },
    "nafld": {
        "flags": {"low_fat"},
        "overrides": [
            {"nutrient": "Choline", "bound": "lower", "value": 550},
            {"nutrient": "Total Fiber", "bound": "lower", "value": 30},
            {"nutrient": "Saturated fatty acids", "bound": "upper", "value": 9},
        ],
    },
    "hypothyroidism": {
        "flags": set(),
        "overrides": [
            {"nutrient": "Iodine", "bound": "lower", "value": 250},
            {"nutrient": "Selenium", "bound": "lower", "value": 100},
            {"nutrient": "Zinc", "bound": "lower", "value": 15},
        ],
    },
    "ibd_crohns": {
        "flags": set(),
        "overrides": [
            {"nutrient": "Total Fiber", "bound": "upper", "value": 15},
            {"nutrient": "Fat", "bound": "upper", "value": 50},
            {"nutrient": "Calcium", "bound": "lower", "value": 1200},
        ],
    },
    "celiac": {
        "flags": set(),
        "dietary_presets": ["gluten_free"],
        "overrides": [
            {"nutrient": "Iron", "bound": "lower", "value": 18},
            {"nutrient": "Calcium", "bound": "lower", "value": 1200},
            {"nutrient": "Folate", "bound": "lower", "value": 600},
        ],
    },
    "pregnancy": {
        "flags": set(),
        "protein_boost": 1.25,  # +25% protein
        "overrides": [
            {"nutrient": "Folate", "bound": "lower", "value": 600},
            {"nutrient": "Iron", "bound": "lower", "value": 27},
            {"nutrient": "Calcium", "bound": "lower", "value": 1000},
            {"nutrient": "Choline", "bound": "lower", "value": 450},
        ],
    },
    "preeclampsia_risk": {
        "flags": set(),
        "overrides": [
            {"nutrient": "Calcium", "bound": "lower", "value": 1500},
            {"nutrient": "Potassium", "bound": "lower", "value": 4700},
            {"nutrient": "Sodium", "bound": "upper", "value": 1500},
            {"nutrient": "Magnesium", "bound": "lower", "value": 400},
        ],
    },
    "sarcopenia": {
        "flags": set(),
        "supplements_off": ["Vitamin D"],
        "protein_boost": 1.2,  # 1.2 g/kg equivalent
        "overrides": [
            {"nutrient": "Leucine", "bound": "lower", "value": 5},
        ],
    },
    "migraine": {
        "flags": set(),
        "overrides": [
            {"nutrient": "Magnesium", "bound": "lower", "value": 500},
            # Riboflavin 400mg is a supplement dose (Schoenen 1998); food sources
            # max ~3mg/100g. Set a high-but-food-achievable floor instead.
            {"nutrient": "Riboflavin", "bound": "lower", "value": 5},
        ],
    },
}


def _apply_dietary_flags(nutrition: dict, req: OptimizeRequest) -> None:
    """Mutate nutrition bounds for low_fat, low_carb, anti_inflammatory,
    disease presets, and manual overrides. Order: flags → diseases → manual
    (manual wins over everything)."""
    from ..data import parse_bound

    # Expand disease presets into flags + overrides + dietary exclusions
    active_flags = set()
    disease_overrides: list[dict] = []
    for dp in req.disease_presets:
        preset = DISEASE_PRESETS.get(dp, {})
        active_flags |= preset.get("flags", set())
        disease_overrides.extend(preset.get("overrides", []))
        # Some disease presets inject dietary presets (e.g. celiac → gluten_free)
        for dp_name in preset.get("dietary_presets", []):
            if dp_name not in req.dietary_presets:
                req.dietary_presets.append(dp_name)

    # Protein boost from disease presets (e.g. pregnancy +25%, sarcopenia +20%)
    max_boost = max(
        (DISEASE_PRESETS.get(dp, {}).get("protein_boost", 1.0) for dp in req.disease_presets),
        default=1.0,
    )
    if max_boost > 1.0 and "Protein" in nutrition:
        cur_lb = parse_bound(nutrition["Protein"]["low_bound"])
        nutrition["Protein"]["low_bound"] = round(cur_lb * max_boost, 1)
        cur_ub = parse_bound(nutrition["Protein"]["high_bound"])
        if cur_ub != float("inf"):
            nutrition["Protein"]["high_bound"] = round(cur_ub * max_boost, 1)

    if req.low_fat or "low_fat" in active_flags:
        # Low fat = 20% of calories from fat.  Scale with energy UB if available.
        energy_ub = parse_bound(nutrition.get("Energy", {}).get("high_bound", 2800))
        fat_cap = round(energy_ub * 0.20 / 9)   # 20% of kcal / 9 kcal per g
        sat_cap = round(fat_cap * 0.30)          # sat fat ≤ 30% of total fat cap
        if "Fat" in nutrition:
            nutrition["Fat"]["low_bound"] = 0
            nutrition["Fat"]["high_bound"] = fat_cap
        if "Saturated fatty acids" in nutrition:
            nutrition["Saturated fatty acids"]["high_bound"] = sat_cap

    if req.low_carb or "low_carb" in active_flags:
        if "Carbohydrate" in nutrition:
            nutrition["Carbohydrate"]["high_bound"] = 130
        # Increase protein by 20% above current lower bound
        if "Protein" in nutrition:
            cur_lb = parse_bound(nutrition["Protein"]["low_bound"])
            nutrition["Protein"]["low_bound"] = round(cur_lb * 1.2, 1)
            cur_ub = parse_bound(nutrition["Protein"]["high_bound"])
            if cur_ub != float("inf"):
                nutrition["Protein"]["high_bound"] = round(cur_ub * 1.2, 1)

    if req.anti_inflammatory or "anti_inflammatory" in active_flags:
        if "Linolenic Acid" in nutrition:
            nutrition["Linolenic Acid"]["low_bound"] = 3.0
        if "Vitamin K" in nutrition:
            nutrition["Vitamin K"]["low_bound"] = 200

    # Apply disease-preset overrides
    for ov in disease_overrides:
        n = ov.get("nutrient", "")
        if n not in nutrition:
            continue
        if ov["bound"] == "lower":
            nutrition[n]["low_bound"] = ov["value"]
        elif ov["bound"] == "upper":
            nutrition[n]["high_bound"] = ov["value"]

    # Manual overrides (highest priority — override everything above)
    for ov in req.nutrient_overrides:
        n = ov.get("nutrient", "")
        if n not in nutrition:
            continue
        if ov.get("bound") == "lower":
            nutrition[n]["low_bound"] = ov["value"]
        elif ov.get("bound") == "upper":
            nutrition[n]["high_bound"] = ov["value"]

    # Clamp: if any lb > ub after all overrides, set lb = ub to avoid
    # infeasibility from conflicting profile + disease constraints
    # (e.g. profile sets fiber lb=36 but IBD sets fiber ub=15).
    for n, content in nutrition.items():
        lb = parse_bound(content.get("low_bound", 0))
        ub = parse_bound(content.get("high_bound", float("inf")))
        if lb != float("inf") and ub != float("inf") and lb > ub:
            content["low_bound"] = ub


# --- Endpoint helpers ---

def _solve(req: OptimizeRequest, priced_foods_path: Path) -> OptimizeResponse:
    from ..data import parse_bound
    from ..model import build_model
    from ..solve import explain_shadow_prices, solve, solve_with_min_serving

    food_info, food_matches, nutrition = load_priced_foods(priced_foods_path.name)

    # Profile-scaled DRI if demographics supplied
    profile_fields = [req.age, req.sex, req.weight_kg, req.height_cm, req.activity]
    if any(f is not None for f in profile_fields):
        if not all(f is not None for f in profile_fields):
            raise HTTPException(
                status_code=400,
                detail="All of age/sex/weight_kg/height_cm/activity must be provided together.",
            )
        profile = UserProfile(
            sex=req.sex, age=req.age,
            weight_kg=req.weight_kg, height_cm=req.height_cm,
            activity=req.activity,
        )
        nutrition = apply_profile(nutrition, profile)

    # Apply dietary presets
    excluded_preset_count = 0
    if req.dietary_presets:
        try:
            excluded = foods_excluded_by_presets(req.dietary_presets, list(food_info))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        excluded_preset_count = len(excluded)
        for food in excluded:
            food_info.pop(food, None)
            food_matches.pop(food, None)

    # Apply blacklist / whitelist (exact name match)
    for food in req.blacklist:
        if food in food_info:
            food_info.pop(food, None)
            food_matches.pop(food, None)

    if not food_info:
        raise HTTPException(status_code=400, detail="No foods remain after filtering")

    _apply_dietary_flags(nutrition, req)

    # Supplements: zero lower bounds for supplemented nutrients.
    # Disease presets can force supplements OFF (e.g. sarcopenia → Vitamin D).
    supps_off: set[str] = set()
    for dp in req.disease_presets:
        supps_off.update(DISEASE_PRESETS.get(dp, {}).get("supplements_off", []))
    _supp_display_lbs: dict[str, Any] = {}
    for nutrient, flag in [("Vitamin B12", req.b12_supplemented), ("Vitamin D", req.vitd_supplemented), ("Iodine", req.iodine_supplemented)]:
        if flag and nutrient in nutrition and nutrient not in supps_off:
            _supp_display_lbs[nutrient] = nutrition[nutrient]["low_bound"]
            nutrition[nutrient]["low_bound"] = 0

    model, variables, _cons = build_model(food_info, food_matches, nutrition)

    if req.anti_inflammatory or any(
        "anti_inflammatory" in DISEASE_PRESETS.get(dp, {}).get("flags", set())
        for dp in req.disease_presets
    ):
        # Omega-3:omega-6 ratio >= 1:2 using bioactive equivalents.
        # ALA converts to EPA/DHA at ~8% in vivo (Burdge & Calder 2005).
        # EPA and DHA are 100% bioactive. So effective omega-3 =
        #   0.08 × ALA + 1.0 × EPA + 1.0 × DHA
        # Constraint: effective_o3 - 0.25 × omega6 >= 0  (ratio >= 1:4)
        import optlang
        from ..model import _safe_name
        ALA_BIOCONVERSION = 0.08
        ratio_terms = []
        for food in food_info:
            fm_entry = food_matches.get(food, {})
            eff_o3 = (ALA_BIOCONVERSION * fm_entry.get("Linolenic Acid", 0)
                      + fm_entry.get("PUFA 20:5 n-3 (EPA)", 0)
                      + fm_entry.get("PUFA 22:6 n-3 (DHA)", 0))
            o6 = fm_entry.get("Linoleic Acid", 0)
            coef = eff_o3 - 0.25 * o6
            if coef != 0:
                safe = _safe_name(food)
                if safe in variables:
                    ratio_terms.append(coef * variables[safe])
        if ratio_terms:
            model.add(optlang.Constraint(sum(ratio_terms), lb=0, name="omega3_omega6_ratio"))

    # Whitelist: enforce lb ≥ 30g for these foods (if still in food_info)
    for food in req.whitelist:
        key = food.replace(" ", "_")
        key = "".join(c if c.isalnum() or c == "_" else "_" for c in key)
        if key in variables:
            variables[key].lb = max(variables[key].lb, 0.30)

    # Solve, with or without min-serving
    warnings = []
    shadow_prices: list[ShadowPriceOut] = []
    if req.min_serving_grams > 0:
        min_units = req.min_serving_grams / 100.0
        result = solve_with_min_serving(model, variables, min_units)
        obj, primals, constraint_values, shadows, _iters = result
        if obj is None:
            raise HTTPException(
                status_code=400,
                detail=f"Infeasible with min serving {req.min_serving_grams}g. "
                       f"Try a smaller threshold or fewer exclusions.",
            )
        # MILP doesn't produce meaningful shadow prices; leave empty.
    else:
        obj, primals, constraint_values, shadows = solve(model, extract_duals=True)
        for s in explain_shadow_prices(shadows, nutrition, top_k=8):
            pass   # rendered below with full structure
        for sp in shadows[:8]:
            key = sp.constraint.replace("_", " ")
            unit = nutrition.get(key, {}).get("units", "")
            direction = "Raising" if sp.bound == "upper" else "Lowering"
            shadow_prices.append(ShadowPriceOut(
                nutrient=key,
                bound=sp.bound,
                bound_value=sp.bound_value,
                savings_per_unit=round(sp.dual, 4),
                explanation=(
                    f"{key} is at its {sp.bound} bound ({sp.bound_value:g} {unit}). "
                    f"{direction} it by 1 {unit} would save ${sp.dual:.3f}/day."
                ),
            ))

    # Build nutrient-status table from constraint values (skip non-nutrient constraints)
    nutrients_out: list[NutrientStatus] = []
    for name, cv in constraint_values.items():
        if name == "volume" or name.startswith("mincap_"):
            continue
        key = name.replace("_", " ")
        if key not in nutrition:
            continue
        n_lb = parse_bound(nutrition[key].get("low_bound", 0))
        n_ub = parse_bound(nutrition[key].get("high_bound", float("inf")))
        # Restore real lower bounds for display when supplemented
        display_lb = n_lb
        if key in _supp_display_lbs:
            display_lb = parse_bound(_supp_display_lbs[key])
        value = cv["val"] or 0.0
        pct = (value / display_lb * 100) if display_lb and display_lb > 0 else None
        binding = None
        if n_lb and n_lb > 0 and abs(value - n_lb) < max(1e-3, n_lb * 1e-4):
            binding = "lower"
        elif n_ub and n_ub != float("inf") and abs(value - n_ub) < max(1e-3, n_ub * 1e-4):
            binding = "upper"
        cat_rank, cat_label = _category_for(key)
        nutrients_out.append(NutrientStatus(
            nutrient=key,
            value=round(value, 2),
            low_bound=None if not display_lb or display_lb == float("inf") else display_lb,
            high_bound=None if n_ub == float("inf") else n_ub,
            units=nutrition[key].get("units", ""),
            pct_of_lower=round(pct, 1) if pct is not None else None,
            binding=binding,
            category=cat_label,
        ))
    # Sort by category (macros → water → vitamins → minerals → amino acids),
    # then by canonical position within category.
    nutrients_out.sort(key=lambda n: (
        _category_for(n.nutrient)[0],
        _NUTRIENT_ORDER.get(n.nutrient, 999),
    ))

    # Assemble food list
    foods_out: list[FoodEntry] = []
    import json as _json
    with open(priced_foods_path) as f:
        priced_raw = _json.load(f)
    for key, amount in sorted(primals.items(), key=lambda kv: -kv[1]):
        name = _reverse_key_lookup(key, priced_raw)
        if name is None:
            continue
        entry = priced_raw.get(name, {})
        foods_out.append(FoodEntry(
            food=name,
            grams=round(amount * 100),
            price_per_100g=round(entry.get("price_per_100g", 0.0), 3),
            price_source=entry.get("price_source", ""),
        ))

    return OptimizeResponse(
        cost_per_day=round(obj, 2),
        foods=foods_out,
        excluded_by_preset=excluded_preset_count,
        warnings=warnings,
        nutrients=nutrients_out,
        shadow_prices=shadow_prices,
    )


def _reverse_key_lookup(safe_key: str, priced: dict) -> str | None:
    """Match a safe_name variable back to its priced_foods.json key."""
    for name in priced:
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name.replace(" ", "_"))
        if safe == safe_key:
            return name
    return None


def _weekly_event_stream(
    req: OptimizeRequest,
    priced_foods_path: Path,
    days: int = 7,
    max_days_per_food: int = 4,
    pool_size: int = 150,
    time_limit_sec: float = 120.0,
    mip_gap: float = 0.02,
):
    """Stream NDJSON events for a 7-day variety diet.

    Uses the global weekly MILP (`diet_opt.weekly_model.build_weekly_model`)
    over a pre-filtered pool of the ~60 most cost-effective foods. The MILP
    jointly optimises 7 days with a hard rotation cap (each food ≤ K days)
    and semi-continuous per-day servings (≥30g if served), producing truly
    distinct daily menus at minimum total weekly cost.

    Sequential daily LPs with a soft reuse penalty were tried first but
    produced degenerate identical solutions on days 5-7 because the LP's
    per-food 400g ceiling binds across many alternatives. The MILP avoids
    that by globally coordinating the week's rotation.

    Event shapes (one JSON object per newline):
      {"event": "start", "days": N, "max_days_per_food": K, "total_foods": M,
       "excluded_by_preset": E, "pool_size": P}
      {"event": "reference", "cost": $, "nutrients": [...], "shadow_prices": [...],
       "elapsed_s": S}
      {"event": "solving", "pool_foods": P, "time_limit_sec": S, "elapsed_s": S}
      {"event": "day", "day": D, "cost": $, "foods": [...], "elapsed_s": S,
       "total_foods_seen": M}
      {"event": "done", "total_cost": $, "avg_cost_per_day": $,
       "unique_foods": N, "elapsed_s": S}
      {"event": "error", "message": "..."}
    """
    import json as _json
    import time

    from ..data import load_priced_foods, parse_bound
    from ..model import build_model
    from ..solve import solve
    from ..weekly_model import (
        build_weekly_model, extract_weekly_solution,
        filter_substitutes, preselect_foods_by_profile, profile_to_emphasis,
    )

    def emit(obj: dict) -> bytes:
        return (_json.dumps(obj) + "\n").encode()

    try:
        food_info, food_matches, nutrition = load_priced_foods(priced_foods_path.name)
    except Exception as e:
        yield emit({"event": "error", "message": f"load failed: {e}"})
        return

    profile_fields = [req.age, req.sex, req.weight_kg, req.height_cm, req.activity]
    if any(f is not None for f in profile_fields):
        if not all(f is not None for f in profile_fields):
            yield emit({"event": "error",
                        "message": "profile scaling requires all of age/sex/weight_kg/height_cm/activity"})
            return
        profile = UserProfile(
            sex=req.sex, age=req.age, weight_kg=req.weight_kg,
            height_cm=req.height_cm, activity=req.activity,
        )
        nutrition = apply_profile(nutrition, profile)

    excluded_preset_count = 0
    if req.dietary_presets:
        try:
            excluded = foods_excluded_by_presets(req.dietary_presets, list(food_info))
        except ValueError as e:
            yield emit({"event": "error", "message": str(e)})
            return
        excluded_preset_count = len(excluded)
        for food in excluded:
            food_info.pop(food, None)
            food_matches.pop(food, None)

    for food in req.blacklist:
        food_info.pop(food, None)
        food_matches.pop(food, None)

    if not food_info:
        yield emit({"event": "error", "message": "No foods remain after filtering"})
        return

    priced_raw = _json.loads(priced_foods_path.read_text())

    _apply_dietary_flags(nutrition, req)

    supps_off: set[str] = set()
    for dp in req.disease_presets:
        supps_off.update(DISEASE_PRESETS.get(dp, {}).get("supplements_off", []))
    _supp_display_lbs: dict[str, Any] = {}
    for nutrient, flag in [("Vitamin B12", req.b12_supplemented), ("Vitamin D", req.vitd_supplemented), ("Iodine", req.iodine_supplemented)]:
        if flag and nutrient in nutrition and nutrient not in supps_off:
            _supp_display_lbs[nutrient] = nutrition[nutrient]["low_bound"]
            nutrition[nutrient]["low_bound"] = 0

    yield emit({
        "event": "start",
        "days": days,
        "max_days_per_food": max_days_per_food,
        "pool_size": pool_size,
        "excluded_by_preset": excluded_preset_count,
        "total_foods": len(food_info),
    })

    t0 = time.perf_counter()

    # --- Reference solve: unperturbed, for nutrient table + shadow prices ---
    try:
        ref_model, ref_vars, _ = build_model(food_info, food_matches, nutrition, include_volume=False)
        for food in req.whitelist:
            key = food.replace(" ", "_")
            key = "".join(c if c.isalnum() or c == "_" else "_" for c in key)
            if key in ref_vars:
                ref_vars[key].lb = max(ref_vars[key].lb, 0.30)
        # Add omega-3 ratio to the reference LP so the pool seeds
        # include fish/flax needed for anti-inflammatory MILP.
        _needs_anti_inflam = req.anti_inflammatory or any(
            "anti_inflammatory" in DISEASE_PRESETS.get(dp, {}).get("flags", set())
            for dp in req.disease_presets
        )
        if _needs_anti_inflam:
            import optlang
            from ..model import _safe_name
            ALA_BIO = 0.08
            _ratio_terms = []
            for food in food_info:
                e = food_matches.get(food, {})
                eff_o3 = (ALA_BIO * e.get("Linolenic Acid", 0)
                          + e.get("PUFA 20:5 n-3 (EPA)", 0)
                          + e.get("PUFA 22:6 n-3 (DHA)", 0))
                o6 = e.get("Linoleic Acid", 0)
                coef = eff_o3 - 0.25 * o6
                safe = _safe_name(food)
                if coef != 0 and safe in ref_vars:
                    _ratio_terms.append(coef * ref_vars[safe])
            if _ratio_terms:
                ref_model.add(optlang.Constraint(
                    sum(_ratio_terms), lb=0, name="omega3_omega6_ratio"
                ))
        from ..solve import explain_shadow_prices
        ref_obj, ref_primals, ref_cv, ref_shadows = solve(ref_model, extract_duals=True)

        nutrients_out: list[dict] = []
        for name, cv in ref_cv.items():
            if name == "volume" or name.startswith("mincap_"):
                continue
            key = name.replace("_", " ")
            if key not in nutrition:
                continue
            n_lb = parse_bound(nutrition[key].get("low_bound", 0))
            n_ub = parse_bound(nutrition[key].get("high_bound", float("inf")))
            display_lb = n_lb
            if key in _supp_display_lbs:
                display_lb = parse_bound(_supp_display_lbs[key])
            value = cv["val"] or 0.0
            pct = (value / display_lb * 100) if display_lb and display_lb > 0 else None
            binding = None
            if n_lb and n_lb > 0 and abs(value - n_lb) < max(1e-3, n_lb * 1e-4):
                binding = "lower"
            elif n_ub and n_ub != float("inf") and abs(value - n_ub) < max(1e-3, n_ub * 1e-4):
                binding = "upper"
            _, cat_label = _category_for(key)
            nutrients_out.append({
                "nutrient": key,
                "value": round(value, 2),
                "low_bound": None if not display_lb or display_lb == float("inf") else display_lb,
                "high_bound": None if n_ub == float("inf") else n_ub,
                "units": nutrition[key].get("units", ""),
                "pct_of_lower": round(pct, 1) if pct is not None else None,
                "binding": binding,
                "category": cat_label,
            })
        nutrients_out.sort(key=lambda n: (
            _category_for(n["nutrient"])[0],
            _NUTRIENT_ORDER.get(n["nutrient"], 999),
        ))

        shadow_out: list[dict] = []
        for sp in ref_shadows[:8]:
            key = sp.constraint.replace("_", " ")
            unit = nutrition.get(key, {}).get("units", "")
            direction = "Raising" if sp.bound == "upper" else "Lowering"
            shadow_out.append({
                "nutrient": key,
                "bound": sp.bound,
                "bound_value": sp.bound_value,
                "savings_per_unit": round(sp.dual, 4),
                "explanation": (
                    f"{key} is at its {sp.bound} bound ({sp.bound_value:g} {unit}). "
                    f"{direction} it by 1 {unit} would save ${sp.dual:.3f}/day."
                ),
            })

        yield emit({
            "event": "reference",
            "cost": round(ref_obj, 2),
            "nutrients": nutrients_out,
            "shadow_prices": shadow_out,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        })
    except Exception as e:
        yield emit({"event": "error", "message": f"reference solve failed: {e}"})
        return

    # --- Weekly MILP: adaptive cap + adaptive pool size ---
    # 1. Start at tightest rotation cap (most variety), relax by +1 if infeasible.
    # 2. If all caps fail, expand the pool and retry from cap=max_days_per_food.
    # This handles tightly constrained disease states that need more food options.
    emphasis = profile_to_emphasis(
        sex=req.sex, age=req.age, activity=req.activity,
    )
    milp_kwargs: dict = {}
    _needs_anti_inflam = req.anti_inflammatory or any(
        "anti_inflammatory" in DISEASE_PRESETS.get(dp, {}).get("flags", set())
        for dp in req.disease_presets
    )
    if _needs_anti_inflam:
        milp_kwargs["omega3_omega6_ratio"] = 0.25

    per_day = None
    weekly = None
    pool_sizes_to_try = [pool_size, pool_size * 2, len(food_info)]  # 150, 300, all

    for current_pool_size in pool_sizes_to_try:
        pool_names = preselect_foods_by_profile(
            food_info, food_matches, nutrition, ref_primals,
            emphasis=emphasis, extra_count=current_pool_size,
        )
        pool_info = {n: food_info[n] for n in pool_names if n in food_info}
        pool_matches = {n: food_matches[n] for n in pool_names if n in food_matches}

        yield emit({
            "event": "solving",
            "pool_foods": len(pool_info),
            "time_limit_sec": time_limit_sec,
            "mip_gap": mip_gap,
            "emphasis": emphasis,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        })

        for attempt_cap in range(max_days_per_food, days + 1):
            try:
                remaining_attempts = days + 1 - attempt_cap
                attempt_time = max(30.0, time_limit_sec / max(remaining_attempts, 1))
                weekly = build_weekly_model(
                    pool_info, pool_matches, nutrition,
                    days=days, max_days_per_food=attempt_cap,
                    min_serving_units=0.30,
                    time_limit_sec=attempt_time,
                    mip_gap=mip_gap,
                    **milp_kwargs,
                )
            except Exception as e:
                yield emit({"event": "error", "message": f"weekly MILP failed: {e}"})
                return

            per_day = extract_weekly_solution(weekly)
            if per_day:
                note_parts = []
                if attempt_cap > max_days_per_food:
                    note_parts.append(f"relaxed rotation cap to {attempt_cap}")
                if current_pool_size > pool_size:
                    note_parts.append(f"expanded pool to {len(pool_info)} foods")
                if note_parts:
                    yield emit({
                        "event": "solving",
                        "pool_foods": len(pool_info),
                        "time_limit_sec": attempt_time,
                        "mip_gap": mip_gap,
                        "emphasis": emphasis,
                        "elapsed_s": round(time.perf_counter() - t0, 1),
                        "note": "; ".join(note_parts),
                    })
                break
        if per_day:
            break

    if not per_day:
        status_str = "unknown"
        if weekly:
            try:
                obj_val = weekly.model.getObjectiveValue()
                status_str = f"obj={obj_val:.2f}"
            except Exception:
                pass
        yield emit({"event": "error",
                    "message": f"weekly MILP infeasible after trying pools up to {len(pool_info)} foods "
                               f"and cap up to {days} ({status_str}). "
                               f"Try fewer dietary restrictions."})
        return

    per_day_cost: list[float] = []
    all_foods_seen: set[str] = set()

    # Emit one "day" event per day (already solved — just rendering)
    for d in sorted(per_day):
        foods_today = per_day[d]
        foods_out: list[dict] = []
        true_cost = 0.0
        for name, grams in sorted(foods_today.items(), key=lambda kv: -kv[1]):
            entry = priced_raw.get(name, {})
            orig = food_info.get(name)
            if orig:
                per_100g_cost = orig["price"] / max(orig.get("yield", 1.0), 0.01) / 4.54
                cost_today = per_100g_cost * (grams / 100.0)
                true_cost += cost_today
            else:
                cost_today = 0.0
            foods_out.append({
                "food": name,
                "grams": round(grams),
                "price_per_100g": round(entry.get("price_per_100g", 0.0), 3),
                "price_source": entry.get("price_source", ""),
                "cost_today": round(cost_today, 3),
            })
            all_foods_seen.add(name)
        per_day_cost.append(true_cost)
        yield emit({
            "event": "day",
            "day": d + 1,
            "cost": round(true_cost, 2),
            "foods": foods_out,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "total_foods_seen": len(all_foods_seen),
        })

    # --- Compute per-day nutrient levels, average across the week ---
    from ..model import GRAMS_PER_LITER
    avg_nutrient_vals: dict[str, float] = {}
    n_days_solved = len(per_day)
    for d in sorted(per_day):
        for nutrient in nutrition:
            val = 0.0
            for food, grams in per_day[d].items():
                amount = food_matches.get(food, {}).get(nutrient, 0)
                if nutrient == "Total Water":
                    amount /= GRAMS_PER_LITER
                val += amount * (grams / 100.0)
            avg_nutrient_vals[nutrient] = avg_nutrient_vals.get(nutrient, 0) + val / n_days_solved

    avg_nutrients_out: list[dict] = []
    for nutrient, content in nutrition.items():
        key = nutrient
        value = avg_nutrient_vals.get(nutrient, 0)
        n_lb = parse_bound(content.get("low_bound", 0))
        n_ub = parse_bound(content.get("high_bound", float("inf")))
        display_lb = n_lb
        if key in _supp_display_lbs:
            display_lb = parse_bound(_supp_display_lbs[key])
        pct = (value / display_lb * 100) if display_lb and display_lb > 0 else None
        binding = None
        if n_lb and n_lb > 0 and abs(value - n_lb) < max(1e-3, n_lb * 0.05):
            binding = "lower"
        elif n_ub and n_ub != float("inf") and abs(value - n_ub) < max(1e-3, n_ub * 0.05):
            binding = "upper"
        _, cat_label = _category_for(key)
        avg_nutrients_out.append({
            "nutrient": key,
            "value": round(value, 2),
            "low_bound": None if not display_lb or display_lb == float("inf") else display_lb,
            "high_bound": None if n_ub == float("inf") else n_ub,
            "units": content.get("units", ""),
            "pct_of_lower": round(pct, 1) if pct is not None else None,
            "binding": binding,
            "category": cat_label,
        })
    # --- Polyphenol content (informational, not an LP constraint) ---
    from ..objectives import load_polyphenol_content
    polyphenols = load_polyphenol_content()
    avg_pp = 0.0
    for d in sorted(per_day):
        day_pp = sum(
            polyphenols.get(food, 0) * (grams / 100.0)
            for food, grams in per_day[d].items()
        )
        avg_pp += day_pp / n_days_solved
    avg_nutrients_out.append({
        "nutrient": "Total Polyphenols",
        "value": round(avg_pp, 1),
        "low_bound": None,
        "high_bound": None,
        "units": "mg",
        "pct_of_lower": None,
        "binding": None,
        "category": "Bioactives",
    })

    # Effective omega-3 (bioactive equivalents: 8% ALA + 100% EPA + 100% DHA)
    _ALA_BIO = 0.08
    avg_eff_o3 = 0.0
    for d in sorted(per_day):
        day_o3 = sum(
            (_ALA_BIO * food_matches.get(food, {}).get("Linolenic Acid", 0)
             + food_matches.get(food, {}).get("PUFA 20:5 n-3 (EPA)", 0)
             + food_matches.get(food, {}).get("PUFA 22:6 n-3 (DHA)", 0))
            * (grams / 100.0)
            for food, grams in per_day[d].items()
        )
        avg_eff_o3 += day_o3 / n_days_solved
    avg_nutrients_out.append({
        "nutrient": "Effective Omega-3",
        "value": round(avg_eff_o3, 2),
        "low_bound": None,
        "high_bound": None,
        "units": "grams",
        "pct_of_lower": None,
        "binding": None,
        "category": "Bioactives",
    })

    avg_nutrients_out.sort(key=lambda n: (
        _category_for(n["nutrient"])[0],
        _NUTRIENT_ORDER.get(n["nutrient"], 999),
    ))

    yield emit({
        "event": "done",
        "total_cost": round(sum(per_day_cost), 2),
        "avg_cost_per_day": round(sum(per_day_cost) / max(days, 1), 2),
        "unique_foods": len(all_foods_seen),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "nutrients": avg_nutrients_out,
        "shadow_prices": shadow_out,
        "citations": _relevant_citations(req),
    })


def _generate_meal_plan(req: MealPlanRequest) -> MealPlanResponse:
    """Import the meal-plan generator and run it."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Server missing ANTHROPIC_API_KEY; meal plan disabled.",
        )
    os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"].strip()

    # The meal-plan script lives outside the package; import by path.
    import importlib.util
    script_path = REPO_ROOT / "scripts" / "generate_meal_plan.py"
    spec = importlib.util.spec_from_file_location("meal_plan_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Multiply the daily diet by days
    lp_totals = {food: grams * req.days for food, grams in req.diet.items()}
    if not lp_totals:
        raise HTTPException(status_code=400, detail="Empty diet")

    import anthropic
    client = anthropic.Anthropic()
    try:
        plan, violations = mod.generate_with_retries(
            client=client,
            model=req.model,
            lp_totals=lp_totals,
            days=req.days,
            cuisine_style=req.cuisine_style,
            max_prep_min=req.max_prep_min,
            blacklist=req.blacklist,
            whitelist=req.whitelist,
            tolerance_g=5.0,
            max_retries=3,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e}")

    if plan is None:
        raise HTTPException(status_code=500, detail="Meal plan generation failed")

    # Post-hoc rebalance if there are violations
    if violations:
        mod.rebalance_plan(plan, lp_totals)
        violations = mod.validate_plan(plan, lp_totals, tolerance_g=5.0)

    yields = mod.load_cooking_yields()
    markdown = mod.render_markdown(plan, yields=yields)

    return MealPlanResponse(
        plan=plan.model_dump(),
        markdown=markdown,
        violations=violations,
    )


# --- App factory ---

def create_app(priced_foods_path: Path | None = None) -> FastAPI:
    if priced_foods_path is None:
        priced_foods_path = DEFAULT_PRICED_FOODS
    priced_foods_path = Path(priced_foods_path)

    app = FastAPI(title="diet-opt", description="Cost-minimizing nutritional diet LP + meal plan")

    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])
        app.state.limiter = limiter
    except ImportError:
        pass

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/presets")
    async def presets() -> dict[str, Any]:
        return {"presets": list_presets()}

    @app.get("/foods")
    async def foods() -> dict[str, Any]:
        if not priced_foods_path.exists():
            return {"foods": []}
        data = json.loads(priced_foods_path.read_text())
        return {"foods": sorted(data.keys())}

    @app.post("/optimize", response_model=OptimizeResponse)
    async def optimize(req: OptimizeRequest, request: Request) -> OptimizeResponse:
        if not priced_foods_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"{priced_foods_path.name} not found on server. "
                       "Run scripts/build_priced_foods.py.",
            )
        return _solve(req, priced_foods_path)

    @app.post("/optimize-weekly")
    async def optimize_weekly(req: OptimizeRequest, request: Request) -> StreamingResponse:
        """Streaming NDJSON: reference solve, 7 daily LPs with rotation cap,
        then a done event. One JSON object per line; client parses
        line-by-line and renders each day's column as it arrives."""
        if not priced_foods_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"{priced_foods_path.name} not found on server.",
            )
        return StreamingResponse(
            _weekly_event_stream(req, priced_foods_path),
            media_type="application/x-ndjson",
        )

    @app.post("/meal-plan", response_model=MealPlanResponse)
    async def meal_plan(req: MealPlanRequest, request: Request) -> MealPlanResponse:
        return _generate_meal_plan(req)

    @app.get("/health")
    async def health() -> dict:
        meta = {}
        if priced_foods_path.exists():
            try:
                data = json.loads(priced_foods_path.read_text())
                meta = data.get("_metadata") or {}
            except Exception:
                pass
        return {
            "status": "ok",
            "priced_foods_present": priced_foods_path.exists(),
            "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "market": meta,
        }

    return app


app = create_app()
