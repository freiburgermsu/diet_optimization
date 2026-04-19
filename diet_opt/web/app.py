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
}
_NUTRIENT_ORDER = {name: i for i, name in enumerate(NUTRIENT_CATEGORIES)}


def _category_for(nutrient: str) -> tuple[int, str]:
    return NUTRIENT_CATEGORIES.get(nutrient, (99, "Other"))


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
    activity: str | None = Field(None, pattern="^(sedentary|light|moderate|active|very_active)$")


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

    model, variables, _cons = build_model(food_info, food_matches, nutrition)

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
        value = cv["val"] or 0.0
        pct = (value / n_lb * 100) if n_lb and n_lb > 0 else None
        binding = None
        if n_lb and n_lb > 0 and abs(value - n_lb) < max(1e-3, n_lb * 1e-4):
            binding = "lower"
        elif n_ub and n_ub != float("inf") and abs(value - n_ub) < max(1e-3, n_ub * 1e-4):
            binding = "upper"
        cat_rank, cat_label = _category_for(key)
        nutrients_out.append(NutrientStatus(
            nutrient=key,
            value=round(value, 2),
            low_bound=None if not n_lb or n_lb == float("inf") else n_lb,
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
    pool_size: int = 60,
    time_limit_sec: float = 30.0,
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
        preselect_foods_by_profile, profile_to_emphasis,
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
            value = cv["val"] or 0.0
            pct = (value / n_lb * 100) if n_lb and n_lb > 0 else None
            binding = None
            if n_lb and n_lb > 0 and abs(value - n_lb) < max(1e-3, n_lb * 1e-4):
                binding = "lower"
            elif n_ub and n_ub != float("inf") and abs(value - n_ub) < max(1e-3, n_ub * 1e-4):
                binding = "upper"
            _, cat_label = _category_for(key)
            nutrients_out.append({
                "nutrient": key,
                "value": round(value, 2),
                "low_bound": None if not n_lb or n_lb == float("inf") else n_lb,
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

    # --- Pre-filter to ~60 most cost-effective foods for the MILP ---
    emphasis = profile_to_emphasis(
        sex=req.sex, age=req.age, activity=req.activity,
    )
    pool_names = preselect_foods_by_profile(
        food_info, food_matches, nutrition, ref_primals,
        emphasis=emphasis, extra_count=pool_size,
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

    # --- Weekly MILP: jointly optimize 7 days with rotation cap ---
    try:
        weekly = build_weekly_model(
            pool_info, pool_matches, nutrition,
            days=days, max_days_per_food=max_days_per_food,
            min_serving_units=0.30,
            time_limit_sec=time_limit_sec,
            mip_gap=mip_gap,
        )
    except Exception as e:
        yield emit({"event": "error", "message": f"weekly MILP failed: {e}"})
        return

    per_day = extract_weekly_solution(weekly)
    if not per_day:
        yield emit({"event": "error",
                    "message": "weekly MILP returned no solution (infeasible or timed out)"})
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

    yield emit({
        "event": "done",
        "total_cost": round(sum(per_day_cost), 2),
        "avg_cost_per_day": round(sum(per_day_cost) / max(days, 1), 2),
        "unique_foods": len(all_foods_seen),
        "elapsed_s": round(time.perf_counter() - t0, 1),
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
        return {
            "status": "ok",
            "priced_foods_present": priced_foods_path.exists(),
            "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        }

    return app


app = create_app()
