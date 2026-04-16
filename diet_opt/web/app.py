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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..data import load_priced_foods
from ..presets import foods_excluded_by_presets, list_presets

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PRICED_FOODS = REPO_ROOT / "priced_foods.json"


# --- Request / response schemas ---

class OptimizeRequest(BaseModel):
    min_serving_grams: float = Field(0.0, ge=0, le=200)
    dietary_presets: list[str] = Field(default_factory=list)
    blacklist: list[str] = Field(default_factory=list)
    whitelist: list[str] = Field(default_factory=list)


class FoodEntry(BaseModel):
    food: str
    grams: int
    price_per_100g: float
    price_source: str


class OptimizeResponse(BaseModel):
    cost_per_day: float
    foods: list[FoodEntry]
    excluded_by_preset: int
    warnings: list[str] = Field(default_factory=list)


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
    from ..model import build_model
    from ..solve import solve, solve_with_min_serving

    food_info, food_matches, nutrition = load_priced_foods(priced_foods_path.name)

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
    if req.min_serving_grams > 0:
        min_units = req.min_serving_grams / 100.0
        result = solve_with_min_serving(model, variables, min_units)
        obj, primals, _cv, _sp, _iters = result
        if obj is None:
            raise HTTPException(
                status_code=400,
                detail=f"Infeasible with min serving {req.min_serving_grams}g. "
                       f"Try a smaller threshold or fewer exclusions.",
            )
    else:
        obj, primals, _cv, _sp = solve(model)

    # Assemble response
    foods_out: list[FoodEntry] = []
    import json as _json
    with open(priced_foods_path) as f:
        priced_raw = _json.load(f)
    for key, amount in sorted(primals.items(), key=lambda kv: -kv[1]):
        # variable names are safe-slugified; reverse-look up original
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
    )


def _reverse_key_lookup(safe_key: str, priced: dict) -> str | None:
    """Match a safe_name variable back to its priced_foods.json key."""
    for name in priced:
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name.replace(" ", "_"))
        if safe == safe_key:
            return name
    return None


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
