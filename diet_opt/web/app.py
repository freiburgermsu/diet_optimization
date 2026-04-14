"""FastAPI app: one POST /optimize endpoint, one static HTML form.

Deployment target: Hugging Face Spaces (free, persistent, FastAPI
support) per my comment on the issue — Railway free tier is gone and
Fly.io has been shrinking.

Rate limiting: SlowAPI, 10/min/IP. Prevents trivial abuse; 1s solves
make this sufficient.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..data import load_pipeline_inputs

STATIC_DIR = Path(__file__).resolve().parent / "static"


class OptimizeRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: str = Field(..., pattern="^(male|female)$")
    weight_kg: float = Field(..., gt=20, lt=300)
    activity_level: str = Field(..., pattern="^(sedentary|moderate|active)$")
    budget_ceiling: float | None = Field(None, gt=0)
    blacklist: list[str] = Field(default_factory=list)
    whitelist: list[str] = Field(default_factory=list)


class OptimizeResponse(BaseModel):
    cost_per_day: float
    shopping_list: list[dict]   # [{food, grams, cost}]
    notes: list[str]


def _solve_for_request(req: OptimizeRequest) -> OptimizeResponse:
    """Build model, apply prefs, solve. Split out for testability."""
    from ..model import build_model
    from ..solve import solve

    food_info, food_matches, nutrition = load_pipeline_inputs()
    model, variables, _constraints = build_model(food_info, food_matches, nutrition)

    # Blacklist/whitelist application lives in diet_opt.prefs (#13).
    # Re-wire once #13 merges — intentionally left here as the extension
    # point so a merge of #13 only needs to add the import + 3 lines.
    for food in req.blacklist:
        key = food.replace(" ", "_")
        if key in variables:
            variables[key].ub = 0
        else:
            raise HTTPException(status_code=400, detail=f"Unknown food: {food}")

    objective, primals, _cv, _duals = solve(model)
    shopping = []
    for food_key, amount in sorted(primals.items(), key=lambda kv: -kv[1]):
        food_name = food_key.replace("_", " ")
        grams = int(amount * 100)
        price_per_100g = food_info.get(food_name, {}).get("price", 0) / \
            max(food_info.get(food_name, {}).get("yield", 1), 0.01) / 4.54
        shopping.append({
            "food": food_name,
            "grams": grams,
            "cost": round(amount * price_per_100g, 2),
        })
    notes = []
    if req.budget_ceiling and objective > req.budget_ceiling:
        notes.append(f"Cost \\${objective:.2f} exceeds your budget of \\${req.budget_ceiling:.2f}")
    return OptimizeResponse(cost_per_day=round(objective, 2), shopping_list=shopping, notes=notes)


def create_app() -> FastAPI:
    app = FastAPI(title="diet-opt", description="Cost-minimizing nutritional diet")

    # SlowAPI is optional — if not installed, rate limiting is disabled
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
        app.state.limiter = limiter
    except ImportError:
        limiter = None

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC_DIR / "index.html").read_text()

    @app.post("/optimize", response_model=OptimizeResponse)
    async def optimize(req: OptimizeRequest, request: Request) -> OptimizeResponse:
        return _solve_for_request(req)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
