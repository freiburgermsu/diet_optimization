"""Command-line entry point."""
from __future__ import annotations

import argparse
import sys

from .data import load_pipeline_inputs, load_priced_foods, validate_bounds
from .model import build_model
from .overrides import apply_overrides, load_overrides
from .report import plot_bounds, write_diet_csv
from .solve import explain_shadow_prices, solve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diet-opt")
    sub = parser.add_subparsers(dest="cmd", required=True)
    opt = sub.add_parser("optimize", help="solve the LP and emit optimum_diet.csv + optimized_diet.png")
    opt.add_argument("--sensitivity", action="store_true", help="also print shadow prices (uses simplex, not exact)")
    opt.add_argument(
        "--priced-foods", default=None,
        help="path to priced_foods.json (from scripts/build_priced_foods.py). "
             "When set, replaces food_info.json + food_matches.json "
             "with the ~610-food expanded table (Kroger + TFP).",
    )
    sub.add_parser("validate", help="check DRI bounds for lb > ub inversions")
    args = parser.parse_args(argv)

    if getattr(args, "priced_foods", None):
        food_info, food_matches, nutrition = load_priced_foods(args.priced_foods)
        print(
            f"loaded {len(food_info)} foods from {args.priced_foods}",
            file=sys.stderr,
        )
    else:
        food_info, food_matches, nutrition = load_pipeline_inputs()
    try:
        nutrition = apply_overrides(nutrition, load_overrides())
    except FileNotFoundError:
        pass

    if args.cmd == "validate":
        violations = validate_bounds(nutrition)
        if violations:
            print("Invalid bounds:", *violations, sep="\n  ")
            return 1
        print(f"OK: {len(nutrition)} nutrients have valid bounds")
        return 0

    if args.cmd == "optimize":
        model, _vars, _cons = build_model(food_info, food_matches, nutrition)
        objective, primals, constraint_values, shadows = solve(
            model, extract_duals=args.sensitivity
        )
        write_diet_csv(primals)
        plot_bounds(constraint_values, nutrition)
        print(f"objective = ${objective:.2f}/day")
        for food, amount in sorted(primals.items(), key=lambda kv: -kv[1]):
            print(f"  {food:30s} {int(amount * 100)} g")
        if shadows:
            print("\nBinding constraints (shadow prices):")
            for line in explain_shadow_prices(shadows, nutrition):
                print(f"  {line}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
