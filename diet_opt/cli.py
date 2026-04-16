"""Command-line entry point."""
from __future__ import annotations

import argparse
import sys

from .data import load_pipeline_inputs, load_priced_foods, validate_bounds
from .dri import ACTIVITY_PAL, UserProfile, apply_profile
from .model import build_model
from .overrides import apply_overrides, load_overrides
from .presets import foods_excluded_by_presets, list_presets
from .report import plot_bounds, write_diet_csv
from .solve import explain_shadow_prices, solve, solve_with_min_serving


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
    opt.add_argument(
        "--min-serving-grams", type=float, default=0.0,
        help="minimum grams/day a food must contribute IF included. "
             "Foods below this threshold are iteratively dropped from the LP "
             "(semi-continuous heuristic). Default 0 disables; pass e.g. 30 "
             "to drop any food whose optimal quantity would be below 30g/day.",
    )
    opt.add_argument(
        "--dietary-preset", default="",
        help=f"comma-separated dietary preset names to exclude matching foods. "
             f"Available presets: {', '.join(list_presets())}. "
             f"Multiple presets union together (e.g. 'vegan,gluten_free').",
    )
    opt.add_argument(
        "--age", type=int, default=None,
        help="age in years; enables profile-scaled DRI (energy, protein, "
             "fiber, water, iron, calcium, etc. adjusted for user demographics)",
    )
    opt.add_argument("--sex", choices=["male", "female", "nonbinary"], default=None)
    opt.add_argument("--weight-kg", type=float, default=None)
    opt.add_argument("--height-cm", type=float, default=None)
    opt.add_argument("--activity", choices=list(ACTIVITY_PAL), default=None,
                     help="physical activity level for BMR × PAL scaling")
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

    # Profile-dependent DRI scaling (new). All four flags must be provided
    # together; otherwise fall back to the baseline nutrition.json (28yo
    # active 150lb male).
    profile_flags = [
        getattr(args, "age", None),
        getattr(args, "sex", None),
        getattr(args, "weight_kg", None),
        getattr(args, "height_cm", None),
        getattr(args, "activity", None),
    ]
    if any(f is not None for f in profile_flags):
        missing = [name for name, val in zip(
            ("--age", "--sex", "--weight-kg", "--height-cm", "--activity"),
            profile_flags) if val is None]
        if missing:
            print(f"ERROR: profile scaling requires all of "
                  f"{', '.join(missing)}", file=sys.stderr)
            return 1
        profile = UserProfile(
            sex=args.sex, age=args.age,
            weight_kg=args.weight_kg, height_cm=args.height_cm,
            activity=args.activity,
        )
        nutrition = apply_profile(nutrition, profile)
        print(
            f"DRI scaled for {profile.sex}, {profile.age}y, "
            f"{profile.weight_kg}kg, {profile.height_cm}cm, {profile.activity}",
            file=sys.stderr,
        )

    if args.cmd == "validate":
        violations = validate_bounds(nutrition)
        if violations:
            print("Invalid bounds:", *violations, sep="\n  ")
            return 1
        print(f"OK: {len(nutrition)} nutrients have valid bounds")
        return 0

    if args.cmd == "optimize":
        # Apply dietary presets before building the model — filters food_info.
        if args.dietary_preset:
            preset_names = [p.strip() for p in args.dietary_preset.split(",") if p.strip()]
            try:
                excluded = foods_excluded_by_presets(preset_names, list(food_info))
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
            for food in excluded:
                food_info.pop(food, None)
                food_matches.pop(food, None)
            print(
                f"dietary preset {preset_names} excluded {len(excluded)} foods; "
                f"{len(food_info)} remain",
                file=sys.stderr,
            )
            if excluded:
                sample = excluded[:8]
                print(f"  examples: {', '.join(sample)}"
                      + (f", +{len(excluded) - 8} more" if len(excluded) > 8 else ""),
                      file=sys.stderr)

        model, variables, _cons = build_model(food_info, food_matches, nutrition)

        if args.min_serving_grams > 0:
            min_units = args.min_serving_grams / 100.0  # variable units = 100g each
            result = solve_with_min_serving(
                model, variables, min_serving_units=min_units,
                extract_duals=args.sensitivity,
            )
            objective, primals, constraint_values, shadows, iters = result
            if objective is None:
                print(f"ERROR: infeasible after dropping foods below "
                      f"{args.min_serving_grams}g/day. Try a smaller threshold.",
                      file=sys.stderr)
                return 1
            print(f"converged in {iters} iteration(s) with "
                  f"min-serving {args.min_serving_grams}g/day",
                  file=sys.stderr)
        else:
            objective, primals, constraint_values, shadows = solve(
                model, extract_duals=args.sensitivity
            )

        write_diet_csv(primals)
        plot_bounds(constraint_values, nutrition)
        print(f"objective = ${objective:.2f}/day")
        for food, amount in sorted(primals.items(), key=lambda kv: -kv[1]):
            print(f"  {food:30s} {round(amount * 100)} g")
        if shadows:
            print("\nBinding constraints (shadow prices):")
            for line in explain_shadow_prices(shadows, nutrition):
                print(f"  {line}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
