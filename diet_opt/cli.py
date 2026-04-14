"""Command-line entry point."""
from __future__ import annotations

import argparse
import sys

from .data import load_pipeline_inputs, validate_bounds
from .model import build_model
from .report import plot_bounds, write_diet_csv
from .solve import solve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diet-opt")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("optimize", help="solve the LP and emit optimum_diet.csv + optimized_diet.png")
    sub.add_parser("validate", help="check DRI bounds for lb > ub inversions")
    args = parser.parse_args(argv)

    food_info, food_matches, nutrition = load_pipeline_inputs()

    if args.cmd == "validate":
        violations = validate_bounds(nutrition)
        if violations:
            print("Invalid bounds:", *violations, sep="\n  ")
            return 1
        print(f"OK: {len(nutrition)} nutrients have valid bounds")
        return 0

    if args.cmd == "optimize":
        model, _vars, _cons = build_model(food_info, food_matches, nutrition)
        objective, primals, constraint_values = solve(model)
        write_diet_csv(primals)
        plot_bounds(constraint_values, nutrition)
        print(f"objective = ${objective:.2f}/day")
        for food, amount in sorted(primals.items(), key=lambda kv: -kv[1]):
            print(f"  {food:30s} {int(amount * 100)} g")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
