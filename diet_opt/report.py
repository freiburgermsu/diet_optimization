"""Emit diet CSV and the Figure-1 bounds plot."""
from __future__ import annotations

from math import inf
from pathlib import Path

from pandas import DataFrame


def write_diet_csv(primals: dict, path: str | Path = "optimum_diet.csv") -> None:
    grams = {"food": list(primals.keys()), "grams": [int(v * 100) for v in primals.values()]}
    DataFrame(grams).to_csv(path)


def plot_bounds(constraint_values: dict, nutrition: dict, path: str | Path = "optimized_diet.png") -> None:
    from matplotlib import pyplot

    fig = pyplot.figure(figsize=(10, 20))
    for y, (name, content) in enumerate(reversed(constraint_values.items())):
        start, mid, end = content["lb"], content["val"], content["ub"]
        end = 100000 if end == inf else end
        pyplot.plot([start, mid], [y, y], "k-", linewidth=2)
        pyplot.plot([mid, end], [y, y], "k-", linewidth=2)
        pyplot.scatter([mid], [y], color="red", zorder=3)
        if name == "volume":
            label, unit = name, "[cups]"
        else:
            key = name.replace("_", " ")
            label, unit = name, f"[{nutrition[key]['units']}]"
        pyplot.text(mid, y + 0.3, f"{label}: {round(mid)} {unit}", fontsize=12, ha="center", va="bottom")
    pyplot.xscale("log")
    pyplot.xlabel("Nutrient requirements")
    pyplot.gca().get_yaxis().set_visible(False)
    pyplot.title("Optimized nutrient requirements for each nutrient")
    pyplot.tight_layout()
    pyplot.savefig(path)
