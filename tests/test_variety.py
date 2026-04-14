from dataclasses import dataclass
from pathlib import Path

import pytest

from diet_opt.variety import DEFAULT_CAPS_G, apply_caps, load_food_groups


@dataclass
class FakeVar:
    lb: float = 0.0
    ub: float = 5.0


def test_load_food_groups_inverts():
    mapping = load_food_groups()
    assert mapping["Spinach"] == "leafy_green"
    assert mapping["Pinto Beans"] == "legume"
    assert mapping["Flax seeds"] == "nut_seed"


def test_load_food_groups_rejects_duplicates(tmp_path: Path):
    dup = tmp_path / "dup.yaml"
    dup.write_text("a:\n  - X\nb:\n  - X\n")
    with pytest.raises(ValueError, match="X"):
        load_food_groups(dup)


def test_apply_caps_shrinks_ub():
    vars_ = {"Carrots": FakeVar(ub=5.0), "Almonds": FakeVar(ub=5.0)}
    mapping = {"Carrots": "root_veg", "Almonds": "nut_seed"}
    apply_caps(vars_, mapping)
    assert vars_["Carrots"].ub == 3.5   # 350g cap
    assert vars_["Almonds"].ub == 0.6   # 60g cap


def test_apply_caps_never_raises_existing_ub():
    vars_ = {"Carrots": FakeVar(ub=1.0)}
    apply_caps(vars_, {"Carrots": "root_veg"})
    assert vars_["Carrots"].ub == 1.0   # already tighter than 3.5


def test_apply_caps_unknown_food_uses_default():
    vars_ = {"Martian_Weed": FakeVar(ub=5.0)}
    apply_caps(vars_, {})  # no mapping
    assert vars_["Martian_Weed"].ub == DEFAULT_CAPS_G["_default"] / 100.0


def test_apply_caps_handles_underscored_keys():
    vars_ = {"Pinto_Beans": FakeVar(ub=5.0)}
    apply_caps(vars_, {"Pinto Beans": "legume"})
    assert vars_["Pinto_Beans"].ub == 2.5   # 250g legume cap


def test_apply_caps_returns_group_counts():
    vars_ = {"Carrots": FakeVar(), "Beets": FakeVar(), "Almonds": FakeVar()}
    mapping = {"Carrots": "root_veg", "Beets": "root_veg", "Almonds": "nut_seed"}
    counts = apply_caps(vars_, mapping)
    assert counts["root_veg"] == 2
    assert counts["nut_seed"] == 1
