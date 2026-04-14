import csv
from pathlib import Path

from diet_opt.fuzzy import load_confirmed_mapping, match_food_names, write_unresolved


CANONICAL = ["Broccoli", "Carrots", "Cauliflower", "Celery", "Bell Peppers"]


def test_obvious_variant_resolves():
    matched, unresolved = match_food_names(["Broccoli florets"], CANONICAL)
    assert matched == {"Broccoli florets": "Broccoli"}
    assert unresolved == []


def test_multi_word_variant_resolves():
    variants = ["Carrots, raw whole", "Carrots, cooked whole"]
    matched, unresolved = match_food_names(variants, CANONICAL)
    assert matched["Carrots, raw whole"] == "Carrots"
    assert matched["Carrots, cooked whole"] == "Carrots"
    assert unresolved == []


def test_unrelated_variant_is_unresolved():
    matched, unresolved = match_food_names(["Dragonfruit"], CANONICAL, threshold=85.0)
    assert matched == {}
    assert len(unresolved) == 1
    assert unresolved[0].variant == "Dragonfruit"


def test_threshold_respected():
    matched, _ = match_food_names(["Brocolli florrets"], CANONICAL, threshold=99.0)
    assert matched == {}


def test_write_and_reload(tmp_path: Path):
    mapping_path = tmp_path / "mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "canonical", "confirmed_by", "confirmed_at"])
        w.writerow(["Broccoli heads", "Broccoli", "test", "2026-04-14"])
    loaded = load_confirmed_mapping(mapping_path)
    assert loaded == {"Broccoli heads": "Broccoli"}


def test_write_unresolved_roundtrip(tmp_path: Path):
    _, unresolved = match_food_names(["Dragonfruit"], CANONICAL)
    path = tmp_path / "unresolved.csv"
    write_unresolved(unresolved, path)
    assert path.read_text().startswith("variant,best_candidate,score")
