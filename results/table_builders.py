"""Pure table builders over the compact public CSV inputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "table_inputs"
SEED = ROOT / "results" / "seed_level"
OUTPUT = ROOT / "results" / "manuscript_tables"


TABLES: dict[str, tuple[Path, Callable[[dict[str, str]], bool]]] = {
    "baseline_grid": (INPUT / "baseline_grid.csv", lambda row: True),
    "real_reference": (INPUT / "real_reference.csv", lambda row: True),
    "stagewise": (INPUT / "stagewise_summary.csv", lambda row: True),
    "internal_tstr": (INPUT / "internal_tstr.csv", lambda row: True),
    "privacy": (INPUT / "privacy_attack_summary.csv", lambda row: True),
    "home_credit": (INPUT / "home_credit.csv", lambda row: True),
    "lambda_grid": (SEED / "lambda_grid_coarse.csv", lambda row: True),
    "breadth": (SEED / "breadth_stagewise.csv", lambda row: True),
    "repair_sensitivity": (INPUT / "repair_strategy_summary.csv", lambda row: True),
    "stabilisation_baselines": (
        SEED / "stabilisation_and_baselines.csv",
        lambda row: row.get("stage", "") in {
            "stabilisation_ablation",
            "direct_x0_coarse",
            "direct_x0_confirmation",
            "direct_x0_tuned",
            "constraint_baselines",
        },
    ),
}


def build_table(name: str, output_dir: Path = OUTPUT) -> Path:
    if name not in TABLES:
        raise KeyError(f"Unknown table {name!r}; choose from {', '.join(sorted(TABLES))}")
    source, predicate = TABLES[name]
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [row for row in reader if predicate(row)]
        fields = reader.fieldnames or []
    if not rows:
        raise RuntimeError(f"No rows selected for {name} from {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{name}.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return target


def build_all() -> list[Path]:
    return [build_table(name) for name in TABLES]
