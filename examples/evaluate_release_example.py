"""Compute compact validity, repair, and release-yield summaries."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.evaluation_metrics import release_yield, repair_burden, rule_violation_rates


before = pd.DataFrame({"age": [-1.0, 35.0, 42.0], "tenure": [2.4, 5.0, 8.0]})
after = pd.DataFrame({"age": [0.0, 35.0, 42.0], "tenure": [2.0, 5.0, 8.0]})

rule_masks = {
    "age_nonnegative": [True, False, False],
    "tenure_plausibility": [False, False, False],
}

print("violation rates:", rule_violation_rates(rule_masks, hard_rules=["age_nonnegative"]))
print("repair burden:", repair_burden(before, after))
print("release yield:", release_yield(released_rows=12_000, proposal_rows=16_201))
