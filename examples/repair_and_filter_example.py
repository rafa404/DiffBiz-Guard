"""Demonstrate one-pass repair followed by the empirical release filter.

The small frames demonstrate interfaces only; they are not study data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.deterministic_repair import RepairRule, apply_repair_once
from code.release_filter import apply_release_filter, fit_release_filter


train = pd.DataFrame(
    {
        "balance": [100.0, 250.0, 400.0, 700.0],
        "term_months": [6.0, 12.0, 18.0, 24.0],
    }
)
candidates = pd.DataFrame(
    {
        "balance": [-5.0, 250.0, 520.0],
        "term_months": [7.4, 12.0, 20.2],
    }
)

rules = [
    RepairRule("balance_nonnegative", "lower", ("balance",), {"bound": 0.0}, execution_order=1),
    RepairRule("term_integer", "integer", ("term_months",), {}, execution_order=2),
]

repaired, repair_summary, repair_ledger = apply_repair_once(candidates, rules)

mean = train.mean()
scale = train.std(ddof=0).replace(0.0, 1.0)


def train_fitted_transform(frame):
    return ((frame.astype(float) - mean) / scale).to_numpy(dtype=np.float64)


fitted = fit_release_filter(
    train,
    feature_columns=["balance", "term_months"],
    train_fitted_transform=train_fitted_transform,
    reference_cap=12_000,
)
released, filter_summary, decisions = apply_release_filter(repaired, fitted)

print("repair summary:", repair_summary)
print("repair ledger:\n", repair_ledger)
print("filter summary:", filter_summary)
print("decisions:\n", decisions)
print("accepted rows:\n", released)
