"""Compact validity, repair, fidelity, utility, and calibration metrics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def rule_violation_rates(rule_masks: Mapping[str, Any], hard_rules: Sequence[str]) -> dict[str, float]:
    import numpy as np

    if not rule_masks:
        return {"any_rule_violation_rate": 0.0, "hard_rule_violation_rate": 0.0, "soft_rule_violation_rate": 0.0}
    names = list(rule_masks)
    matrix = np.column_stack([np.asarray(rule_masks[name], dtype=bool) for name in names])
    hard_index = [index for index, name in enumerate(names) if name in set(hard_rules)]
    soft_index = [index for index, name in enumerate(names) if name not in set(hard_rules)]
    return {
        "any_rule_violation_rate": float(matrix.any(axis=1).mean()),
        "hard_rule_violation_rate": float(matrix[:, hard_index].any(axis=1).mean()) if hard_index else 0.0,
        "soft_rule_violation_rate": float(matrix[:, soft_index].any(axis=1).mean()) if soft_index else 0.0,
    }


def repair_burden(before: Any, after: Any) -> dict[str, float]:
    changed = before.astype("string").fillna("__MISSING__") != after.astype("string").fillna("__MISSING__")
    return {"modified_rows": int(changed.any(axis=1).sum()), "modified_cells": int(changed.to_numpy().sum()), "modified_row_rate": float(changed.any(axis=1).mean())}


def normalized_correction_magnitude(before: Any, after: Any, train_scale: Mapping[str, float]) -> dict[str, float]:
    import numpy as np
    import pandas as pd

    values = []
    for column in before.columns.intersection(after.columns):
        left = pd.to_numeric(before[column], errors="coerce")
        right = pd.to_numeric(after[column], errors="coerce")
        delta = (right - left).abs() / max(float(train_scale.get(column, 1.0)), 1e-12)
        values.extend(delta[np.isfinite(delta)].tolist())
    return {"mean": float(np.mean(values)) if values else 0.0, "p95": float(np.quantile(values, 0.95)) if values else 0.0, "max": float(np.max(values)) if values else 0.0}


def numerical_ks(real: Any, synthetic: Any, numerical_columns: Sequence[str]) -> float:
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp

    values = []
    for column in numerical_columns:
        left = pd.to_numeric(real[column], errors="coerce").dropna()
        right = pd.to_numeric(synthetic[column], errors="coerce").dropna()
        if len(left) and len(right):
            values.append(float(ks_2samp(left, right).statistic))
    return float(np.mean(values)) if values else float("nan")


def categorical_tvd(real: Any, synthetic: Any, categorical_columns: Sequence[str]) -> float:
    import numpy as np

    values = []
    for column in categorical_columns:
        p = real[column].astype("string").fillna("__MISSING__").value_counts(normalize=True)
        q = synthetic[column].astype("string").fillna("__MISSING__").value_counts(normalize=True)
        support = p.index.union(q.index)
        values.append(0.5 * float(np.abs(p.reindex(support, fill_value=0) - q.reindex(support, fill_value=0)).sum()))
    return float(np.mean(values)) if values else float("nan")


def numerical_correlation_error(real: Any, synthetic: Any, numerical_columns: Sequence[str]) -> float:
    import numpy as np

    if len(numerical_columns) < 2:
        return float("nan")
    left = real[list(numerical_columns)].corr().fillna(0).to_numpy()
    right = synthetic[list(numerical_columns)].corr().fillna(0).to_numpy()
    return float(np.abs(left - right).mean())


def target_prior_error(real_target: Any, synthetic_target: Any) -> float:
    return abs(float(real_target.mean()) - float(synthetic_target.mean()))


def brier_score(y_true: Any, probability: Any) -> float:
    import numpy as np

    return float(np.mean((np.asarray(probability) - np.asarray(y_true)) ** 2))


def expected_calibration_error(y_true: Any, probability: Any, bins: int = 10) -> float:
    import numpy as np

    y, p = np.asarray(y_true), np.asarray(probability)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & ((p <= edges[index + 1]) if index == bins - 1 else (p < edges[index + 1]))
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def c2st_auc(real_features: Any, synthetic_features: Any, seed: int = 2026) -> float:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    x = np.concatenate([real_features, synthetic_features], axis=0)
    y = np.concatenate([np.zeros(len(real_features)), np.ones(len(synthetic_features))])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
    model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(x_train, y_train)
    return float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))


def release_yield(released_rows: int, proposal_rows: int) -> float:
    return float(released_rows) / max(int(proposal_rows), 1)

