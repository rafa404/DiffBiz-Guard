"""Independent empirical privacy audits used in the revision protocol.

The membership audit is a classifier-based density-ratio membership audit
motivated by DOMIAS; it is not claimed to be the exact DOMIAS algorithm.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


ATTACK_CAP = 5_000
SEEDS = (13, 33, 73, 101, 202)


def stable_sample(frame: Any, n: int, seed: int) -> Any:
    return frame.sample(n=min(int(n), len(frame)), replace=False, random_state=int(seed)).reset_index(drop=True)


def _normalize(frame: Any, columns: Sequence[str], categorical: Sequence[str]) -> Any:
    import numpy as np
    import pandas as pd

    out = frame[list(columns)].copy()
    categorical_set = set(categorical)
    for column in columns:
        if column in categorical_set:
            out[column] = out[column].astype("string").fillna("__MISSING__").astype(str)
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def _fit_catboost(x: Any, y: Any, categorical: Sequence[str], seed: int, iterations: int) -> Any:
    from catboost import CatBoostClassifier

    categorical_indexes = [x.columns.get_loc(column) for column in categorical if column in x.columns]
    model = CatBoostClassifier(iterations=iterations, depth=6, learning_rate=0.08, loss_function="Logloss", eval_metric="AUC", random_seed=int(seed), thread_count=2, verbose=False, allow_writing_files=False, l2_leaf_reg=3.0)
    return model.fit(x, y, cat_features=categorical_indexes)


def density_ratio_membership_auc(train: Any, validation: Any, test: Any, released: Any, feature_columns: Sequence[str], categorical_columns: Sequence[str], seed: int, cap: int = ATTACK_CAP) -> float:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    n_reference = min(cap, len(validation), len(released))
    reference = stable_sample(validation, n_reference, seed + 101)
    synthetic = stable_sample(released, n_reference, seed + 103)
    x_reference = _normalize(reference, feature_columns, categorical_columns)
    x_synthetic = _normalize(synthetic, feature_columns, categorical_columns)
    x_ratio = pd.concat([x_reference, x_synthetic], ignore_index=True)
    y_ratio = np.concatenate([np.zeros(len(x_reference), dtype=int), np.ones(len(x_synthetic), dtype=int)])
    ratio_model = _fit_catboost(x_ratio, y_ratio, categorical_columns, seed + 107, 300)
    n_attack = min(cap, len(train), len(test))
    members = _normalize(stable_sample(train, n_attack, seed + 109), feature_columns, categorical_columns)
    nonmembers = _normalize(stable_sample(test, n_attack, seed + 113), feature_columns, categorical_columns)
    p_member = np.clip(ratio_model.predict_proba(members)[:, 1], 1e-6, 1 - 1e-6)
    p_nonmember = np.clip(ratio_model.predict_proba(nonmembers)[:, 1], 1e-6, 1 - 1e-6)
    scores = np.concatenate([np.log(p_member) - np.log1p(-p_member), np.log(p_nonmember) - np.log1p(-p_nonmember)])
    labels = np.concatenate([np.ones(n_attack, dtype=int), np.zeros(n_attack, dtype=int)])
    return float(roc_auc_score(labels, scores))


def attribute_inference_auc(train: Any, test: Any, released: Any, target: str, feature_columns: Sequence[str], categorical_columns: Sequence[str], seed: int, cap: int = ATTACK_CAP) -> tuple[float, float, float]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    labels = sorted(train[target].dropna().unique().tolist())
    if len(labels) != 2:
        raise ValueError("The final audit supports binary target attributes only")
    mapping = {labels[0]: 0, labels[1]: 1}
    x_synthetic = _normalize(released, feature_columns, categorical_columns)
    y_synthetic = released[target].map(mapping).to_numpy(dtype=int)
    attacker = _fit_catboost(x_synthetic, y_synthetic, categorical_columns, seed + 211, 250)
    n_attack = min(cap, len(train), len(test))
    members = stable_sample(train, n_attack, seed + 223)
    nonmembers = stable_sample(test, n_attack, seed + 227)
    member_auc = float(roc_auc_score(members[target].map(mapping), attacker.predict_proba(_normalize(members, feature_columns, categorical_columns))[:, 1]))
    nonmember_auc = float(roc_auc_score(nonmembers[target].map(mapping), attacker.predict_proba(_normalize(nonmembers, feature_columns, categorical_columns))[:, 1]))
    return member_auc, nonmember_auc, member_auc - nonmember_auc

