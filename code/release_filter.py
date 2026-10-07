"""Empirical exact-match and nearest-neighbour release filtering.

This is an empirical release filter, not a formal privacy mechanism and not a
differential-privacy guarantee.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence


def normalized_row_hashes(frame: Any, numerical_decimals: int = 8) -> Any:
    import pandas as pd

    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_numeric_dtype(normalized[column]):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round(numerical_decimals)
        else:
            normalized[column] = normalized[column].astype("string").fillna("__NA__")
    serialized = normalized.astype(str).agg("|".join, axis=1)
    return serialized.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


@dataclass
class FittedReleaseFilter:
    feature_columns: tuple[str, ...]
    transform: Callable[[Any], Any]
    nearest_neighbors: Any
    threshold: float
    train_hashes: set[str]


def fit_release_filter(
    train: Any,
    feature_columns: Sequence[str],
    train_fitted_transform: Callable[[Any], Any],
    reference_cap: int = 12_000,
    threshold_quantile: float = 0.01,
    seed: int = 2026,
) -> FittedReleaseFilter:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    reference = train
    if len(reference) > reference_cap:
        reference = reference.iloc[rng.choice(len(reference), reference_cap, replace=False)]
    reference = reference.reset_index(drop=True)
    matrix = train_fitted_transform(reference[list(feature_columns)])
    model = NearestNeighbors(n_neighbors=min(2, len(matrix)), metric="euclidean", n_jobs=-1).fit(matrix)
    distances = model.kneighbors(matrix)[0]
    within_train = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
    positive = within_train[within_train > 1e-12]
    threshold = float(np.quantile(positive, threshold_quantile)) if len(positive) else 0.0
    hashes = set(normalized_row_hashes(train[list(feature_columns)], 8).tolist())
    return FittedReleaseFilter(tuple(feature_columns), train_fitted_transform, model, threshold, hashes)


def apply_release_filter(frame: Any, fitted: FittedReleaseFilter) -> tuple[Any, dict[str, Any], Any]:
    import numpy as np
    import pandas as pd

    features = frame[list(fitted.feature_columns)]
    distances = fitted.nearest_neighbors.kneighbors(fitted.transform(features), n_neighbors=1)[0][:, 0]
    exact = normalized_row_hashes(features, 8).isin(fitted.train_hashes).to_numpy()
    below = distances <= fitted.threshold
    nearest_only = below & ~exact
    keep = ~(exact | below)
    ledger = pd.DataFrame({"distance_to_closest_train": distances, "exact": exact, "nearest_neighbor_only": nearest_only, "kept": keep})
    summary = {
        "threshold": fitted.threshold,
        "exact_rejections": int(exact.sum()),
        "nearest_neighbor_only_rejections": int(nearest_only.sum()),
        "kept_rows": int(keep.sum()),
        "kept_rate": float(np.mean(keep)),
    }
    return frame.loc[keep].reset_index(drop=True), summary, ledger

