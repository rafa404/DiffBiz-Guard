"""Locked label policies, proposal sizing, and accepted-row construction."""

from __future__ import annotations

import math
from typing import Any, Iterable


def locked_proposal_count(release_size: int = 12_000, oversample_factor: float = 1.35) -> int:
    """Mirror the locked computation; CPython floating arithmetic yields 16,201."""

    return int(math.ceil(int(release_size) * float(oversample_factor)))


def requested_labels(train_labels: Any, n_rows: int, policy: str, seed: int) -> Any:
    import numpy as np

    rng = np.random.default_rng(seed)
    values = np.asarray(train_labels)
    classes, counts = np.unique(values, return_counts=True)
    if policy == "balanced":
        base = n_rows // len(classes)
        labels = np.concatenate([np.repeat(label, base) for label in classes])
        remainder = n_rows - len(labels)
        if remainder:
            labels = np.concatenate([labels, classes[:remainder]])
    elif policy == "preserve":
        probabilities = counts / counts.sum()
        labels = rng.choice(classes, size=n_rows, replace=True, p=probabilities)
    else:
        raise ValueError("policy must be 'balanced' or 'preserve'")
    rng.shuffle(labels)
    return labels


def select_accepted_rows(candidates: Any, release_size: int = 12_000) -> Any:
    if len(candidates) < release_size:
        raise ValueError(f"Accepted pool has {len(candidates)} rows; {release_size} required")
    return candidates.iloc[:release_size].reset_index(drop=True)


def paired_refill(rounds: Iterable[tuple[Any, Any]], release_size: int = 12_000, proposal_ceiling: int = 48_000) -> tuple[Any, Any]:
    """Accumulate paired unguided/guided accepted pools without unpairing rounds."""

    import pandas as pd

    left_parts, right_parts, proposed = [], [], 0
    for left, right in rounds:
        proposed += max(len(left), len(right))
        if proposed > proposal_ceiling:
            break
        left_parts.append(left)
        right_parts.append(right)
        left_pool = pd.concat(left_parts, ignore_index=True)
        right_pool = pd.concat(right_parts, ignore_index=True)
        if len(left_pool) >= release_size and len(right_pool) >= release_size:
            return select_accepted_rows(left_pool, release_size), select_accepted_rows(right_pool, release_size)
    raise RuntimeError("Paired proposal ceiling reached before both releases were complete")

