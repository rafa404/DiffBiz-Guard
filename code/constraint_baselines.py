"""Study-matched constraint comparators.

These functions are not a full HARPOON reproduction and are not a full
reproduction of a published projected-diffusion system.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence


def direct_x0_loss_adjustment(mean: Any, variance: Any, gradient: Any, eta: float, rms_normalize: bool = True, gradient_clip: float | None = 10.0, variance_scale: bool = True) -> Any:
    import torch

    update = gradient
    if rms_normalize:
        update = update / update.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-7)
    if gradient_clip is not None:
        update = update.clamp(-float(gradient_clip), float(gradient_clip))
    scale = torch.sqrt(variance.clamp_min(1e-12)) if variance_scale else 1.0
    return mean - float(eta) * scale * update


def hard_rule_rejection_sampling(proposal_batches: Iterable[Any], hard_valid_mask: Callable[[Any], Any], release_size: int = 12_000) -> tuple[Any, int]:
    import pandas as pd

    accepted, proposals = [], 0
    for batch in proposal_batches:
        proposals += len(batch)
        accepted.append(batch.loc[hard_valid_mask(batch)])
        pool = pd.concat(accepted, ignore_index=True)
        if len(pool) >= release_size:
            return pool.iloc[:release_size].reset_index(drop=True), proposals
    raise RuntimeError("Proposal stream ended before the requested hard-valid release was filled")


def matched_linear_rule_projection(frame: Any, rules: Sequence[Mapping[str, Any]]) -> Any:
    """Project hard numerical bounds and pairwise linear orders once."""

    import numpy as np
    import pandas as pd

    projected = frame.copy()
    for rule in rules:
        if not bool(rule.get("hard", True)):
            continue
        kind, columns, params = rule["kind"], rule["columns"], rule.get("params", {})
        if kind == "lower":
            projected[columns[0]] = pd.to_numeric(projected[columns[0]], errors="coerce").clip(lower=float(params["bound"]))
        elif kind == "range":
            projected[columns[0]] = pd.to_numeric(projected[columns[0]], errors="coerce").clip(float(params["lower"]), float(params["upper"]))
        elif kind == "less_equal":
            left, right = columns
            projected[left] = np.minimum(pd.to_numeric(projected[left], errors="coerce"), pd.to_numeric(projected[right], errors="coerce"))
    return projected

