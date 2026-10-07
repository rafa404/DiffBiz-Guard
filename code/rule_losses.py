"""Differentiable numerical rule losses used by DiffBiz-Guard.

Only numerical lower bounds, bounded ranges, and supported pairwise order
relations enter this objective. Integer and categorical constraints remain
outside the differentiable path and are handled by auditing/repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class RegisteredNumericalRule:
    rule_id: str
    rule_type: str
    columns: tuple[str, ...]
    parameters: Mapping[str, float] = field(default_factory=dict)
    provenance_category: str = "domain"
    differentiable: bool = True
    applicability: Any | None = None


def positive_part(value: Any) -> Any:
    """Return rho(u) = max(u, 0)."""

    return _torch().relu(value)


def lower_bound_hinge(values: Any, bound: float, scale: float = 1.0) -> Any:
    return positive_part(float(bound) - values) / max(float(scale), 1e-12)


def bounded_range_hinge(values: Any, lower: float, upper: float, scale: float = 1.0) -> Any:
    return (
        positive_part(float(lower) - values)
        + positive_part(values - float(upper))
    ) / max(float(scale), 1e-12)


def pairwise_order_hinge(left: Any, right: Any, scale: float = 1.0) -> Any:
    return positive_part(left - right) / max(float(scale), 1e-12)


def train_derived_scale(lower: float, upper: float, fallback: float = 1.0) -> float:
    width = float(upper) - float(lower)
    return width if width > 1e-6 else max(float(fallback), 1e-6)


def _masked_mean_squared(distance: Any, mask: Any | None) -> Any:
    if mask is None:
        return distance.square().mean()
    selected = distance[mask]
    if selected.numel() == 0:
        return distance.sum() * 0.0
    return selected.square().mean()


def weighted_registered_rule_loss(
    data_space: Any,
    rules: Sequence[RegisteredNumericalRule],
    column_index: Mapping[str, int],
    train_scales: Mapping[str, float],
    domain_weight: float = 1.0,
    plausibility_weight: float = 0.20,
) -> Any:
    """Compute the batch-mean weighted squared-hinge objective.

    This mirrors the final implementation: domain rules receive weight 1.00,
    train-derived plausibility rules receive 0.20, and unsupported rule forms
    are excluded rather than silently approximated.
    """

    losses = []
    for rule in rules:
        if not rule.differentiable:
            continue
        if any(column not in column_index for column in rule.columns):
            continue
        weight = plausibility_weight if rule.provenance_category == "plausibility" else domain_weight
        mask = rule.applicability
        if rule.rule_type == "lower":
            column = rule.columns[0]
            distance = lower_bound_hinge(
                data_space[:, column_index[column]],
                rule.parameters["bound"],
                train_scales.get(column, 1.0),
            )
        elif rule.rule_type == "range":
            column = rule.columns[0]
            distance = bounded_range_hinge(
                data_space[:, column_index[column]],
                rule.parameters["lower"],
                rule.parameters["upper"],
                train_scales.get(column, 1.0),
            )
        elif rule.rule_type == "less_equal":
            left, right = rule.columns
            scale = max(train_scales.get(left, 1.0) + train_scales.get(right, 1.0), 1e-6)
            distance = pairwise_order_hinge(
                data_space[:, column_index[left]], data_space[:, column_index[right]], scale
            )
        else:
            continue
        losses.append(float(weight) * _masked_mean_squared(distance, mask))
    if not losses:
        return data_space.sum() * 0.0
    return _torch().stack(losses).sum()


def batch_mean_rule_objective(*args: Any, **kwargs: Any) -> Any:
    return weighted_registered_rule_loss(*args, **kwargs)

