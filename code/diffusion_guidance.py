"""Frozen-backbone inference-time numerical guidance.

The gradient modifies only the current numerical reverse trajectory. Generator
parameters remain frozen, and the categorical posterior is detached from the
guidance path and otherwise unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Callable


def _torch():
    import torch

    return torch


def differentiable_inverse_quantile(z: Any, quantiles: Any, references: Any) -> Any:
    """Invert train-fitted quantiles with piecewise-linear PyTorch interpolation."""

    torch = _torch()
    probability = (0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))).clamp(0.0, 1.0)
    outputs = []
    references = references.to(device=z.device, dtype=z.dtype)
    quantiles = quantiles.to(device=z.device, dtype=z.dtype)
    for column in range(z.shape[1]):
        p = probability[:, column].contiguous()
        upper = torch.searchsorted(references, p, right=False).clamp(1, len(references) - 1)
        lower = upper - 1
        r0, r1 = references[lower], references[upper]
        q0, q1 = quantiles[lower, column], quantiles[upper, column]
        weight = (p - r0) / (r1 - r0).clamp_min(1e-7)
        outputs.append(q0 + weight * (q1 - q0))
    return torch.stack(outputs, dim=1)


def estimate_clean_state(predict_x0: Callable[[Any, Any, Any], Any], state: Any, time: Any, epsilon: Any) -> Any:
    return predict_x0(state, time, epsilon).clamp(-8.0, 8.0)


def normalize_guidance_gradient(gradient: Any, rms_floor: float = 1e-7, clip: float = 10.0) -> Any:
    rms = gradient.square().mean(dim=1, keepdim=True).sqrt().clamp_min(float(rms_floor))
    return (gradient / rms).clamp(-float(clip), float(clip))


def guided_numerical_reverse_mean(
    current_numerical_state: Any,
    categorical_log_state: Any,
    time: Any,
    labels: Any,
    denoiser: Callable[[Any, Any, Any], Any],
    predict_x0: Callable[[Any, Any, Any], Any],
    posterior_mean: Callable[[Any, Any, Any], Any],
    posterior_variance: Any,
    rule_loss: Callable[[Any], Any],
    numerical_width: int,
    guidance_strength: float = 0.10,
    rms_floor: float = 1e-7,
    gradient_clip: float = 10.0,
) -> tuple[Any, Any, Any]:
    """Return the guided numerical mean, clipped clean estimate, and categorical logits."""

    torch = _torch()
    state = current_numerical_state.detach().requires_grad_(True)
    model_input = torch.cat([state, categorical_log_state.detach()], dim=1).float()
    model_output = denoiser(model_input, time, labels).float()
    numerical_output = model_output[:, :numerical_width]
    categorical_output = model_output[:, numerical_width:].detach()
    clean = estimate_clean_state(predict_x0, state, time, numerical_output)
    loss = rule_loss(clean)
    gradient = torch.autograd.grad(loss, state, retain_graph=False, create_graph=False)[0].detach()
    clean = clean.detach()
    variance = posterior_variance.clamp_min(1e-12)
    mean = posterior_mean(clean, state.detach(), time)
    normalized = normalize_guidance_gradient(gradient, rms_floor=rms_floor, clip=gradient_clip)
    mean = mean - float(guidance_strength) * torch.sqrt(variance) * normalized
    return mean.detach(), clean, categorical_output

