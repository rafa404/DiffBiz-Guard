"""Small executable demonstration of the numerical guidance interface.

The tensors are illustrative only; they are not manuscript data or results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.diffusion_guidance import guided_numerical_reverse_mean
from code.rule_losses import lower_bound_hinge


class FrozenToyDenoiser(torch.nn.Module):
    def forward(self, model_input, time, labels):
        del time, labels
        return torch.zeros_like(model_input)


denoiser = FrozenToyDenoiser().eval()
for parameter in denoiser.parameters():
    parameter.requires_grad_(False)

current = torch.tensor([[-0.8, 0.3], [0.4, -0.2]], dtype=torch.float32)
categorical_log_state = torch.zeros((2, 1), dtype=torch.float32)
time = torch.zeros(2, dtype=torch.long)
labels = torch.tensor([0, 1], dtype=torch.long)
variance = torch.full_like(current, 0.02)


def predict_x0(state, _time, predicted_noise):
    return state - predicted_noise


def posterior_mean(clean, state, _time):
    return 0.3 * clean + 0.7 * state


def rule_loss(clean):
    distance = lower_bound_hinge(clean[:, 0], bound=0.0, scale=1.0)
    return distance.square().mean()


guided_mean, clean_estimate, categorical_output = guided_numerical_reverse_mean(
    current_numerical_state=current,
    categorical_log_state=categorical_log_state,
    time=time,
    labels=labels,
    denoiser=denoiser,
    predict_x0=predict_x0,
    posterior_mean=posterior_mean,
    posterior_variance=variance,
    rule_loss=rule_loss,
    numerical_width=2,
)

print("guided numerical mean:\n", guided_mean)
print("clean estimate:\n", clean_estimate)
print("categorical output shape:", tuple(categorical_output.shape))
