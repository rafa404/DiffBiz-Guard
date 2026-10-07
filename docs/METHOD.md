# Method

DiffBiz-Guard adds registered numerical business-rule guidance to the reverse trajectory of a frozen mixed-type tabular diffusion model. It does not retrain the generator for guidance, and it does not alter the categorical posterior.

## Supported differentiable rules

Let \(\rho(u)=\max(u,0)\). The public loss implementation supports:

- lower bounds, \(x_j \ge b\), with distance \(\rho(b-x_j)\);
- bounded ranges, \(l \le x_j \le h\), with distance \(\rho(l-x_j)+\rho(x_j-h)\);
- pairwise numerical order, \(x_a \le x_b\), with distance \(\rho(x_a-x_b)\).

Distances are divided by train-derived scales and squared before aggregation. Domain rules use weight `1.00`; train-derived plausibility rules use `0.20`. Integer-schema and categorical constraints remain outside the differentiable objective.

## Guided reverse step

At each guided numerical reverse step:

1. detach the categorical log state from the guidance path;
2. evaluate the frozen denoiser;
3. estimate the clean numerical state and clip it to `[-8, 8]`;
4. map the normalized clean estimate to data space using differentiable inverse-quantile interpolation;
5. evaluate the registered numerical-rule objective;
6. differentiate with respect to the current numerical diffusion state only;
7. normalize each row by gradient RMS with floor `1e-7`;
8. clip gradient elements to `[-10, 10]`;
9. adjust the numerical reverse mean using reverse-variance scaling and locked guidance strength `0.10`.

The implementation is in [`code/diffusion_guidance.py`](../code/diffusion_guidance.py), with losses in [`code/rule_losses.py`](../code/rule_losses.py).

## Raw → repair → filter → release

The study preserves stage attribution:

- **Raw:** uncorrected generator output, used to measure generator behavior.
- **Repair:** registered hard numerical operators are applied once in execution order.
- **Filter:** exact training-row hashes and a train-fitted nearest-neighbour threshold are used as empirical rejection criteria.
- **Release:** accepted proposals are selected under the locked size and requested-label policy.

The filter is an empirical release rule, not a formal privacy mechanism. The privacy analyses are separate attacks over completed releases.

## Repair operators

The public implementation includes lower-bound clipping, bounded-range clipping, integer rounding, pairwise-order correction, sum-lower correction, missingness flags, category-to-numeric maps, approximate equality, and calendar-day correction where registered. `repeat_until_stable(...)` is exposed for sensitivity analysis; the principal pipeline uses one registered pass.

## Baselines

The compact baseline module exposes:

- direct clean-state loss adjustment;
- hard-rule rejection sampling;
- matched one-pass projection for supported linear numerical rules.

These are matched study implementations. The direct clean-state comparator is not presented as a full HARPOON reproduction, and the projection helper is not presented as a full reproduction of a published projected-diffusion system.

## Public API map

| Function | Module | Role |
|---|---|---|
| `weighted_registered_rule_loss` | `rule_losses.py` | Batch-mean weighted numerical-rule objective. |
| `guided_numerical_reverse_mean` | `diffusion_guidance.py` | Adjust the current numerical reverse mean. |
| `apply_repair_once` | `deterministic_repair.py` | Apply hard repairs in registered order. |
| `fit_release_filter` / `apply_release_filter` | `release_filter.py` | Fit and apply the empirical filter. |
| `requested_labels` / `paired_refill` | `release_construction.py` | Construct release policies and paired refills. |
| `rule_violation_rates` | `evaluation_metrics.py` | Summarize any/hard/soft violations. |
| `density_ratio_membership_auc` | `privacy_audits.py` | Classifier-based density-ratio membership audit motivated by DOMIAS. |
| `attribute_inference_auc` | `privacy_audits.py` | Binary-target attribute-inference audit. |
| `hard_rule_rejection_sampling` | `constraint_baselines.py` | Matched hard-valid rejection baseline. |
| `matched_linear_rule_projection` | `constraint_baselines.py` | Matched linear-rule projection baseline. |
