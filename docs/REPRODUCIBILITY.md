# Reproducibility

## Scope

This compact release supports two levels of reproducibility:

1. **Deterministic table reconstruction** from the included compact result evidence.
2. **Method inspection and reuse** through clean Python modules for guidance, rule losses, repair, filtering, release construction, metrics, privacy attacks, and matched baselines.

It does not claim end-to-end training reproducibility from raw data. Historical cloud orchestration, model checkpoints, generated samples, and provider data are intentionally excluded.

## Environment

The reported experiments used:

- Python 3.12.13
- PyTorch 2.10.0
- CUDA 12.8
- two Tesla T4 GPUs

CUDA-specific wheels are not forced by `requirements.txt`. Exact package versions other than PyTorch were not recoverable from the final environment evidence, so unverified pins were not invented.

## Reconstruct all public tables

From the repository root:

```bash
python results/reproduce_tables.py --all
```

This creates ten CSVs under `results/manuscript_tables/`. No raw dataset or model checkpoint is required. The builders read public CSV inputs, apply only documented row selection, and preserve column order.

## Raw-data use

Download data directly from the sources in [`DATASETS.md`](DATASETS.md) and keep it under a local ignored directory such as `data/`. The compact repository does not expose a universal provider-download command or a turnkey historical training pipeline. Use the split manifests to verify source identity, preparation, counts, seeds, and hashes where available.

## Validation status

The public artifacts, internal checksums, table reconstruction, registry totals, split privacy, path/secret scans, and repository links are validated. The costly original training, generation, and attack computations were not rerun during repository assembly; their supplied final outputs were checked for completeness, internal consistency, and recorded hashes.

The release validation record uses `Verification Status: ANALYZED` for this reason. See [`RELEASE_VALIDATION.md`](RELEASE_VALIDATION.md).

## Interpretation boundary

Table reconstruction confirms deterministic transformation of the included evidence. It does not independently replicate the stochastic experiments. Privacy analyses remain empirical and attack-specific, and repair-order sensitivity remains limited to the evaluated saved tables.
