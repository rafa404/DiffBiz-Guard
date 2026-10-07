# Independent privacy-attack audit

- No generator training or synthetic resampling was performed.
- Existing frozen train/validation/test splits and completed released tables were reused.
- Attack cap: 5,000 real records per member, non-member, or reference sample.
- The density-ratio membership attack does not use nearest-neighbour distance or the release-filter threshold.
- Attribute inference treats the binary dataset target as the withheld sensitive attribute and reports member and non-member AUC.

## Public outputs

- `results/seed_level/privacy_attack_seed_level.csv`
- `results/table_inputs/privacy_attack_summary.csv`
- `results/validation/privacy_attack_table.tex`
