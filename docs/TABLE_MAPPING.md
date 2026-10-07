# Manuscript table mapping

Run commands from the repository root. Outputs are written to `results/manuscript_tables/`.

| Public table | Source data | Command |
|---|---|---|
| Baseline grid | `results/table_inputs/baseline_grid.csv` | `python results/reproduce_tables.py --table baseline_grid` |
| Real reference validity | `results/table_inputs/real_reference.csv` | `python results/reproduce_tables.py --table real_reference` |
| Stage-wise attribution | `results/table_inputs/stagewise_summary.csv` | `python results/reproduce_tables.py --table stagewise` |
| Internal TSTR | `results/table_inputs/internal_tstr.csv` | `python results/reproduce_tables.py --table internal_tstr` |
| Independent privacy attacks | `results/table_inputs/privacy_attack_summary.csv` | `python results/reproduce_tables.py --table privacy` |
| Home Credit effects | `results/table_inputs/home_credit.csv` | `python results/reproduce_tables.py --table home_credit` |
| Guidance grid | `results/seed_level/lambda_grid_coarse.csv` | `python results/reproduce_tables.py --table lambda_grid` |
| Breadth effects | `results/seed_level/breadth_stagewise.csv` | `python results/reproduce_tables.py --table breadth` |
| Repair sensitivity | `results/table_inputs/repair_strategy_summary.csv` | `python results/reproduce_tables.py --table repair_sensitivity` |
| Stabilisation and constraint baselines | `results/seed_level/stabilisation_and_baselines.csv` | `python results/reproduce_tables.py --table stabilisation_baselines` |

`python results/reproduce_tables.py --all` reconstructs all ten tables.
