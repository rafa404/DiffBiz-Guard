# Release validation

- Date/time: `2026-10-20T16:49:31+06:00`
- Verification Status: **ANALYZED**
- Public files: 105
- Overall validation: **PASS**

`ANALYZED` means the public artifacts and deterministic table reconstruction were checked without rerunning costly stochastic training, generation, or attack computations.

## Repository tree

```text
DiffBiz-Guard/
  .gitignore
  artifact_manifest.json
  CITATION.cff
  code/ (9 files)
  configs/ (4 files)
  docs/ (6 files)
  examples/ (3 files)
  images/ (6 files)
  LICENSE
  README.md
  registries/ (7 files)
  requirements.txt
  results/ (57 files)
    manuscript_tables/ (10 files)
    seed_level/ (13 files)
    table_inputs/ (14 files)
    validation/ (18 files)
  splits/ (7 files)
```

## Checks

| Check | Status | Evidence |
|---|---|---|
| Python syntax | PASS | All public Python files parsed. |
| Public imports | PASS | Reusable modules and table tools imported with dependency-lazy boundaries. |
| JSON and CSV | PASS | All JSON parsed; every CSV has at least one data row. |
| Repository structure | PASS | Code, settings, registries, splits, examples, results, images, and documentation are present. |
| README links and images | PASS | Relative links resolve and every referenced local image exists. |
| Visual assets | PASS | Five PNGs open successfully; the original SVG banner is safe and below 5 MB. |
| Registry reconciliation | PASS | Seven complete registries match row, form, hard/soft, provenance, and split-context totals. |
| Split privacy | PASS | Seven manifests contain metadata and hashes but no record arrays. |
| Table reconstruction | PASS | 10 non-empty manuscript tables were rebuilt from public inputs. |
| Dataset links | PASS | All seven final provider URLs are present, including the corrected DataCo source. |
| Citation metadata | PASS | Author/title metadata are present with no DOI or publication date. |
| Local paths | PASS | No machine-specific absolute path or cloud-runtime mount path. |
| Secrets and placeholders | PASS | No credential assignment pattern or unresolved placeholder. |
| Public terminology | PASS | Public paths, Markdown headings, and Python source avoid development-history labels. |
| Raw/private artifacts | PASS | No raw dataset, checkpoint, archive, notebook, or figure-generation script. |
| Zero-byte files | PASS | No empty public artifact. |
| Checksum manifest | PASS | Generated after this record and verified for every covered file; the two manifest files are self-reference exempt. |

## Registry detail

- bank_marketing.csv: PASS (47 rows)
- bank_account_fraud.csv: PASS (39 rows)
- large_customer_churn.csv: PASS (44 rows)
- airline.csv: PASS (57 rows)
- hotel.csv: PASS (41 rows)
- dataco.csv: PASS (59 rows)
- home_credit.csv: PASS (3 contexts x 167 rows)

## Boundaries

- Dependency-heavy examples were syntax-checked; PyTorch, SciPy, scikit-learn, and CatBoost were not installed in the lightweight validation runtime.
- Breadth source and split hashes were unavailable in the preserved lineage and remain explicit null values.
- The privacy and repair-sensitivity outputs were checked against their final manifests and deterministic aggregates; the underlying experiments were not rerun.
- No animated GIF was added because the existing scientific pipeline plus the compact original SVG banner communicate the workflow without redundant motion or repository weight.
- The software code is released under the MIT License; third-party dataset and provider terms remain separate.

## Decision

**PASS.** The public working tree is structurally complete, scientifically scoped, reproducible at the table level, and clear of raw data, private paths, credentials, and development-history presentation.
