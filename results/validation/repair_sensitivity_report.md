# Internal repair-strategy sensitivity

## Scope

- CPU-only analysis of saved guided pre-repair tables.
- Three internal datasets and five frozen seeds per dataset: 15 task groups.
- No generator training, sampling, filtering, release reconstruction, or real-test access.
- Imported inputs were read-only.

## Strategies

1. No repair.
2. Original fixed registry order, one pass.
3. Original fixed registry order, repeated until stable, maximum five passes.
4. Reverse hard-rule order, one pass.
5. Bounds-first hard-rule order, one pass.
6. Relation-first hard-rule order, one pass.

## Audit summary

- Repeat-until-stable exactly matched original one-pass in 15/15 tasks.
- None of 45 alternative-order evaluations differed from original one-pass.
- None of 45 alternative-order evaluations retained a hard-rule violation.
- Every original one-pass result was stable under an additional original-order pass.

These findings are exploratory and limited to the evaluated saved tables. They do not establish global convergence or universal order independence.
