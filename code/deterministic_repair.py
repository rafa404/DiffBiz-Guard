"""One-pass deterministic repair operators used in the evaluated pipeline."""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RepairRule:
    rule_id: str
    rule_type: str
    columns: tuple[str, ...]
    parameters: Mapping[str, Any]
    hard: bool = True
    execution_order: int = 0


def _coerce_rule(rule: RepairRule | Mapping[str, Any]) -> RepairRule:
    if isinstance(rule, RepairRule):
        return rule
    columns = rule.get("columns", rule.get("affected_columns", []))
    if isinstance(columns, str):
        columns = tuple(part for part in columns.split("|") if part)
    params = rule.get("parameters", rule.get("params", {}))
    if isinstance(params, str):
        params = json.loads(params or "{}")
    hard_value = rule.get("hard", rule.get("hard_or_soft", "hard"))
    hard = hard_value is True or str(hard_value).lower() in {"true", "hard", "1"}
    return RepairRule(
        rule_id=str(rule.get("rule_id", rule.get("name", "unnamed"))),
        rule_type=str(rule.get("rule_type", rule.get("kind", ""))),
        columns=tuple(columns), parameters=params, hard=hard,
        execution_order=int(rule.get("execution_order", 0)),
    )


def apply_repair_once(frame: Any, rules: Iterable[RepairRule | Mapping[str, Any]]) -> tuple[Any, dict[str, Any], Any]:
    """Apply each hard rule once in registered order and return an audit ledger."""

    import numpy as np
    import pandas as pd

    repaired = frame.copy()
    touched: set[Any] = set()
    ledger: list[dict[str, Any]] = []
    modified_cells = 0
    ordered = sorted((_coerce_rule(rule) for rule in rules), key=lambda item: item.execution_order)
    for rule in ordered:
        if not rule.hard:
            continue
        changed = pd.DataFrame(False, index=repaired.index, columns=list(rule.columns))
        kind, columns, params = rule.rule_type, rule.columns, rule.parameters
        if not columns or any(column not in repaired.columns for column in columns):
            continue
        before = repaired[list(columns)].copy()
        if kind in {"lower", "range", "integer"}:
            column = columns[0]
            values = pd.to_numeric(repaired[column], errors="coerce")
            allow_missing = bool(params.get("allow_missing", False))
            if kind == "lower":
                fixed = values.clip(lower=float(params["bound"]))
                fixed = fixed if allow_missing else fixed.fillna(float(params["bound"]))
            elif kind == "range":
                fixed = values.clip(float(params["lower"]), float(params["upper"]))
                fixed = fixed if allow_missing else fixed.fillna(float(params["lower"]))
            else:
                fixed = np.rint(values)
                fixed = fixed if allow_missing else fixed.fillna(0)
            repaired[column] = fixed
        elif kind == "less_equal":
            left, right = columns
            lval = pd.to_numeric(repaired[left], errors="coerce")
            rval = pd.to_numeric(repaired[right], errors="coerce")
            repaired[left] = np.minimum(lval.fillna(rval), rval)
        elif kind == "sum_lower":
            preferred = str(params.get("repair_column", columns[0]))
            total = sum(pd.to_numeric(repaired[c], errors="coerce").fillna(0) for c in columns)
            repaired[preferred] = pd.to_numeric(repaired[preferred], errors="coerce").fillna(0) + (float(params["bound"]) - total).clip(lower=0)
        elif kind == "missing_flag":
            value_column, flag_column = columns
            repaired[flag_column] = repaired[value_column].isna().astype(int)
        elif kind == "category_numeric_map":
            category_column, numeric_column = columns
            mapping = {str(k): float(v) for k, v in params["mapping"].items()}
            expected = repaired[category_column].astype("string").map(mapping)
            repaired.loc[expected.notna(), numeric_column] = expected[expected.notna()]
        elif kind == "approx_equal":
            left, right = columns
            repaired[left] = pd.to_numeric(repaired[right], errors="coerce")
        elif kind == "calendar_day":
            year_column, month_column, day_column = columns
            fixed = []
            for year, month, day in zip(repaired[year_column], repaired[month_column], repaired[day_column]):
                try:
                    yi, mi = int(round(year)), min(max(int(round(month)), 1), 12)
                    fixed.append(min(max(int(round(day)), 1), calendar.monthrange(yi, mi)[1]))
                except Exception:
                    fixed.append(1)
            repaired[day_column] = fixed
        after = repaired[list(columns)]
        for column in columns:
            changed[column] = before[column].astype("string").fillna("__MISSING__") != after[column].astype("string").fillna("__MISSING__")
        changed_rows = set(changed.index[changed.any(axis=1)].tolist())
        cells = int(changed.to_numpy().sum())
        touched.update(changed_rows)
        modified_cells += cells
        ledger.append({"rule_id": rule.rule_id, "modified_records": len(changed_rows), "modified_cells": cells})
    summary = {"modified_records": len(touched), "modified_cells": modified_cells, "modified_record_rate": len(touched) / max(len(frame), 1)}
    return repaired, summary, pd.DataFrame(ledger)


def apply_repair_in_order(frame: Any, rules: Sequence[RepairRule | Mapping[str, Any]], rule_ids: Sequence[str]) -> tuple[Any, dict[str, Any], Any]:
    indexed = {_coerce_rule(rule).rule_id: _coerce_rule(rule) for rule in rules}
    ordered = [indexed[rule_id] for rule_id in rule_ids if rule_id in indexed]
    return apply_repair_once(frame, ordered)


def repeat_until_stable(frame: Any, rules: Sequence[RepairRule | Mapping[str, Any]], max_passes: int = 20) -> tuple[Any, int, bool]:
    current = frame.copy()
    for pass_index in range(1, max_passes + 1):
        updated, _, _ = apply_repair_once(current, rules)
        if updated.equals(current):
            return updated, pass_index, True
        current = updated
    return current, max_passes, False

