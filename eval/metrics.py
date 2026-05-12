"""Evaluation metrics for text-to-SQL.

Primary metric: **execution accuracy** — a prediction is correct iff its result
set on a fresh DB equals the gold query's result set (compared as sorted lists
of tuples). Secondary: exact-match (after sqlglot normalization where possible)
and valid-SQL rate.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import sqlglot
from sqlglot.errors import ParseError

log = logging.getLogger(__name__)


def _execute(con: sqlite3.Connection, sql: str) -> list[tuple[Any, ...]] | None:
    """Run a SQL statement; return rows or None on failure."""
    try:
        return con.execute(sql).fetchall()
    except sqlite3.Error as e:
        log.debug("execute failed: %s — %.80s", e, sql)
        return None


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Sort rows so we can compare result sets order-insensitively."""
    return sorted(rows, key=lambda r: tuple(repr(c) for c in r))


def execution_accuracy(pred: str, gold: str, db_path: str) -> bool:
    """True iff `pred` and `gold` produce the same result set on `db_path`."""
    con = sqlite3.connect(db_path)
    try:
        gold_rows = _execute(con, gold)
        pred_rows = _execute(con, pred)
    finally:
        con.close()
    if gold_rows is None:
        # Gold should always run; if it doesn't, treat as a hard failure.
        return False
    if pred_rows is None:
        return False
    if not gold_rows and not pred_rows:
        return True
    if gold_rows and pred_rows and len(gold_rows[0]) != len(pred_rows[0]):
        return False
    return _normalize_rows(gold_rows) == _normalize_rows(pred_rows)


def _normalize_sql(sql: str) -> str:
    """Normalize via sqlglot if it parses; else fall back to whitespace+lower."""
    try:
        out = sqlglot.transpile(sql, read="sqlite", write="sqlite", pretty=False)
        return out[0].strip().lower() if out else sql.strip().lower()
    except ParseError:
        return " ".join(sql.split()).strip().lower()


def exact_match(pred: str, gold: str) -> bool:
    """True iff predicted SQL matches gold after normalization."""
    return _normalize_sql(pred) == _normalize_sql(gold)


def valid_sql_rate(preds: list[str], db_path: str) -> float:
    """Fraction of predictions that parse and execute against `db_path`."""
    if not preds:
        return 0.0
    con = sqlite3.connect(db_path)
    try:
        ok = sum(1 for p in preds if _execute(con, p) is not None)
    finally:
        con.close()
    return ok / len(preds)


def score(predictions: list[str], golds: list[str], db_path: str) -> dict[str, float]:
    """Return all three metrics as a single dict.

    Lengths of `predictions` and `golds` must match.
    """
    if len(predictions) != len(golds):
        msg = f"length mismatch: {len(predictions)} preds vs {len(golds)} golds"
        raise ValueError(msg)
    if not predictions:
        return {"execution_accuracy": 0.0, "exact_match": 0.0, "valid_sql_rate": 0.0}

    n = len(predictions)
    exec_correct = sum(
        1 for p, g in zip(predictions, golds, strict=True) if execution_accuracy(p, g, db_path)
    )
    exact_correct = sum(1 for p, g in zip(predictions, golds, strict=True) if exact_match(p, g))
    valid_rate = valid_sql_rate(predictions, db_path)
    return {
        "execution_accuracy": exec_correct / n,
        "exact_match": exact_correct / n,
        "valid_sql_rate": valid_rate,
    }
