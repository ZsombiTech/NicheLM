"""Tests for `eval/metrics.py`.

Uses an in-memory SQLite seeded with a tiny dataset (3 customers, 4 products)
so test cases are fully self-contained — no fixtures, no network, no DB on disk.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from eval.metrics import (
    exact_match,
    execution_accuracy,
    score,
    valid_sql_rate,
)


@pytest.fixture
def tiny_db() -> Path:
    """Create a tiny SQLite DB on disk and yield its path."""
    tmp = Path(tempfile.mkdtemp(prefix="nichelm_test_")) / "tiny.sqlite"
    con = sqlite3.connect(str(tmp))
    con.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE products  (id INTEGER PRIMARY KEY, name TEXT, price REAL);
        INSERT INTO customers VALUES (1,'Alice','US'),(2,'Bob','US'),(3,'Cara','UK');
        INSERT INTO products  VALUES (1,'Phone',300.0),(2,'Book',15.5),(3,'Lamp',45.0),(4,'Chair',120.0);
        """
    )
    con.commit()
    con.close()
    return tmp


def test_execution_accuracy_true(tiny_db: Path) -> None:
    gold = "SELECT name FROM customers WHERE country = 'US';"
    pred = "SELECT name FROM customers WHERE country='US';"  # whitespace differs
    assert execution_accuracy(pred, gold, str(tiny_db)) is True


def test_execution_accuracy_order_invariant(tiny_db: Path) -> None:
    # Same rows, different order — should still match.
    gold = "SELECT id FROM products ORDER BY id ASC;"
    pred = "SELECT id FROM products ORDER BY id DESC;"
    assert execution_accuracy(pred, gold, str(tiny_db)) is True


def test_execution_accuracy_false_different_rows(tiny_db: Path) -> None:
    gold = "SELECT name FROM customers WHERE country = 'US';"
    pred = "SELECT name FROM customers WHERE country = 'UK';"
    assert execution_accuracy(pred, gold, str(tiny_db)) is False


def test_execution_accuracy_invalid_pred(tiny_db: Path) -> None:
    gold = "SELECT COUNT(*) FROM customers;"
    pred = "SELECT FROM nope WHERE;"
    assert execution_accuracy(pred, gold, str(tiny_db)) is False


def test_execution_accuracy_column_count_mismatch(tiny_db: Path) -> None:
    gold = "SELECT name FROM customers;"
    pred = "SELECT id, name FROM customers;"
    assert execution_accuracy(pred, gold, str(tiny_db)) is False


def test_execution_accuracy_both_empty(tiny_db: Path) -> None:
    gold = "SELECT name FROM customers WHERE country = 'ZZ';"
    pred = "SELECT id FROM products WHERE price < 0;"
    # Both empty — even with different shapes, treat as equal.
    assert execution_accuracy(pred, gold, str(tiny_db)) is True


def test_exact_match_normalizes_whitespace_and_case() -> None:
    gold = "SELECT id FROM customers WHERE country = 'US';"
    pred = "select id from customers where country = 'US'"  # no semicolon, lowercase
    assert exact_match(pred, gold) is True


def test_exact_match_false_on_semantic_difference() -> None:
    assert (
        exact_match(
            "SELECT id FROM customers WHERE country = 'UK';",
            "SELECT id FROM customers WHERE country = 'US';",
        )
        is False
    )


def test_valid_sql_rate(tiny_db: Path) -> None:
    preds = [
        "SELECT 1;",
        "SELECT id FROM products;",
        "definitely not sql",
    ]
    rate = valid_sql_rate(preds, str(tiny_db))
    assert rate == pytest.approx(2 / 3)


def test_score_aggregates_all_three(tiny_db: Path) -> None:
    golds = [
        "SELECT name FROM customers WHERE country = 'US';",
        "SELECT COUNT(*) FROM products;",
    ]
    preds = [
        "SELECT name FROM customers WHERE country = 'US';",  # exec ok, exact ok
        "SELECT count(*) from products;",  # exec ok, exact ok (after normalization)
    ]
    out = score(preds, golds, str(tiny_db))
    assert out["execution_accuracy"] == pytest.approx(1.0)
    assert out["exact_match"] == pytest.approx(1.0)
    assert out["valid_sql_rate"] == pytest.approx(1.0)


def test_score_length_mismatch_raises(tiny_db: Path) -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        score(["SELECT 1;"], ["SELECT 1;", "SELECT 2;"], str(tiny_db))
