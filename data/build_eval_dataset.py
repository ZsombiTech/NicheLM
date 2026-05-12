"""Build the held-out e-commerce test set (`test.jsonl`).

Generates ~500 (question, SQL) pairs against the seeded `ecom.sqlite` using a
registry of ~30 SQL "shapes". Each generated SQL is validated by executing it
against the DB; failures are dropped and re-drawn until the quota is met. The
test split is guaranteed to contain at least one example of every shape.

The `--use-llm-paraphrase` flag is a stub for a later session and never makes
network calls in the default offline run.
"""

from __future__ import annotations

import argparse
import logging
import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faker import Faker

from data._common import (
    ECOM_DB_ID,
    extract_ddl,
    make_record,
    seed_all,
    write_jsonl,
)

log = logging.getLogger(__name__)

# Builder takes (cur, rng, fake) and returns (question, sql).
Builder = Callable[[sqlite3.Cursor, random.Random, Faker], tuple[str, str]]


@dataclass(frozen=True)
class Shape:
    name: str
    build: Builder


def _pick_country(cur: sqlite3.Cursor, rng: random.Random) -> str:
    rows = cur.execute("SELECT DISTINCT country FROM customers").fetchall()
    return rng.choice(rows)[0]


def _pick_status(rng: random.Random) -> str:
    return rng.choice(("pending", "paid", "shipped", "delivered", "cancelled", "refunded"))


def _pick_top_category(cur: sqlite3.Cursor, rng: random.Random) -> tuple[int, str]:
    rows = cur.execute("SELECT id, name FROM categories WHERE parent_id IS NULL").fetchall()
    return rng.choice(rows)


def _pick_sub_category(cur: sqlite3.Cursor, rng: random.Random) -> tuple[int, str]:
    rows = cur.execute("SELECT id, name FROM categories WHERE parent_id IS NOT NULL").fetchall()
    return rng.choice(rows)


# -- Shape builders ------------------------------------------------------------
# Each builder returns a (question, sql) pair that runs against the seeded DB.


def _b1_select_limit(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    k = rng.choice((5, 10, 20))
    return (
        f"List the names of the first {k} customers (by id).",
        f"SELECT name FROM customers ORDER BY id LIMIT {k};",
    )


def _b2_where_numeric_gt(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    threshold = rng.choice((50, 100, 200))
    return (
        f"Show product names with a price greater than {threshold}.",
        f"SELECT name FROM products WHERE price > {threshold};",
    )


def _b3_where_date_range(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    year = rng.choice((2023, 2024))
    return (
        f"List order ids from {year}.",
        f"SELECT id FROM orders WHERE order_date >= '{year}-01-01' "
        f"AND order_date <= '{year}-12-31';",
    )


def _b4_where_string_like(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "Find customers whose name starts with 'A'.",
        "SELECT id, name FROM customers WHERE name LIKE 'A%';",
    )


def _b5_where_in(cur, rng, _fake):  # type: ignore[no-untyped-def]
    countries = [_pick_country(cur, rng) for _ in range(2)]
    in_list = ", ".join(f"'{c}'" for c in countries)
    return (
        f"How many customers are in {countries[0]} or {countries[1]}?",
        f"SELECT COUNT(*) FROM customers WHERE country IN ({in_list});",
    )


def _b6_count_all(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return ("How many products are there?", "SELECT COUNT(*) FROM products;")


def _b7_count_filter(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    status = _pick_status(rng)
    return (
        f"How many orders have status '{status}'?",
        f"SELECT COUNT(*) FROM orders WHERE status = '{status}';",
    )


def _b8_avg(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return ("What is the average price of products?", "SELECT AVG(price) FROM products;")


def _b8b_sum(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    status = _pick_status(rng)
    return (
        f"What is the total revenue from orders with status '{status}'?",
        f"SELECT SUM(total) FROM orders WHERE status = '{status}';",
    )


def _b8c_min(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return ("What is the lowest product price?", "SELECT MIN(price) FROM products;")


def _b8d_max(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return ("What is the highest order total?", "SELECT MAX(total) FROM orders;")


def _b9_top_n(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    k = rng.choice((3, 5, 10))
    return (
        f"List the top {k} most expensive products by price.",
        f"SELECT name, price FROM products ORDER BY price DESC LIMIT {k};",
    )


def _b10_join_orders_customers(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "List each order id and the name of the customer who placed it.",
        "SELECT o.id, c.name FROM orders o JOIN customers c ON c.id = o.customer_id;",
    )


def _b11_join_items_products_orders(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "For each order item, return the order id, product name and quantity.",
        "SELECT oi.order_id, p.name, oi.quantity "
        "FROM order_items oi JOIN products p ON p.id = oi.product_id "
        "JOIN orders o ON o.id = oi.order_id;",
    )


def _b12_join_reviews_products_customers(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "List each review's rating along with the product name and reviewer name.",
        "SELECT r.rating, p.name AS product, c.name AS reviewer "
        "FROM reviews r JOIN products p ON p.id = r.product_id "
        "JOIN customers c ON c.id = r.customer_id;",
    )


def _b13_groupby_count(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "How many orders per status are there?",
        "SELECT status, COUNT(*) AS n FROM orders GROUP BY status;",
    )


def _b14_groupby_sum(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "Total revenue per order status.",
        "SELECT status, SUM(total) AS revenue FROM orders GROUP BY status;",
    )


def _b15_groupby_two(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "For each customer country and order status, count the number of orders.",
        "SELECT c.country, o.status, COUNT(*) AS n "
        "FROM orders o JOIN customers c ON c.id = o.customer_id "
        "GROUP BY c.country, o.status;",
    )


def _b16_having_count(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    threshold = rng.choice((5, 10, 20))
    return (
        f"List customer ids who have placed more than {threshold} orders.",
        f"SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > {threshold};",
    )


def _b17_having_sum(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    threshold = rng.choice((500, 1000, 2000))
    return (
        f"List customer ids whose total spend across all orders exceeds {threshold}.",
        f"SELECT customer_id, SUM(total) AS spend FROM orders "
        f"GROUP BY customer_id HAVING SUM(total) > {threshold};",
    )


def _b18_groupby_top_n(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    k = rng.choice((3, 5))
    return (
        f"Top {k} categories by number of products.",
        f"SELECT category_id, COUNT(*) AS n FROM products "
        f"GROUP BY category_id ORDER BY n DESC LIMIT {k};",
    )


def _b19_subquery_in(cur, rng, _fake):  # type: ignore[no-untyped-def]
    country = _pick_country(cur, rng)
    return (
        f"List ids of orders placed by customers from {country}.",
        f"SELECT id FROM orders WHERE customer_id IN "
        f"(SELECT id FROM customers WHERE country = '{country}');",
    )


def _b20_subquery_from(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "Average per-customer total order value.",
        "SELECT AVG(spend) AS avg_spend FROM ("
        "SELECT customer_id, SUM(total) AS spend FROM orders GROUP BY customer_id"
        ") sub;",
    )


def _b21_correlated_exists(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "List customer ids who have at least one review.",
        "SELECT id FROM customers c WHERE EXISTS "
        "(SELECT 1 FROM reviews r WHERE r.customer_id = c.id);",
    )


def _b22_cte_single(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "Using a CTE, return the total spend per customer ordered descending.",
        "WITH spend AS (SELECT customer_id, SUM(total) AS s FROM orders GROUP BY customer_id) "
        "SELECT customer_id, s FROM spend ORDER BY s DESC;",
    )


def _b23_cte_chained(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "Using two CTEs, return the average spend among the top 10 spenders.",
        "WITH spend AS (SELECT customer_id, SUM(total) AS s FROM orders GROUP BY customer_id), "
        "top10 AS (SELECT customer_id, s FROM spend ORDER BY s DESC LIMIT 10) "
        "SELECT AVG(s) AS avg_top10 FROM top10;",
    )


def _b24_window_row_number(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "For each customer, number their orders chronologically (1 = earliest).",
        "SELECT id, customer_id, order_date, "
        "ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS rn "
        "FROM orders;",
    )


def _b25_window_rank(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "Rank products by price within each category (1 = most expensive).",
        "SELECT id, category_id, price, "
        "RANK() OVER (PARTITION BY category_id ORDER BY price DESC) AS rk "
        "FROM products;",
    )


def _b26_window_running_sum(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "Running total of order totals over time, ordered by order_date.",
        "SELECT id, order_date, total, "
        "SUM(total) OVER (ORDER BY order_date) AS running_total "
        "FROM orders;",
    )


def _b27_self_join_categories(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "List sub-category names alongside their parent top-level category name.",
        "SELECT sub.name AS sub_category, top.name AS parent "
        "FROM categories sub JOIN categories top ON sub.parent_id = top.id;",
    )


def _b28_left_join_null(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "Find product ids that have never appeared in any order item.",
        "SELECT p.id FROM products p LEFT JOIN order_items oi ON oi.product_id = p.id "
        "WHERE oi.id IS NULL;",
    )


def _b29_date_bucket(_cur, _rng, _fake):  # type: ignore[no-untyped-def]
    return (
        "Number of orders per month across all years (YYYY-MM).",
        "SELECT strftime('%Y-%m', order_date) AS ym, COUNT(*) AS n "
        "FROM orders GROUP BY ym ORDER BY ym;",
    )


def _b30_union(_cur, rng, _fake):  # type: ignore[no-untyped-def]
    s1 = _pick_status(rng)
    s2 = _pick_status(rng)
    while s2 == s1:
        s2 = _pick_status(rng)
    return (
        f"List ids of orders with status '{s1}' or '{s2}'.",
        f"SELECT id FROM orders WHERE status = '{s1}' "
        f"UNION SELECT id FROM orders WHERE status = '{s2}';",
    )


SHAPES: tuple[Shape, ...] = (
    Shape("select_limit", _b1_select_limit),
    Shape("where_numeric_gt", _b2_where_numeric_gt),
    Shape("where_date_range", _b3_where_date_range),
    Shape("where_string_like", _b4_where_string_like),
    Shape("where_in", _b5_where_in),
    Shape("count_all", _b6_count_all),
    Shape("count_filter", _b7_count_filter),
    Shape("avg_filter", _b8_avg),
    Shape("sum_filter", _b8b_sum),
    Shape("min_filter", _b8c_min),
    Shape("max_filter", _b8d_max),
    Shape("top_n", _b9_top_n),
    Shape("join_orders_customers", _b10_join_orders_customers),
    Shape("join_items_products_orders", _b11_join_items_products_orders),
    Shape("join_reviews_products_customers", _b12_join_reviews_products_customers),
    Shape("groupby_count", _b13_groupby_count),
    Shape("groupby_sum", _b14_groupby_sum),
    Shape("groupby_two_cols", _b15_groupby_two),
    Shape("having_count", _b16_having_count),
    Shape("having_sum", _b17_having_sum),
    Shape("groupby_top_n", _b18_groupby_top_n),
    Shape("subquery_in", _b19_subquery_in),
    Shape("subquery_from", _b20_subquery_from),
    Shape("correlated_exists", _b21_correlated_exists),
    Shape("cte_single", _b22_cte_single),
    Shape("cte_chained", _b23_cte_chained),
    Shape("window_row_number", _b24_window_row_number),
    Shape("window_rank", _b25_window_rank),
    Shape("window_running_sum", _b26_window_running_sum),
    Shape("self_join_categories", _b27_self_join_categories),
    Shape("left_join_null", _b28_left_join_null),
    Shape("date_bucket", _b29_date_bucket),
    Shape("union", _b30_union),
)


def _validate(con: sqlite3.Connection, sql: str) -> bool:
    try:
        con.execute(sql).fetchall()
    except sqlite3.Error as e:
        log.warning("validation drop: %s — %.80s", e, sql)
        return False
    return True


def build(
    *,
    db_path: Path,
    out_path: Path,
    n: int,
    seed: int,
) -> None:
    """Generate `n` validated (question, SQL) examples against `db_path`."""
    if not db_path.exists():
        msg = f"DB not found at {db_path}; run `make seed` first."
        raise SystemExit(msg)

    seed_all(seed)
    rng = random.Random(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    ddl = extract_ddl(db_path)

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    records: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    try:
        # Pass 1: every shape at least once.
        for shape in SHAPES:
            for _ in range(20):  # bounded retries
                question, sql = shape.build(cur, rng, fake)
                if question in seen_questions or not _validate(con, sql):
                    continue
                records.append(make_record(ddl=ddl, question=question, sql=sql, db_id=ECOM_DB_ID))
                seen_questions.add(question)
                break
            else:
                log.error("shape %s never produced a valid example", shape.name)

        # Pass 2: random shapes until quota.
        guard = 0
        while len(records) < n and guard < n * 10:
            guard += 1
            shape = rng.choice(SHAPES)
            question, sql = shape.build(cur, rng, fake)
            if question in seen_questions or not _validate(con, sql):
                continue
            records.append(make_record(ddl=ddl, question=question, sql=sql, db_id=ECOM_DB_ID))
            seen_questions.add(question)
    finally:
        con.close()

    write_jsonl(out_path, records)
    log.info("wrote %d unique e-commerce eval examples", len(records))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/processed/ecom.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-llm-paraphrase",
        action="store_true",
        help="Stub for a later session; no-op in offline default.",
    )
    args = parser.parse_args()

    if args.use_llm_paraphrase:
        log.warning("--use-llm-paraphrase is not implemented in this scaffold; ignoring")

    build(db_path=args.db, out_path=args.out, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
