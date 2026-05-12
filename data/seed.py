"""Populate `data/processed/ecom.sqlite` with deterministic synthetic data.

500 customers, 200 products across 20 categories (4 top + 16 sub), 5000 orders
spanning 2 years, ~3 items per order, 3000 reviews. Driven by a single --seed
flag for full reproducibility.
"""

from __future__ import annotations

import argparse
import logging
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from data._common import seed_all

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

N_CUSTOMERS = 500
N_TOP_CATEGORIES = 4
N_SUB_CATEGORIES = 16
N_PRODUCTS = 200
N_ORDERS = 5_000
N_REVIEWS = 3_000

START_DATE = date(2023, 1, 1)
END_DATE = date(2024, 12, 31)

ORDER_STATUSES = (
    ("pending", 5),
    ("paid", 15),
    ("shipped", 20),
    ("delivered", 50),
    ("cancelled", 7),
    ("refunded", 3),
)

TOP_CATEGORY_NAMES = ("Electronics", "Apparel", "Home & Kitchen", "Books")
SUB_CATEGORY_POOLS: dict[str, tuple[str, ...]] = {
    "Electronics": ("Phones", "Laptops", "Audio", "Cameras"),
    "Apparel": ("Mens", "Womens", "Kids", "Footwear"),
    "Home & Kitchen": ("Cookware", "Bedding", "Furniture", "Lighting"),
    "Books": ("Fiction", "Non-fiction", "Children", "Technical"),
}


def _random_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def _weighted_choice(rng: random.Random, choices: tuple[tuple[str, int], ...]) -> str:
    items, weights = zip(*choices, strict=True)
    return rng.choices(items, weights=weights, k=1)[0]


def _create_schema(con: sqlite3.Connection) -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    con.executescript(schema_sql)


def _seed_customers(con: sqlite3.Connection, fake: Faker) -> None:
    rows = []
    seen_emails: set[str] = set()
    for i in range(1, N_CUSTOMERS + 1):
        # Faker.email() is unique-ish but not guaranteed; force uniqueness.
        email = fake.unique.email()
        seen_emails.add(email)
        rows.append(
            (
                i,
                fake.name(),
                email,
                fake.country_code(),
                fake.date_between(start_date=START_DATE, end_date=END_DATE).isoformat(),
            )
        )
    con.executemany(
        "INSERT INTO customers (id, name, email, country, signup_date) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    log.info("seeded %d customers", len(rows))


def _seed_categories(con: sqlite3.Connection) -> None:
    rows = []
    cat_id = 1
    sub_to_top: dict[int, int] = {}
    # Top-level categories first (parent_id = NULL).
    top_ids: dict[str, int] = {}
    for name in TOP_CATEGORY_NAMES:
        rows.append((cat_id, name, None))
        top_ids[name] = cat_id
        cat_id += 1
    # Sub-categories, each pointing to its parent top-level.
    for top_name, subs in SUB_CATEGORY_POOLS.items():
        for sub_name in subs:
            rows.append((cat_id, sub_name, top_ids[top_name]))
            sub_to_top[cat_id] = top_ids[top_name]
            cat_id += 1
    con.executemany("INSERT INTO categories (id, name, parent_id) VALUES (?, ?, ?)", rows)
    log.info(
        "seeded %d categories (%d top + %d sub)",
        len(rows),
        N_TOP_CATEGORIES,
        N_SUB_CATEGORIES,
    )


def _seed_products(con: sqlite3.Connection, fake: Faker, rng: random.Random) -> None:
    sub_category_ids = [
        row[0]
        for row in con.execute("SELECT id FROM categories WHERE parent_id IS NOT NULL").fetchall()
    ]
    rows = []
    for i in range(1, N_PRODUCTS + 1):
        rows.append(
            (
                i,
                f"{fake.company()} {fake.word().title()}",
                rng.choice(sub_category_ids),
                round(rng.uniform(5.0, 500.0), 2),
                rng.randint(0, 250),
            )
        )
    con.executemany(
        "INSERT INTO products (id, name, category_id, price, stock) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    log.info("seeded %d products", len(rows))


def _seed_orders_and_items(con: sqlite3.Connection, rng: random.Random) -> None:
    customer_ids = [r[0] for r in con.execute("SELECT id FROM customers").fetchall()]
    products = con.execute("SELECT id, price FROM products").fetchall()

    order_rows = []
    item_rows = []
    item_id = 1

    for order_id in range(1, N_ORDERS + 1):
        customer_id = rng.choice(customer_ids)
        order_date = _random_date(rng, START_DATE, END_DATE).isoformat()
        status = _weighted_choice(rng, ORDER_STATUSES)
        # Items per order: 1-6, skewed low.
        n_items = rng.choices((1, 2, 3, 4, 5, 6), weights=(15, 30, 25, 15, 10, 5), k=1)[0]
        chosen = rng.sample(products, k=min(n_items, len(products)))
        order_total = 0.0
        for product_id, price in chosen:
            qty = rng.randint(1, 4)
            unit_price = round(float(price), 2)
            item_rows.append((item_id, order_id, product_id, qty, unit_price))
            order_total += qty * unit_price
            item_id += 1
        order_rows.append((order_id, customer_id, order_date, status, round(order_total, 2)))

    con.executemany(
        "INSERT INTO orders (id, customer_id, order_date, status, total) VALUES (?, ?, ?, ?, ?)",
        order_rows,
    )
    con.executemany(
        "INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) "
        "VALUES (?, ?, ?, ?, ?)",
        item_rows,
    )
    log.info("seeded %d orders and %d order_items", len(order_rows), len(item_rows))


def _seed_reviews(con: sqlite3.Connection, fake: Faker, rng: random.Random) -> None:
    customer_ids = [r[0] for r in con.execute("SELECT id FROM customers").fetchall()]
    product_ids = [r[0] for r in con.execute("SELECT id FROM products").fetchall()]

    rows = []
    for i in range(1, N_REVIEWS + 1):
        rating = rng.choices((1, 2, 3, 4, 5), weights=(5, 10, 20, 35, 30), k=1)[0]
        rows.append(
            (
                i,
                rng.choice(product_ids),
                rng.choice(customer_ids),
                rating,
                fake.sentence(nb_words=rng.randint(6, 20)),
                fake.date_time_between(start_date=START_DATE, end_date=END_DATE).isoformat(),
            )
        )
    con.executemany(
        "INSERT INTO reviews (id, product_id, customer_id, rating, body, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    log.info("seeded %d reviews", len(rows))


def seed(out_path: Path, seed_value: int) -> None:
    """Build a fresh SQLite DB at `out_path` populated with deterministic data."""
    seed_all(seed_value)
    rng = random.Random(seed_value)
    fake = Faker("en_US")
    Faker.seed(seed_value)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    con = sqlite3.connect(str(out_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        _create_schema(con)
        _seed_categories(con)
        _seed_customers(con, fake)
        _seed_products(con, fake, rng)
        _seed_orders_and_items(con, rng)
        _seed_reviews(con, fake, rng)
        con.commit()
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Seed the e-commerce SQLite DB.")
    parser.add_argument("--out", type=Path, default=Path("data/processed/ecom.sqlite"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed(args.out, args.seed)
    log.info("done: %s", args.out)


if __name__ == "__main__":
    main()
