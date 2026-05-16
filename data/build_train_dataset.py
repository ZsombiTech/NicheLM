"""Build train.jsonl + val.jsonl from the public Spider dataset.

Loads `xlangai/spider` from Hugging Face datasets, builds a per-DB DDL string
from each Spider SQLite file, validates each gold query executes against its
DB, and writes the canonical chat-format JSONL with `db_id` and `gold_sql`.

Spider DBs whose schema overlaps obviously with our held-out e-commerce
evaluation are excluded via DENYLIST so the eval-set claim stays honest.

This script is NOT runnable in the scaffolding session (no HF download
allowed). Lazy import of `datasets` keeps `ruff check` and `pytest` collection
working without the `data` extra installed.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data._common import (
    extract_ddl,
    make_record,
    seed_all,
    stable_hash,
    write_jsonl,
)

log = logging.getLogger(__name__)

# Spider DBs whose schemas are clearly e-commerce-flavored. Excluded from
# training so the e-commerce eval remains a held-out domain. Curated by
# inspection of Spider's database catalogue; expand if a new overlap is found.
DENYLIST: frozenset[str] = frozenset(
    {
        "e_commerce",
        "store_product",
        "store_1",
        "products_for_hire",
        "products_gen_characteristics",
        "product_catalog",
        "customers_and_addresses",
        "customers_and_invoices",
        "customers_and_products_contacts",
        "customers_campaigns_ecommerce",
        "customers_card_transactions",
    }
)


@dataclass(frozen=True)
class SpiderRow:
    db_id: str
    question: str
    query: str


def _row_to_spider(item: dict[str, Any]) -> SpiderRow:
    """Adapt a HF dataset item to our SpiderRow. Field names follow xlangai/spider."""
    return SpiderRow(
        db_id=str(item["db_id"]),
        question=str(item["question"]).strip(),
        query=str(item["query"]).strip(),
    )


def _find_spider_db_dir(hf_cache_dir: Path) -> Path | None:
    """Walk the HF cache for a directory that looks like Spider's per-DB SQLite tree.

    Spider's archive extracts to `<some_path>/database/<db_id>/<db_id>.sqlite`.
    We look for any `database/` folder that has at least one
    `<name>/<name>.sqlite` child and return its path; `None` if not found.
    """
    if not hf_cache_dir.exists():
        return None
    for candidate in hf_cache_dir.rglob("database"):
        if not candidate.is_dir():
            continue
        for child in candidate.iterdir():
            if child.is_dir() and (child / f"{child.name}.sqlite").exists():
                return candidate
    return None


def _ensure_spider_databases(cache_dir: Path) -> Path:
    """Download and extract Spider's per-DB SQLite files.

    The `xlangai/spider` HF dataset is parquet-only (text data); the per-DB
    SQLite files are not part of it. `SALT-NLP/spider_VALUE/data.zip` is a
    public mirror of the canonical Yale Spider tarball that contains
    `data/database/<db_id>/<db_id>.sqlite` for every Spider database.

    Extracts to `<cache_dir>/spider_databases/data/database/` and returns that
    path. Idempotent: re-uses the extracted dir if it already exists.
    """
    extract_root = cache_dir / "spider_databases"
    db_dir = extract_root / "data" / "database"
    if db_dir.exists() and any(db_dir.iterdir()):
        log.info("Spider databases already extracted at %s", db_dir)
        return db_dir

    import zipfile

    from huggingface_hub import hf_hub_download

    log.info("downloading SALT-NLP/spider_VALUE/data.zip (one-time, ~95 MB) …")
    zip_path = hf_hub_download(
        repo_id="SALT-NLP/spider_VALUE",
        filename="data.zip",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )

    log.info("extracting %s → %s", Path(zip_path).name, extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith("data/database/")]
        zf.extractall(extract_root, members=members)

    if not db_dir.exists():
        msg = f"expected {db_dir} after extracting data.zip but it's missing"
        raise SystemExit(msg)
    return db_dir


def _resolve_db_path(spider_db_dir: Path, db_id: str) -> Path:
    """Locate the SQLite file for a Spider DB.

    HF's `xlangai/spider` ships SQLite files under
    `<cache>/database/<db_id>/<db_id>.sqlite`. We accept that layout or a flat
    `<dir>/<db_id>.sqlite` for ease of local debugging.
    """
    nested = spider_db_dir / db_id / f"{db_id}.sqlite"
    if nested.exists():
        return nested
    flat = spider_db_dir / f"{db_id}.sqlite"
    if flat.exists():
        return flat
    msg = f"Spider DB not found for db_id={db_id!r} in {spider_db_dir}"
    raise FileNotFoundError(msg)


def _validate_sql(db_path: Path, sql: str) -> bool:
    """Return True iff the gold SQL executes without error against `db_path`."""
    try:
        con = sqlite3.connect(str(db_path))
        try:
            con.execute(sql).fetchall()
        finally:
            con.close()
    except sqlite3.Error as e:
        log.debug("SQL drop (%s): %s", e, sql[:80])
        return False
    return True


def _build_records(
    rows: list[SpiderRow],
    spider_db_dir: Path,
    cap: int,
) -> list[dict[str, Any]]:
    """Build chat-format records from Spider rows, capped at `cap` after filtering."""
    ddl_cache: dict[str, str] = {}
    db_path_cache: dict[str, Path] = {}
    out: list[dict[str, Any]] = []

    # Deterministic ordering before capping: sort by stable hash of (db_id, question).
    rows_sorted = sorted(rows, key=lambda r: stable_hash(f"{r.db_id}|{r.question}"))

    for r in rows_sorted:
        if r.db_id in DENYLIST:
            continue
        try:
            db_path = db_path_cache.setdefault(r.db_id, _resolve_db_path(spider_db_dir, r.db_id))
        except FileNotFoundError as e:
            log.warning("%s; dropping row", e)
            continue

        ddl = ddl_cache.get(r.db_id)
        if ddl is None:
            try:
                ddl = extract_ddl(db_path)
            except sqlite3.Error as e:
                log.warning("DDL extraction failed for %s: %s", r.db_id, e)
                continue
            ddl_cache[r.db_id] = ddl

        if not _validate_sql(db_path, r.query):
            continue

        out.append(make_record(ddl=ddl, question=r.question, sql=r.query, db_id=r.db_id))
        if len(out) >= cap:
            break

    return out


def build(
    *,
    train_out: Path,
    val_out: Path,
    max_train: int,
    max_val: int,
    seed: int,
    spider_db_dir: Path | None,
    hf_cache_dir: Path | None,
) -> None:
    """Build train.jsonl + val.jsonl from `xlangai/spider`."""
    seed_all(seed)

    # Lazy import — `datasets` is in the optional `data` extra.
    from datasets import load_dataset  # type: ignore[import-not-found]

    log.info("loading xlangai/spider …")
    ds = load_dataset(
        "xlangai/spider",
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
    )

    if spider_db_dir is None:
        # `xlangai/spider` is parquet-only — the per-DB SQLite files aren't
        # part of `load_dataset`. Try the cache first (in case we extracted
        # them before), then fall back to downloading from a mirror.
        cache_root = hf_cache_dir or Path(".hf_cache")
        spider_db_dir = _find_spider_db_dir(cache_root)
        if spider_db_dir is None:
            spider_db_dir = _ensure_spider_databases(cache_root)
        log.info("Spider DB dir: %s", spider_db_dir)

    train_rows = [_row_to_spider(item) for item in ds["train"]]
    val_rows = [_row_to_spider(item) for item in ds["validation"]]
    log.info("spider train=%d, validation=%d", len(train_rows), len(val_rows))

    train_records = _build_records(train_rows, spider_db_dir, max_train)
    val_records = _build_records(val_rows, spider_db_dir, max_val)

    write_jsonl(train_out, train_records)
    write_jsonl(val_out, val_records)
    log.info(
        "wrote %d train + %d val records (denylist=%d entries)",
        len(train_records),
        len(val_records),
        len(DENYLIST),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val-out", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--max-train", type=int, default=5_000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--spider-db-dir",
        type=Path,
        default=None,
        help="Directory containing Spider per-DB SQLite files. "
        "Optional — auto-discovered from --hf-cache-dir if omitted.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path(".hf_cache"),
        help="Hugging Face datasets cache directory.",
    )
    args = parser.parse_args()

    build(
        train_out=args.train_out,
        val_out=args.val_out,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed,
        spider_db_dir=args.spider_db_dir,
        hf_cache_dir=args.hf_cache_dir,
    )


if __name__ == "__main__":
    main()
