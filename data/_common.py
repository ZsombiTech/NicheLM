"""Shared helpers: seeding, JSONL I/O, schema-to-DDL extraction, prompt building."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = (
    "You are a text-to-SQL assistant. Given the schema below, write a single "
    "valid SQLite query that answers the user's question. Reply with SQL "
    "only, no commentary, no markdown fences.\n\nSchema:\n{ddl}"
)

ECOM_DB_ID = "ecom"


def seed_all(seed: int) -> None:
    """Seed `random`, `numpy`, and (lazily) `torch` for full determinism."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        log.debug("torch not installed; skipping torch.manual_seed")


def stable_hash(text: str) -> int:
    """Deterministic 64-bit hash of a string. Use for split assignment / sampling."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def extract_ddl(db_path: str | Path) -> str:
    """Read all CREATE TABLE statements from a SQLite DB into a single DDL string.

    Used both for the e-commerce schema (post-seeding) and per-Spider-DB at
    train-build time, so the same prompt format covers both cases.
    """
    con = sqlite3.connect(str(db_path))
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    return "\n\n".join(r["sql"].strip() + ";" for r in rows)


def build_messages(ddl: str, question: str, sql: str) -> list[dict[str, str]]:
    """Build the canonical 3-message chat sequence used in every JSONL row."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(ddl=ddl)},
        {"role": "user", "content": question},
        {"role": "assistant", "content": sql},
    ]


def make_record(*, ddl: str, question: str, sql: str, db_id: str) -> dict[str, Any]:
    """Build a single JSONL record with `messages`, `db_id`, and `gold_sql`."""
    return {
        "messages": build_messages(ddl, question, sql),
        "db_id": db_id,
        "gold_sql": sql,
    }


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Stream records to a JSONL file. Returns the number of lines written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1
    log.info("wrote %d records to %s", n, p)
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file. Raises on malformed lines."""
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"{path}:{i}: invalid JSON ({e.msg})"
                raise ValueError(msg) from e


def execute_sql(db_path: str | Path, sql: str) -> list[tuple[Any, ...]]:
    """Execute a single SQL statement and return rows. Raises on SQL errors."""
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.execute(sql)
        return cur.fetchall()
    finally:
        con.close()
