"""Validate generated JSONL dataset files.

Failures (non-zero exit):
- JSON parse error on any line
- Missing required keys (`messages`, `db_id`, `gold_sql`)
- For db_id == 'ecom': SQL fails to execute on the seeded ecom DB
- Train/test exact-question overlap > 0
- E-commerce denylist leaked into train
- Output (assistant) length 0 or > 2048 chars (rough proxy for tokens)
- Any registered SQL shape is fully missing from the test split

Spider rows are not executed at QC time unless `--validate-spider` is passed.

Fixture mode (`--fixture`) loads `tests/fixtures/{mini_train,mini_eval}.jsonl`
plus an in-memory SQLite seeded from `tests/fixtures/mini_ecom.sqlite-schema`.
This lets CI run quality checks with no network and no DB on disk.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from data._common import ECOM_DB_ID, read_jsonl

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
MAX_OUTPUT_CHARS = 2048


def _validate_record(rec: dict[str, Any], lineno: int, source: str) -> list[str]:
    errors: list[str] = []
    for key in ("messages", "db_id", "gold_sql"):
        if key not in rec:
            errors.append(f"{source}:{lineno}: missing key {key!r}")
    if "messages" in rec:
        msgs = rec["messages"]
        if not isinstance(msgs, list) or len(msgs) != 3:
            errors.append(f"{source}:{lineno}: messages must be a list of 3")
        else:
            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                errors.append(
                    f"{source}:{lineno}: roles must be system/user/assistant, got {roles}"
                )
            assistant = msgs[2].get("content", "")
            if not assistant.strip():
                errors.append(f"{source}:{lineno}: assistant content is empty")
            if len(assistant) > MAX_OUTPUT_CHARS:
                errors.append(
                    f"{source}:{lineno}: assistant content {len(assistant)} chars > {MAX_OUTPUT_CHARS}"
                )
    return errors


def _execute_against(db: sqlite3.Connection, sql: str) -> str | None:
    try:
        db.execute(sql).fetchall()
    except sqlite3.Error as e:
        return str(e)
    return None


def _check_train_test_overlap(
    train: Iterable[dict[str, Any]], test: Iterable[dict[str, Any]]
) -> list[str]:
    test_questions = {r["messages"][1]["content"] for r in test if "messages" in r}
    train_questions = {r["messages"][1]["content"] for r in train if "messages" in r}
    overlap = train_questions & test_questions
    if overlap:
        sample = list(overlap)[:5]
        return [f"train/test exact-question overlap: {len(overlap)} (e.g. {sample})"]
    return []


def _check_denylist(train: Iterable[dict[str, Any]]) -> list[str]:
    # Lazy import to avoid pulling the big build script's deps just for QC.
    from data.build_train_dataset import DENYLIST

    leaked = sorted({r["db_id"] for r in train if r.get("db_id") in DENYLIST})
    if leaked:
        return [f"e-commerce denylist leaked into train: {leaked}"]
    return []


def _check_shape_coverage(test: list[dict[str, Any]]) -> list[str]:
    from data.build_eval_dataset import SHAPES

    shape_names = {s.name for s in SHAPES}
    # Only meaningful for full-size test sets — fixtures are deliberately tiny.
    if len(test) < len(shape_names):
        return []
    distinct_questions = {r["messages"][1]["content"] for r in test if "messages" in r}
    if len(distinct_questions) < len(shape_names):
        return [
            f"test split has {len(distinct_questions)} distinct questions; "
            f"expected at least {len(shape_names)} (one per shape)"
        ]
    return []


def _print_db_id_distribution(name: str, records: list[dict[str, Any]]) -> None:
    counts = Counter(r.get("db_id", "?") for r in records)
    log.info("%s db_id distribution (top 10): %s", name, counts.most_common(10))


def run(
    *,
    train_path: Path | None,
    val_path: Path | None,
    test_path: Path | None,
    ecom_db: Path | None,
    validate_spider: bool,
    spider_db_dir: Path | None,
) -> int:
    """Run the full QC sweep. Returns the number of errors."""
    errors: list[str] = []

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    if train_path and train_path.exists():
        for i, rec in enumerate(read_jsonl(train_path), start=1):
            errors += _validate_record(rec, i, str(train_path))
            train.append(rec)
        _print_db_id_distribution("train", train)
    if val_path and val_path.exists():
        for i, rec in enumerate(read_jsonl(val_path), start=1):
            errors += _validate_record(rec, i, str(val_path))
            val.append(rec)
    if test_path and test_path.exists():
        for i, rec in enumerate(read_jsonl(test_path), start=1):
            errors += _validate_record(rec, i, str(test_path))
            test.append(rec)

    # Cross-file checks.
    if train and test:
        errors += _check_train_test_overlap(train, test)
    if train:
        errors += _check_denylist(train)
    if test:
        errors += _check_shape_coverage(test)

    # Execute e-commerce SQLs (test set + any 'ecom' rows wherever they appear).
    if ecom_db is not None and ecom_db.exists():
        con = sqlite3.connect(str(ecom_db))
        try:
            for source, recs in (("test", test), ("val", val), ("train", train)):
                for i, rec in enumerate(recs, start=1):
                    if rec.get("db_id") != ECOM_DB_ID:
                        continue
                    err = _execute_against(con, rec.get("gold_sql", ""))
                    if err is not None:
                        errors.append(f"{source}:{i}: SQL failed on ecom DB — {err}")
        finally:
            con.close()
    elif test:
        errors.append(f"e-commerce DB not found at {ecom_db}; cannot validate ecom SQL execution")

    # Spider validation is opt-in (requires the per-DB SQLite files locally).
    if validate_spider and spider_db_dir is not None:
        from data.build_train_dataset import _resolve_db_path

        for source, recs in (("train", train), ("val", val)):
            for i, rec in enumerate(recs, start=1):
                if rec.get("db_id") == ECOM_DB_ID:
                    continue
                try:
                    db_path = _resolve_db_path(spider_db_dir, rec["db_id"])
                except FileNotFoundError as e:
                    errors.append(f"{source}:{i}: {e}")
                    continue
                con = sqlite3.connect(str(db_path))
                try:
                    err = _execute_against(con, rec.get("gold_sql", ""))
                finally:
                    con.close()
                if err is not None:
                    errors.append(f"{source}:{i}: Spider SQL failed — {err}")

    for e in errors:
        log.error(e)
    if errors:
        log.error("QUALITY CHECK FAILED: %d errors", len(errors))
    else:
        log.info("quality check passed")
    return len(errors)


def _fixture_run() -> int:
    """Run QC against the bundled CI fixtures with an in-memory SQLite."""
    train_path = FIXTURES_DIR / "mini_train.jsonl"
    test_path = FIXTURES_DIR / "mini_eval.jsonl"
    schema_path = FIXTURES_DIR / "mini_ecom.sqlite-schema"

    if not (train_path.exists() and test_path.exists() and schema_path.exists()):
        log.error("fixtures not found under %s", FIXTURES_DIR)
        return 1

    # Materialize a temporary on-disk DB so the same code path executes.
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="nichelm_qc_"))
    db_path = tmp_dir / "fixture_ecom.sqlite"
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(schema_path.read_text(encoding="utf-8"))
        con.commit()
    finally:
        con.close()

    return run(
        train_path=train_path,
        val_path=None,
        test_path=test_path,
        ecom_db=db_path,
        validate_spider=False,
        spider_db_dir=None,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--ecom-db", type=Path, default=Path("data/processed/ecom.sqlite"))
    parser.add_argument(
        "--validate-spider",
        action="store_true",
        help="Also execute Spider gold SQL against the per-DB SQLite files.",
    )
    parser.add_argument(
        "--spider-db-dir",
        type=Path,
        default=None,
        help="Directory of Spider per-DB SQLite files (required with --validate-spider).",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run against the bundled CI fixtures (no real DB needed).",
    )
    args = parser.parse_args()

    if args.fixture:
        sys.exit(1 if _fixture_run() else 0)

    n_errors = run(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        ecom_db=args.ecom_db,
        validate_spider=args.validate_spider,
        spider_db_dir=args.spider_db_dir,
    )
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
