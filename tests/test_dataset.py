"""Tests for the dataset fixtures and shared helpers.

Validates the JSONL fixtures used by `make ci` parse cleanly and obey the
canonical chat-format contract. Does NOT execute any SQL — that's the
responsibility of `data/quality_check.py --fixture` (also exercised by CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data._common import ECOM_DB_ID, build_messages, make_record, stable_hash

FIXTURES = Path(__file__).parent / "fixtures"
TRAIN = FIXTURES / "mini_train.jsonl"
EVAL = FIXTURES / "mini_eval.jsonl"


@pytest.mark.parametrize("path", [TRAIN, EVAL])
def test_fixture_jsonl_parses(path: Path) -> None:
    assert path.exists(), f"missing fixture: {path}"
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        assert "messages" in rec, f"{path}:{i}: missing 'messages'"
        assert "db_id" in rec, f"{path}:{i}: missing 'db_id'"
        assert "gold_sql" in rec, f"{path}:{i}: missing 'gold_sql'"


@pytest.mark.parametrize("path", [TRAIN, EVAL])
def test_fixture_message_structure(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        msgs = rec["messages"]
        assert len(msgs) == 3
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert "Schema:" in msgs[0]["content"], "system prompt must embed the Schema"
        assert msgs[1]["content"].strip(), "user question must be non-empty"
        assert msgs[2]["content"].strip(), "assistant SQL must be non-empty"
        # gold_sql must duplicate the assistant content exactly.
        assert rec["gold_sql"] == msgs[2]["content"]


def test_eval_fixture_db_id_is_ecom() -> None:
    for line in EVAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        assert rec["db_id"] == ECOM_DB_ID


def test_train_fixture_avoids_denylist() -> None:
    from data.build_train_dataset import DENYLIST

    for line in TRAIN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        assert rec["db_id"] not in DENYLIST, (
            f"fixture train row uses denylisted db_id={rec['db_id']!r}; "
            "the denylist is for keeping the e-commerce eval honest."
        )


def test_build_messages_shape() -> None:
    msgs = build_messages(
        ddl="CREATE TABLE t (id INTEGER);", question="how many?", sql="SELECT COUNT(*) FROM t;"
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "CREATE TABLE t" in msgs[0]["content"]
    assert msgs[1]["content"] == "how many?"
    assert msgs[2]["content"] == "SELECT COUNT(*) FROM t;"


def test_make_record_duplicates_sql_in_gold() -> None:
    rec = make_record(ddl="X", question="Q", sql="SELECT 1;", db_id="ecom")
    assert rec["db_id"] == "ecom"
    assert rec["gold_sql"] == "SELECT 1;"
    assert rec["messages"][2]["content"] == "SELECT 1;"


def test_stable_hash_is_deterministic() -> None:
    assert stable_hash("foo") == stable_hash("foo")
    assert stable_hash("foo") != stable_hash("bar")


def test_shape_registry_has_at_least_thirty() -> None:
    from data.build_eval_dataset import SHAPES

    assert len(SHAPES) >= 30, f"need ≥30 shapes; found {len(SHAPES)}"
    names = {s.name for s in SHAPES}
    assert len(names) == len(SHAPES), "shape names must be unique"


def test_inspect_runs_against_fixture(caplog: pytest.LogCaptureFixture) -> None:
    from data.inspect import inspect

    with caplog.at_level("INFO"):
        inspect(EVAL, sample_n=2, seed=0)
    text = caplog.text
    assert "rows: 5" in text
    assert "db_id=ecom" in text
    assert "question length" in text


def test_find_spider_db_dir_synthetic(tmp_path: Path) -> None:
    from data.build_train_dataset import _find_spider_db_dir

    fake = tmp_path / "extracted" / "abc123" / "database"
    (fake / "concert_singer").mkdir(parents=True)
    (fake / "concert_singer" / "concert_singer.sqlite").write_bytes(b"")
    assert _find_spider_db_dir(tmp_path) == fake
    assert _find_spider_db_dir(tmp_path / "empty") is None
