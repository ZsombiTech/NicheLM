"""Inspect a generated JSONL dataset: counts, distributions, length stats, samples.

Usage:
    uv run python -m data.inspect data/processed/test.jsonl
    uv run python -m data.inspect data/processed/train.jsonl --sample 10
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics
from collections import Counter
from pathlib import Path

from data._common import read_jsonl

log = logging.getLogger(__name__)


def _pct(values: list[int], p: float) -> int:
    """Pick the p-th percentile (0.0..1.0) by sorted-index, clamped to valid range."""
    return sorted(values)[min(int(len(values) * p), len(values) - 1)]


def inspect(path: Path, *, sample_n: int = 5, seed: int = 42) -> None:
    """Print a human-readable summary of a JSONL dataset to the log."""
    rows = list(read_jsonl(path))
    if not rows:
        log.error("empty JSONL: %s", path)
        return

    log.info("file: %s   rows: %d", path, len(rows))

    db_counts = Counter(r.get("db_id", "?") for r in rows)
    log.info("db_id distribution (top 15):")
    for db_id, n in db_counts.most_common(15):
        log.info("  %-40s %d", db_id, n)

    q_lens = [len(r["messages"][1]["content"]) for r in rows if "messages" in r]
    s_lens = [len(r["messages"][2]["content"]) for r in rows if "messages" in r]
    for label, lens in (("question", q_lens), ("SQL", s_lens)):
        log.info(
            "%s length (chars): min=%d median=%d mean=%.0f p95=%d max=%d",
            label,
            min(lens),
            int(statistics.median(lens)),
            statistics.fmean(lens),
            _pct(lens, 0.95),
            max(lens),
        )

    rng = random.Random(seed)
    samples = rng.sample(rows, min(sample_n, len(rows)))
    log.info("--- %d random samples ---", len(samples))
    for i, rec in enumerate(samples, start=1):
        log.info("[%d] db_id=%s", i, rec.get("db_id"))
        log.info("    Q: %s", rec["messages"][1]["content"])
        log.info("    A: %s", rec["messages"][2]["content"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inspect(args.path, sample_n=args.sample, seed=args.seed)


if __name__ == "__main__":
    main()
