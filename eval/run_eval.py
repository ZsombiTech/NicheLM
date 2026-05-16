"""Evaluate a tuned model checkpoint on the held-out e-commerce test set.

Loads a merged-model directory (produced by the training pipeline), runs it
over `test.jsonl`, executes predictions against `ecom.sqlite`, and writes a
results CSV plus a markdown summary. Stub-but-functional — won't run until a
checkpoint exists.

Lazy-imports `transformers`/`torch`, so this module is importable without the
`eval` extra installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data._common import read_jsonl
from eval.metrics import exact_match, execution_accuracy, valid_sql_rate
from eval.run_baselines import _strip_fences

log = logging.getLogger(__name__)

CACHE_DIR = Path("eval/results/.cache")


@dataclass
class Prediction:
    pred: str
    gold: str
    latency_s: float
    correct_exec: bool
    correct_exact: bool


def _cache_key(checkpoint: str, system: str, user: str) -> Path:
    h = hashlib.sha1(f"{checkpoint}\n{system}\n{user}".encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"tuned__{h}.json"


def _generate(pipe: Any, system: str, user: str, max_new_tokens: int) -> tuple[str, float]:
    prompt = pipe.tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    t0 = time.perf_counter()
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
    return _strip_fences(out[0]["generated_text"]) if out else "", time.perf_counter() - t0


def evaluate(
    *,
    checkpoint: Path,
    test_path: Path,
    db_path: Path,
    out_csv: Path,
    out_md: Path,
    no_cache: bool,
    limit: int | None,
    max_new_tokens: int,
) -> None:
    if not checkpoint.exists():
        msg = f"checkpoint not found: {checkpoint}"
        raise SystemExit(msg)
    if not test_path.exists():
        msg = f"test set not found: {test_path}"
        raise SystemExit(msg)
    if not db_path.exists():
        msg = f"DB not found: {db_path}"
        raise SystemExit(msg)

    # Lazy imports — `transformers`/`torch` are in the optional `eval` extra.
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )

    log.info("loading checkpoint from %s …", checkpoint)
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), torch_dtype="auto", device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tok)

    examples = list(read_jsonl(test_path))
    if limit:
        examples = examples[:limit]
    log.info("loaded %d test examples", len(examples))

    preds: list[Prediction] = []
    for i, ex in enumerate(examples, start=1):
        system = ex["messages"][0]["content"]
        user = ex["messages"][1]["content"]
        gold = ex["gold_sql"]

        cache_path = _cache_key(str(checkpoint), system, user)
        if not no_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            text, latency = cached["text"], cached["latency_s"]
        else:
            text, latency = _generate(pipe, system, user, max_new_tokens)
            cache_path.write_text(
                json.dumps({"text": text, "latency_s": latency}), encoding="utf-8"
            )

        preds.append(
            Prediction(
                pred=text,
                gold=gold,
                latency_s=latency,
                correct_exec=execution_accuracy(text, gold, str(db_path)),
                correct_exact=exact_match(text, gold),
            )
        )
        if i % 25 == 0:
            log.info("evaluated %d/%d", i, len(examples))

    n = len(preds)
    exec_acc = sum(p.correct_exec for p in preds) / n if n else 0.0
    exact_acc = sum(p.correct_exact for p in preds) / n if n else 0.0
    valid_rate = valid_sql_rate([p.pred for p in preds], str(db_path))
    mean_latency = statistics.fmean(p.latency_s for p in preds) if preds else 0.0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "gold", "pred", "exec_correct", "exact_correct", "latency_s"])
        for i, p in enumerate(preds, start=1):
            w.writerow(
                [i, p.gold, p.pred, int(p.correct_exec), int(p.correct_exact), f"{p.latency_s:.4f}"]
            )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    md = (
        f"# Tuned model results\n\n"
        f"- checkpoint: `{checkpoint}`\n"
        f"- test set: `{test_path}` (n={n})\n"
        f"- execution accuracy: **{exec_acc:.3f}**\n"
        f"- exact match: {exact_acc:.3f}\n"
        f"- valid SQL rate: {valid_rate:.3f}\n"
        f"- mean latency: {mean_latency:.3f}s\n"
    )
    out_md.write_text(md, encoding="utf-8")
    log.info("wrote %s and %s", out_csv, out_md)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/processed/ecom.sqlite"))
    parser.add_argument("--out-csv", type=Path, default=Path("eval/results/tuned.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("eval/results/tuned.md"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    try:
        evaluate(
            checkpoint=args.checkpoint,
            test_path=args.test,
            db_path=args.db,
            out_csv=args.out_csv,
            out_md=args.out_md,
            no_cache=args.no_cache,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
    except SystemExit as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
