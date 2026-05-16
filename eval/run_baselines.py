"""Run baseline models (Claude Haiku, raw Llama 3.2 3B) against the held-out
e-commerce test set, with on-disk response caching.

The same system prompt the fine-tuned model receives is sent to every baseline
(schema-in-prompt format); only the user question varies. Outputs a markdown
results table with execution accuracy, exact match, valid-SQL %, mean latency,
and mean cost per request.

Lazy-imports anthropic/transformers so the module loads cleanly without the
`eval` extra installed (CI keeps `dev` only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data._common import read_jsonl
from eval.metrics import score

log = logging.getLogger(__name__)

# Per-token pricing in USD per 1M tokens.
# Update when prices change — last verified 2026-05-10.
PRICING: dict[str, dict[str, float]] = {
    # Anthropic full id used at call time; keep alias for CLI ergonomics.
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # Local model — power/runtime cost only; left as 0 for parity reporting.
    "llama-3.2-3b-base": {"input": 0.00, "output": 0.00},
}

ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"
LOCAL_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

CACHE_DIR = Path("eval/results/.cache")


@dataclass
class CallResult:
    text: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class ModelStats:
    name: str
    latencies: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    golds: list[str] = field(default_factory=list)


def _cache_key(model: str, system: str, user: str) -> Path:
    h = hashlib.sha1(f"{model}\n{system}\n{user}".encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{model}__{h}.json"


def _load_cached(path: Path) -> CallResult | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CallResult(**raw)


def _save_cached(path: Path, result: CallResult) -> None:
    path.write_text(json.dumps(result.__dict__), encoding="utf-8")


def _strip_fences(text: str) -> str:
    """Strip ```sql … ``` fences models love to add despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        # Drop the opening fence (with optional language tag) and trailing fence.
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def _call_anthropic(model_alias: str, system: str, user: str) -> CallResult:
    from anthropic import Anthropic  # type: ignore[import-not-found]

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=ANTHROPIC_MODEL_ID,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0.0,
    )
    latency = time.perf_counter() - t0
    # Concatenate any text blocks (Anthropic returns a list).
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    text = _strip_fences("".join(parts))
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    return CallResult(
        text=text,
        latency_s=latency,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=_cost(model_alias, in_tok, out_tok),
    )


_LOCAL_PIPELINE: Any = None


def _call_local_llama(_model_alias: str, system: str, user: str) -> CallResult:
    """CPU/MPS/CUDA-agnostic local generation. Slow on laptop — that's expected."""
    global _LOCAL_PIPELINE
    if _LOCAL_PIPELINE is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

        log.info("loading %s (this may be slow on CPU/MPS) …", LOCAL_MODEL_ID)
        tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_ID, torch_dtype="auto", device_map="auto"
        )
        _LOCAL_PIPELINE = pipeline("text-generation", model=model, tokenizer=tok)

    prompt = _LOCAL_PIPELINE.tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    t0 = time.perf_counter()
    out = _LOCAL_PIPELINE(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
    latency = time.perf_counter() - t0
    text = _strip_fences(out[0]["generated_text"]) if out else ""
    return CallResult(text=text, latency_s=latency)


_DISPATCH: dict[str, Callable[[str, str, str], CallResult]] = {
    "claude-haiku-4-5": _call_anthropic,
    "llama-3.2-3b-base": _call_local_llama,
}


def _run_one_model(
    model: str,
    examples: list[dict[str, Any]],
    no_cache: bool,
) -> ModelStats:
    fn = _DISPATCH[model]
    stats = ModelStats(name=model)
    for ex in examples:
        msgs = ex["messages"]
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        cache_path = _cache_key(model, system, user)
        result = None if no_cache else _load_cached(cache_path)
        if result is None:
            result = fn(model, system, user)
            _save_cached(cache_path, result)
        stats.latencies.append(result.latency_s)
        stats.costs.append(result.cost_usd)
        stats.predictions.append(result.text)
        stats.golds.append(ex["gold_sql"])
    return stats


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = ["model", "exec_acc", "exact", "valid_sql", "mean_latency_s", "mean_cost_usd", "n"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append(
            "| {model} | {exec_acc:.3f} | {exact:.3f} | {valid_sql:.3f} | "
            "{mean_latency_s:.3f} | {mean_cost_usd:.6f} | {n} |".format(**r)
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Load `.env` so API keys can live in a file the user doesn't have to export.
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="claude-haiku-4-5,llama-3.2-3b-base",
        help="Comma-separated subset of " + ", ".join(_DISPATCH),
    )
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/processed/ecom.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/baselines.md"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Truncate test set for smoke runs.")
    args = parser.parse_args()

    if not args.test.exists():
        log.error("test set not found at %s", args.test)
        sys.exit(1)
    if not args.db.exists():
        log.error("DB not found at %s; run `make seed`", args.db)
        sys.exit(1)

    examples = list(read_jsonl(args.test))
    if args.limit:
        examples = examples[: args.limit]
    log.info("loaded %d test examples", len(examples))

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected if m not in _DISPATCH]
    if unknown:
        log.error("unknown models: %s", unknown)
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for model in selected:
        log.info("running %s …", model)
        stats = _run_one_model(model, examples, no_cache=args.no_cache)
        metrics = score(stats.predictions, stats.golds, str(args.db))
        rows.append(
            {
                "model": model,
                "exec_acc": metrics["execution_accuracy"],
                "exact": metrics["exact_match"],
                "valid_sql": metrics["valid_sql_rate"],
                "mean_latency_s": statistics.fmean(stats.latencies),
                "mean_cost_usd": statistics.fmean(stats.costs),
                "n": len(stats.predictions),
            }
        )

    table = _format_table(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        f"# Baseline results\n\nTest set: `{args.test}` (n={len(examples)})\n\n{table}\n",
        encoding="utf-8",
    )
    log.info("wrote %s", args.out)
    log.info("\n%s", table)


if __name__ == "__main__":
    main()
