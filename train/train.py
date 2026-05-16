# RUN ON GPU ONLY
"""QLoRA fine-tune of Llama 3.2 3B Instruct on the Spider train set.

Driven by `train/configs/default.yaml`. Heavy deps (`unsloth`, `trl`,
`datasets`, `wandb`) are imported lazily inside `main()` so the module loads
cleanly on a CI container without GPU drivers.

Usage (on a rented GPU):

    uv sync --extra train
    uv run python -m train.train --config train/configs/default.yaml
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from data._common import seed_all

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cfg:
    base_model: str
    max_seq_length: int
    load_in_4bit: bool
    train_path: Path
    val_path: Path
    lora: dict[str, Any]
    training: dict[str, Any]
    reporting: dict[str, Any]


def load_cfg(path: Path) -> Cfg:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Cfg(
        base_model=raw["base_model"],
        max_seq_length=int(raw["max_seq_length"]),
        load_in_4bit=bool(raw["load_in_4bit"]),
        train_path=Path(raw["dataset"]["train"]),
        val_path=Path(raw["dataset"]["validation"]),
        lora=raw["lora"],
        training=raw["training"],
        reporting=raw.get("reporting", {}),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv()  # picks up WANDB_API_KEY / HF_TOKEN from .env
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("train/configs/default.yaml"))
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override `reporting.run_name` from the config.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    seed = int(cfg.training.get("seed", 42))
    seed_all(seed)

    # --- Lazy imports: GPU-only deps live behind this guard ---------------
    from datasets import load_dataset  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
    from unsloth import FastLanguageModel, is_bfloat16_supported  # type: ignore[import-not-found]

    log.info("loading base model: %s", cfg.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )

    log.info("attaching LoRA adapters: %s", cfg.lora)
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg.lora["r"]),
        lora_alpha=int(cfg.lora["alpha"]),
        lora_dropout=float(cfg.lora.get("dropout", 0.0)),
        target_modules=list(cfg.lora["target_modules"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    log.info("loading dataset: train=%s val=%s", cfg.train_path, cfg.val_path)
    raw_ds = load_dataset(
        "json",
        data_files={"train": str(cfg.train_path), "validation": str(cfg.val_path)},
    )

    def _format(example: dict[str, Any]) -> dict[str, str]:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds_train = raw_ds["train"].map(_format, remove_columns=raw_ds["train"].column_names)
    ds_val = raw_ds["validation"].map(_format, remove_columns=raw_ds["validation"].column_names)

    run_name = args.run_name or cfg.reporting.get("run_name", "nichelm")
    report_to = cfg.reporting.get("report_to", "none")
    if report_to == "wandb" and not os.environ.get("WANDB_API_KEY"):
        log.warning("WANDB_API_KEY not set; falling back to report_to='none'")
        report_to = "none"

    # `trl` has churned through several incompatible SFTConfig signatures.
    # Build the candidate kwargs and filter against the installed version's
    # signature so the script survives version drift.
    candidate_kwargs: dict[str, Any] = {
        "per_device_train_batch_size": int(cfg.training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(cfg.training["gradient_accumulation_steps"]),
        "num_train_epochs": int(cfg.training["num_train_epochs"]),
        "learning_rate": float(cfg.training["learning_rate"]),
        "warmup_ratio": float(cfg.training["warmup_ratio"]),
        "bf16": bool(cfg.training.get("bf16", True)) and is_bfloat16_supported(),
        "fp16": not (bool(cfg.training.get("bf16", True)) and is_bfloat16_supported()),
        "logging_steps": int(cfg.training.get("logging_steps", 10)),
        "save_strategy": str(cfg.training.get("save_strategy", "epoch")),
        "eval_strategy": str(cfg.training.get("eval_strategy", "epoch")),
        "output_dir": str(Path(cfg.training["output_dir"]) / run_name),
        "optim": str(cfg.training.get("optim", "adamw_8bit")),
        "lr_scheduler_type": str(cfg.training.get("lr_scheduler_type", "cosine")),
        "weight_decay": float(cfg.training.get("weight_decay", 0.0)),
        "seed": seed,
        "report_to": report_to,
        "run_name": run_name,
        # Version-dependent — both old and new names included; only valid ones
        # survive the signature filter below.
        "max_seq_length": cfg.max_seq_length,
        "max_length": cfg.max_seq_length,
        "dataset_text_field": "text",
        "packing": False,
        # trl >= 0.18 added an `eos_token` field whose default is the literal
        # placeholder '<EOS_TOKEN>'. SFTTrainer then validates that string
        # against the tokenizer vocab and crashes. Pin it to the real EOS.
        "eos_token": tokenizer.eos_token,
    }
    sft_params = set(inspect.signature(SFTConfig.__init__).parameters)
    sft_cfg = SFTConfig(**{k: v for k, v in candidate_kwargs.items() if k in sft_params})

    # trl >= 0.18 ships SFTConfig with an `eos_token` field whose default is the
    # literal placeholder '<EOS_TOKEN>'. SFTTrainer validates that string
    # against the tokenizer vocab and crashes. Force it to a real EOS token.
    real_eos = tokenizer.eos_token
    if not real_eos or real_eos == "<EOS_TOKEN>":
        # Llama 3.2 Instruct's standard EOS; falls back here when Unsloth's
        # "legacy tokenizer" mode leaves eos_token unset.
        real_eos = "<|end_of_text|>"
        tokenizer.eos_token = real_eos
    log.info("eos_token resolved to %r (vocab size=%d)", real_eos, len(tokenizer.get_vocab()))
    if real_eos not in tokenizer.get_vocab():
        msg = f"resolved eos_token {real_eos!r} not in tokenizer vocab"
        raise SystemExit(msg)

    # Some SFTConfig versions freeze fields after init; use object.__setattr__
    # as a fallback so we override the placeholder regardless.
    try:
        sft_cfg.eos_token = real_eos
    except Exception:
        object.__setattr__(sft_cfg, "eos_token", real_eos)
    log.info("sft_cfg.eos_token after override = %r", getattr(sft_cfg, "eos_token", "<missing>"))

    # Belt-and-suspenders for trl versions that dropped max_seq_length from
    # SFTConfig: pin the tokenizer's max length so the trainer respects it.
    tokenizer.model_max_length = cfg.max_seq_length

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    log.info("SFTTrainer accepts params: %s", sorted(trainer_params))
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": ds_train,
        "eval_dataset": ds_val,
        "args": sft_cfg,
    }
    # trl >= 0.12 renamed `tokenizer=` to `processing_class=`; pass whichever exists.
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    # trl >= 0.22 lifted `eos_token` into SFTTrainer's own __init__ with a
    # placeholder default that shadows whatever lives on `args`. Pass the real
    # EOS through whichever kwarg the installed version exposes.
    for eos_kwarg in ("eos_token", "eos_token_id"):
        if eos_kwarg in trainer_params:
            trainer_kwargs[eos_kwarg] = (
                real_eos if eos_kwarg == "eos_token" else tokenizer.eos_token_id
            )
    trainer = SFTTrainer(**trainer_kwargs)

    log.info("training …")
    trainer.train()

    final_dir = Path(sft_cfg.output_dir) / "final"
    log.info("saving merged model to %s", final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
