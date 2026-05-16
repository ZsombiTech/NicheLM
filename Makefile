.PHONY: install install-all seed dataset-train dataset-eval dataset qc baselines baselines-smoke inspect test lint format ci clean

# Default args; override on the command line, e.g. `make qc QC_ARGS='--fixture'`.
SEED ?= 42
QC_ARGS ?=
FILE ?= data/processed/test.jsonl
SAMPLE ?= 5
# How many test prompts to send to baseline models. Default 20 keeps cost low
# while iterating; bump to 500 for the final published number:
#   make baselines BASELINE_LIMIT=500
BASELINE_LIMIT ?= 20

install:
	uv sync --extra dev

install-all:
	uv sync --extra dev --extra data --extra eval

seed:
	uv run python -m data.seed --out data/processed/ecom.sqlite --seed $(SEED)

dataset-eval:
	uv run python -m data.build_eval_dataset \
	  --db data/processed/ecom.sqlite \
	  --out data/processed/test.jsonl \
	  --n 500 --seed $(SEED)

dataset-train:
	uv run python -m data.build_train_dataset \
	  --train-out data/processed/train.jsonl \
	  --val-out data/processed/val.jsonl \
	  --max-train 5000 --max-val 500 --seed $(SEED)

dataset: dataset-eval dataset-train

qc:
	uv run python -m data.quality_check $(QC_ARGS)

baselines:
	uv run python -m eval.run_baselines \
	  --models claude-haiku-4-5,llama-3.2-3b-base \
	  --test data/processed/test.jsonl \
	  --db data/processed/ecom.sqlite \
	  --out eval/results/baselines.md \
	  --limit $(BASELINE_LIMIT)

baselines-smoke:
	uv run python -m eval.run_baselines \
	  --models claude-haiku-4-5 \
	  --test data/processed/test.jsonl \
	  --db data/processed/ecom.sqlite \
	  --out eval/results/baselines.smoke.md \
	  --limit 5 --no-cache

inspect:
	uv run python -m data.inspect $(FILE) --sample $(SAMPLE)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

ci: lint test
	uv run python -m data.quality_check --fixture

clean:
	rm -rf data/processed/*.sqlite data/processed/*.jsonl
	rm -rf eval/results/.cache
	rm -rf outputs wandb
