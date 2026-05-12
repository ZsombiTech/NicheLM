#!/usr/bin/env pwsh
# Windows-friendly wrapper that mirrors the Makefile targets one-for-one.
# Usage:  .\make.ps1 <target> [-Seed 42] [-QcArgs '--fixture']
# The Linux/macOS Makefile is the source of truth; keep these in sync.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        'install', 'install-all',
        'seed',
        'dataset-train', 'dataset-eval', 'dataset',
        'qc', 'baselines',
        'test', 'lint', 'format',
        'ci', 'clean',
        'help'
    )]
    [string]$Target = 'help',

    [int]$Seed = 42,
    [string]$QcArgs = ''
)

$ErrorActionPreference = 'Stop'

function Invoke-Uv {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & uv @Args
    if ($LASTEXITCODE -ne 0) {
        throw "uv exited with code $LASTEXITCODE"
    }
}

switch ($Target) {
    'install'      { Invoke-Uv sync --extra dev }
    'install-all'  { Invoke-Uv sync --extra dev --extra data --extra eval }
    'seed' {
        Invoke-Uv run python -m data.seed `
            --out data/processed/ecom.sqlite `
            --seed $Seed
    }
    'dataset-eval' {
        Invoke-Uv run python -m data.build_eval_dataset `
            --db data/processed/ecom.sqlite `
            --out data/processed/test.jsonl `
            --n 500 --seed $Seed
    }
    'dataset-train' {
        Invoke-Uv run python -m data.build_train_dataset `
            --train-out data/processed/train.jsonl `
            --val-out data/processed/val.jsonl `
            --max-train 5000 --max-val 500 --seed $Seed
    }
    'dataset' {
        & $PSCommandPath dataset-eval -Seed $Seed
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $PSCommandPath dataset-train -Seed $Seed
    }
    'qc' {
        $extra = if ($QcArgs) { $QcArgs.Split(' ') } else { @() }
        Invoke-Uv run python -m data.quality_check @extra
    }
    'baselines' {
        Invoke-Uv run python -m eval.run_baselines `
            --models claude-haiku-4-5,llama-3.2-3b-base `
            --test data/processed/test.jsonl `
            --db data/processed/ecom.sqlite `
            --out eval/results/baselines.md
    }
    'test'   { Invoke-Uv run pytest }
    'lint' {
        Invoke-Uv run ruff check .
        Invoke-Uv run ruff format --check .
    }
    'format' {
        Invoke-Uv run ruff check --fix .
        Invoke-Uv run ruff format .
    }
    'ci' {
        & $PSCommandPath lint;  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $PSCommandPath test;  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Invoke-Uv run python -m data.quality_check --fixture
    }
    'clean' {
        Remove-Item -Force -ErrorAction SilentlyContinue `
            data/processed/*.sqlite, data/processed/*.jsonl
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
            eval/results/.cache, outputs, wandb
    }
    'help' {
        Write-Host @"
NicheLM PowerShell wrapper. Targets:

  install         uv sync --extra dev
  install-all     uv sync --extra dev --extra data --extra eval
  seed            seed the e-commerce SQLite DB
  dataset-eval    build the held-out e-commerce test.jsonl
  dataset-train   build train.jsonl + val.jsonl from Spider (needs `data` extra)
  dataset         dataset-eval + dataset-train
  qc              run the data quality check
  baselines       run Claude Haiku + raw Llama 3.2 baselines
  test            pytest
  lint            ruff check + format check
  format          auto-fix lint + auto-format
  ci              lint + test + qc --fixture
  clean           remove generated data, caches, training outputs

Flags:
  -Seed <int>     RNG seed for seed/dataset (default 42)
  -QcArgs '...'   extra args forwarded to quality_check (e.g. -QcArgs '--fixture')

Example:  .\make.ps1 ci
"@
    }
}
