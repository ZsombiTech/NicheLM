"""FastAPI wrapper around a merged NicheLM checkpoint.

Usage:

    export MODEL_PATH=./outputs/nichelm-v1/final
    uv run uvicorn serve.server:app --port 8000

Endpoints:
- `GET /healthz` — readiness probe
- `POST /generate` — `{schema_ddl, question, max_tokens?}` -> `{sql}`

The schema is passed in by the caller, so the server can be pointed at any
SQLite DB that has been described by its CREATE TABLE statements.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data._common import SYSTEM_PROMPT_TEMPLATE
from eval.run_baselines import _strip_fences

log = logging.getLogger(__name__)

_PIPELINE: Any = None


def _load_pipeline(model_path: str) -> Any:
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )

    log.info("loading model from %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tok)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _PIPELINE
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        _PIPELINE = _load_pipeline(model_path)
    else:
        log.warning("MODEL_PATH not set; /generate will return 503 until configured")
    yield
    _PIPELINE = None


app = FastAPI(title="NicheLM", version="0.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    schema_ddl: str = Field(..., description="Full CREATE TABLE statements for the target DB.")
    question: str = Field(..., min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)


class GenerateResponse(BaseModel):
    sql: str


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "model_loaded": _PIPELINE is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if _PIPELINE is None:
        raise HTTPException(status_code=503, detail="model not loaded; set MODEL_PATH and restart")

    system = SYSTEM_PROMPT_TEMPLATE.format(ddl=req.schema_ddl)
    prompt = _PIPELINE.tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": req.question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    out = _PIPELINE(prompt, max_new_tokens=req.max_tokens, do_sample=False, return_full_text=False)
    sql = _strip_fences(out[0]["generated_text"]) if out else ""
    return GenerateResponse(sql=sql)
