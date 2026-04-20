"""
FastAPI serving endpoint for the fine-tuned text-to-SQL model.

Loads the base Llama-3-8B-Instruct model with the QLoRA adapter and exposes
a REST API for converting natural language questions to SQL queries.

Usage:
    # Local (requires GPU):
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    python -m src.serving.app

Endpoints:
    POST /predict         — Generate SQL from a question + table schema
    POST /predict/batch   — Generate SQL for multiple questions
    GET  /health          — Health check + model info
    GET  /                — API documentation redirect
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
ADAPTER_ID = os.getenv("ADAPTER_ID", "oLittle-five/llama3-8b-wikisql-qlora")
QUANTIZE_4BIT = os.getenv("QUANTIZE_4BIT", "true").lower() == "true"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.3"))


# ── Request / Response Models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single prediction request."""
    question: str = Field(..., description="Natural language question", examples=["How many people live in Tokyo?"])
    columns: list[str] = Field(..., description="Column names from the table", examples=[["city", "country", "population"]])
    types: list[str] = Field(..., description="Column types (text, real, number)", examples=[["text", "text", "real"]])

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How many people live in Tokyo?",
                "columns": ["city", "country", "population"],
                "types": ["text", "text", "real"],
            }
        }


class PredictResponse(BaseModel):
    """Single prediction response."""
    sql: str = Field(..., description="Generated SQL query")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    prompt_tokens: int = Field(..., description="Number of input tokens")
    generated_tokens: int = Field(..., description="Number of generated tokens")


class BatchRequest(BaseModel):
    """Batch prediction request."""
    queries: list[PredictRequest] = Field(..., description="List of prediction requests")


class BatchResponse(BaseModel):
    """Batch prediction response."""
    results: list[PredictResponse]
    total_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    adapter: str
    quantization: str
    device: str


# ── Model Loading ─────────────────────────────────────────────────────────────

# Global model state
model = None
tokenizer = None
generation_config = None

# Chat template prefix (matches what SFTTrainer applied during training)
CHAT_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"


def load_model():
    """Load the base model + QLoRA adapter."""
    global model, tokenizer, generation_config

    print(f"Loading base model: {BASE_MODEL_ID}")
    print(f"Loading adapter: {ADAPTER_ID}")
    print(f"4-bit quantization: {QUANTIZE_4BIT}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    if QUANTIZE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_config = None

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Load QLoRA adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()

    # Generation config
    generation_config = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    )

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    load_model()
    yield
    # Cleanup (model will be garbage collected)


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Text-to-SQL API",
    description=(
        "Generate SQL queries from natural language questions using a fine-tuned "
        "Llama-3-8B-Instruct model with QLoRA adapters. Trained on WikiSQL."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helper Functions ──────────────────────────────────────────────────────────

def build_prompt(question: str, columns: list[str], types: list[str]) -> str:
    """Build the input prompt with chat template prefix."""
    col_defs = ", ".join(
        f"{name} ({dtype})" for name, dtype in zip(columns, types)
    )
    raw_prompt = (
        f"### Input:\n"
        f"Columns: {col_defs}\n\n"
        f"Question: {question}\n\n"
        f"### SQL:\n"
    )
    return CHAT_PREFIX + raw_prompt


def generate_sql(question: str, columns: list[str], types: list[str]) -> PredictResponse:
    """Generate SQL for a single question."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prompt = build_prompt(question, columns, types)

    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)
    elapsed_ms = (time.time() - start) * 1000

    new_tokens = outputs[0][prompt_len:]
    generated_len = len(new_tokens)
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up: take first line, remove "assistant" artifact
    sql = response.split("\n")[0].strip()
    sql = sql.replace("assistant", "").strip()

    return PredictResponse(
        sql=sql,
        generation_time_ms=round(elapsed_ms, 1),
        prompt_tokens=prompt_len,
        generated_tokens=generated_len,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns model info and status."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    device = str(next(model.parameters()).device)
    return HealthResponse(
        status="ok",
        model=BASE_MODEL_ID,
        adapter=ADAPTER_ID,
        quantization="4-bit NF4" if QUANTIZE_4BIT else "none",
        device=device,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Generate a SQL query from a natural language question.

    Provide the question, column names, and column types from the target table.
    The model returns the predicted SQL query.
    """
    if len(request.columns) != len(request.types):
        raise HTTPException(
            status_code=400,
            detail=f"columns and types must have the same length "
                   f"(got {len(request.columns)} vs {len(request.types)})",
        )
    return generate_sql(request.question, request.columns, request.types)


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """
    Generate SQL queries for multiple questions in one request.

    Each query in the batch is processed sequentially.
    """
    start = time.time()
    results = []
    for query in request.queries:
        if len(query.columns) != len(query.types):
            raise HTTPException(
                status_code=400,
                detail=f"columns and types must have the same length for query: {query.question}",
            )
        result = generate_sql(query.question, query.columns, query.types)
        results.append(result)

    total_ms = (time.time() - start) * 1000
    return BatchResponse(results=results, total_time_ms=round(total_ms, 1))


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
