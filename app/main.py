from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from ingestions.pipeline import run_ingestion
from retrieval.embedder import warmup, embed_texts
from retrieval.bm25_store import build_index, load_index, index_info
from retrieval.qdrant_store import (
    upsert_children,
    upsert_parents,
    collection_info,
)
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank_and_fetch
from evaluation.latency_tracker import reset, get_current
from evaluation.ragas_eval import score_response

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


class IngestRequest(BaseModel):
    directory: str
    max_workers: int = 4


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None
    evaluate: bool = False
    ground_truth: str | None = None


async def generate_answer(query: str, contexts: list[str]):
    prompt = f"""You are an enterprise RAG assistant. Answer ONLY from provided context.\n\nContext:\n{chr(10).join(contexts)}\n\nQuestion: {query}\n"""

    if LLM_PROVIDER == "groq":
        from groq import AsyncGroq
        client = AsyncGroq(api_key=GROQ_API_KEY)
        stream = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.2,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    elif LLM_PROVIDER == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        stream = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.2,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    elif LLM_PROVIDER == "ollama":
        import httpx
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
    else:
        raise RuntimeError(f"Unsupported provider: {LLM_PROVIDER}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise RAG API...")
    try:
        warmup()
    except Exception as e:
        logger.warning(f"Embedder warmup failed: {e}")
    try:
        load_index()
    except Exception:
        logger.info("No existing BM25 index found.")
    yield


app = FastAPI(title="Enterprise Hybrid RAG API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "bm25": index_info() if callable(index_info) else {},
        "qdrant": True,
    }


@app.post("/ingest")
async def ingest(req: IngestRequest):
    reset()
    parents, children = await run_ingestion(
            Path(req.directory),
            req.max_workers
        )
    if not children:
        raise HTTPException(400, "No documents processed")

    embeddings = embed_texts([c["content"] for c in children])
    upsert_children(children, embeddings)
    upsert_parents(parents)
    build_index(children)

    return {
        "parents": len(parents),
        "children": len(children),
        "latency": get_current(),
    }


@app.post("/search")
async def search(req: SearchRequest):
    reset()
    hits = await hybrid_search(req.query, top_k=10, filters=req.filters)
    ranked = rerank_and_fetch(req.query, hits, top_k=req.top_k)
    contexts = [r.context_for_llm for r in ranked]

    async def event_stream():
        full_answer = ""
        async for token in generate_answer(req.query, contexts):
            full_answer += token
            yield f"data: {token}\n\n"

        if req.evaluate and req.ground_truth:
            result = score_response(req.query, full_answer, contexts, req.ground_truth)
            yield f"data: {json.dumps({'evaluation': result})}\n\n"

        yield f"data: {json.dumps({'latency': get_current()})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics():
    return get_current()
