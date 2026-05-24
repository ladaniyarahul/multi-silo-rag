"""
HyDE — Hypothetical Document Embedding query rewriter.

Given a user query, the Groq LLM generates a short hypothetical passage
that *would* answer the question if it existed in the document corpus.
That passage is then embedded instead of (or in addition to) the raw query,
dramatically improving dense-retrieval recall for abstract questions.

Provider routing:
    LLM_PROVIDER=groq    → Groq API  (default, free tier)
    LLM_PROVIDER=openai  → OpenAI API
    LLM_PROVIDER=ollama  → local Ollama

Public API:
    rewrite_query(query)          -> str          (single expansion)
    rewrite_queries(queries)      -> list[str]    (batch, one call each)
"""

import os
import textwrap
from typing import Any

from loguru import logger

from evaluation.latency_tracker import track

# ── Config ────────────────────────────────────────────────────────────────────

LLM_PROVIDER  = os.getenv("LLM_PROVIDER",  "groq")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY",  "")
GROQ_MODEL    = os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY","")
OPENAI_MODEL  = os.getenv("OPENAI_MODEL",  "gpt-4o-mini")
OLLAMA_BASE   = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "llama3")

HYDE_MAX_TOKENS  = int(os.getenv("HYDE_MAX_TOKENS", "200"))
HYDE_TEMPERATURE = float(os.getenv("HYDE_TEMPERATURE", "0.5"))

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a search-augmentation assistant.
    Your task is to write a SHORT hypothetical passage (2-4 sentences) that would
    directly answer the user's question if it appeared in an enterprise document.

    Rules:
    - Write as if you are the document, not as an assistant answering the user.
    - Use specific, technical language matching the likely source document.
    - Do NOT begin with "This document" or "According to".
    - Do NOT include disclaimers or say you don't know.
    - Output ONLY the hypothetical passage — no preamble, no explanation.
""")


# ── Provider clients (lazy init) ──────────────────────────────────────────────

_groq_client   = None
_openai_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq SDK required. Run: pip install groq")
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialised.")
    return _groq_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai SDK required. Run: pip install openai")
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialised.")
    return _openai_client


# ── Provider call implementations ─────────────────────────────────────────────

def _call_groq(query: str) -> str:
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": query},
        ],
        max_tokens=HYDE_MAX_TOKENS,
        temperature=HYDE_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _call_openai(query: str) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": query},
        ],
        max_tokens=HYDE_MAX_TOKENS,
        temperature=HYDE_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _call_ollama(query: str) -> str:
    try:
        import requests
    except ImportError:
        raise ImportError("requests required for Ollama. Run: pip install requests")

    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": query},
        ],
        "stream": False,
        "options": {
            "num_predict": HYDE_MAX_TOKENS,
            "temperature": HYDE_TEMPERATURE,
        },
    }
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ── Router ────────────────────────────────────────────────────────────────────

_PROVIDERS = {
    "groq":   _call_groq,
    "openai": _call_openai,
    "ollama": _call_ollama,
}


def _call_llm(query: str) -> str:
    """Route to the configured LLM provider and return raw text."""
    provider = LLM_PROVIDER.lower()
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Choose from: {list(_PROVIDERS.keys())}"
        )
    return fn(query)


# ── Fallback passthrough ───────────────────────────────────────────────────────

def _fallback(query: str, error: Exception) -> str:
    """
    If the LLM call fails for any reason, log the error and return the
    original query unchanged so retrieval can still proceed.
    """
    logger.warning(
        f"HyDE rewrite failed ({type(error).__name__}: {error}) — "
        "falling back to original query."
    )
    return query


# ── Public API ────────────────────────────────────────────────────────────────

@track("hyde")
def rewrite_query(query: str) -> str:
    """
    Expand a user query into a hypothetical document passage via LLM.

    Args:
        query: The raw user search query.

    Returns:
        A hypothetical passage string, or the original query if the LLM fails.
    """
    if not query or not query.strip():
        raise ValueError("rewrite_query received an empty query.")

    query = query.strip()
    logger.debug(f"HyDE rewriting query: '{query[:80]}…'")

    try:
        expanded = _call_llm(query)
        if not expanded:
            raise ValueError("LLM returned an empty response.")
        logger.debug(f"HyDE expansion: '{expanded[:120]}…'")
        return expanded
    except Exception as e:
        return _fallback(query, e)


def rewrite_queries(queries: list[str]) -> list[str]:
    """
    Expand a list of queries. Each is expanded independently.

    Args:
        queries: List of raw user queries.

    Returns:
        Parallel list of expanded passages (or original queries on failure).
    """
    return [rewrite_query(q) for q in queries]


def hyde_embed(query: str) -> "np.ndarray":  # noqa: F821
    """
    Convenience function: rewrite then embed in one call.
    Returns a (384,) float32 vector ready for Qdrant search.

    Avoids importing embedder at module level to keep dependency graph clean.
    """
    from retrieval.embedder import embed_query
    expanded = rewrite_query(query)
    return embed_query(expanded)