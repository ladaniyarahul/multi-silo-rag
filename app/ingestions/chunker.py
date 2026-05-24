"""
Parent-Child Chunker — splits parser output into retrieval-ready chunks.

Why two sizes?
  Child chunks (200 tok): small, precise — used for vector similarity search.
  Parent chunks (1000 tok): large, context-rich — sent to the LLM verbatim.

Each child stores a `parent_id` so the retrieval layer can
fetch the full parent context after ranking.

Also injects a `document_topic` tag computed by a zero-shot classifier
(falls back to the parser-supplied section header if the model is unavailable).

Returns two parallel lists:
    parents : list[dict]   — full-context chunks (stored separately in Qdrant)
    children: list[dict]   — search-target chunks (embedded + indexed)

Standard chunk schema:
    {
        "id":      str,          # deterministic SHA-1 of content + source
        "content": str,
        "metadata": {
            "file_type":       str,
            "source":          str,
            "last_modified":   str,
            "section":         str,
            "element_type":    str,
            "document_topic":  str,
            "parent_id":       str | None,   # None for parent chunks
            "chunk_type":      "parent" | "child",
        }
    }
"""

import hashlib
import os
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from evaluation.latency_tracker import track

# ── Config (can be overridden via env) ───────────────────────────────────────

PARENT_CHUNK_SIZE    = int(os.getenv("CHUNK_PARENT_TOKENS", 1000))
CHILD_CHUNK_SIZE     = int(os.getenv("CHUNK_CHILD_TOKENS",  200))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP",        20))

# ── Zero-shot topic classifier (optional) ────────────────────────────────────
# If transformers is available, we use a tiny zero-shot model to auto-tag
# document_topic.  If not, we fall back to the section heading.

_TOPIC_CANDIDATES = [
    "policy and compliance",
    "project management",
    "technical documentation",
    "financial report",
    "human resources",
    "product specification",
    "meeting notes",
    "legal agreement",
    "general information",
]

try:
    from transformers import pipeline as hf_pipeline
    _classifier = hf_pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-MiniLM2-L6-H768",
        device=-1,           # CPU
    )
    _USE_CLASSIFIER = True
    logger.info("Zero-shot topic classifier loaded.")
except Exception:
    _classifier = None
    _USE_CLASSIFIER = False
    logger.warning("Zero-shot classifier unavailable — using section heading as document_topic.")


def _classify_topic(text: str, fallback: str) -> str:
    """Return the most-likely topic label for a text snippet."""
    if not _USE_CLASSIFIER or _classifier is None:
        return fallback
    try:
        snippet = text[:512]          # classifier needs only a short sample
        result = _classifier(snippet, _TOPIC_CANDIDATES, multi_label=False)
        return result["labels"][0]    # type: ignore[index]
    except Exception as e:
        logger.warning(f"Topic classification failed: {e}")
        return fallback


# ── ID generation ─────────────────────────────────────────────────────────────

def _make_id(content: str, source: str, suffix: str = "") -> str:
    """Deterministic SHA-1 based chunk ID."""
    raw = f"{source}::{content}{suffix}"
    return hashlib.sha1(raw.encode()).hexdigest()


# ── Splitter factory ──────────────────────────────────────────────────────────

def _make_splitter(chunk_size: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,           # character-level; swap for tiktoken if needed
        separators=["\n\n", "\n", ". ", " ", ""],
    )


_parent_splitter = _make_splitter(PARENT_CHUNK_SIZE)
_child_splitter  = _make_splitter(CHILD_CHUNK_SIZE)


# ── Core chunking logic ───────────────────────────────────────────────────────

def _build_parent(raw: dict[str, Any]) -> dict[str, Any]:
    """Wrap a raw parser chunk as a parent chunk (no further splitting)."""
    content = raw["content"]
    meta    = raw["metadata"]
    pid     = _make_id(content, meta.get("source", ""), suffix=":parent")

    topic = meta.get("document_topic") or _classify_topic(content, meta.get("section", ""))

    return {
        "id": pid,
        "content": content,
        "metadata": {
            **meta,
            "document_topic": topic,
            "parent_id": None,
            "chunk_type": "parent",
        },
    }


def _build_children(parent: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a parent chunk into child chunks, each pointing back to its parent."""
    pid     = parent["id"]
    meta    = parent["metadata"]
    content = parent["content"]

    texts = _child_splitter.split_text(content)
    children = []

    for i, text in enumerate(texts):
        if not text.strip():
            continue
        cid = _make_id(text, meta.get("source", ""), suffix=f":child:{i}")
        children.append({
            "id": cid,
            "content": text,
            "metadata": {
                **meta,
                "parent_id": pid,
                "chunk_type": "child",
            },
        })

    return children


# ── Public API ────────────────────────────────────────────────────────────────

@track("chunker")
def chunk_documents(
    raw_chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Main entry point.

    Args:
        raw_chunks: Combined output from any parser (pdf / spreadsheet / markdown).

    Returns:
        (parents, children) — two parallel lists ready for dual indexing.
    """
    parents:  list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []

    for raw in raw_chunks:
        content = raw.get("content", "").strip()
        if not content:
            continue

        # --- Parent pass ---
        # If the raw block is already ≤ PARENT_CHUNK_SIZE, keep it whole.
        # If larger, split into parent-sized segments first.
        parent_texts = (
            [content]
            if len(content) <= PARENT_CHUNK_SIZE
            else _parent_splitter.split_text(content)
        )

        for pt in parent_texts:
            if not pt.strip():
                continue
            raw_copy = {**raw, "content": pt}
            parent   = _build_parent(raw_copy)
            kids     = _build_children(parent)

            parents.append(parent)
            children.extend(kids)

    logger.info(
        f"Chunker: {len(raw_chunks)} raw blocks → "
        f"{len(parents)} parents + {len(children)} children"
    )
    return parents, children