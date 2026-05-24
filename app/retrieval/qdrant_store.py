"""
Qdrant Store — vector DB interface for dense retrieval.

Responsibilities:
    - Create / verify the Qdrant collection on first use.
    - Upsert child chunks (embedding + metadata payload).
    - Store parent chunks in a separate collection for context hydration.
    - Dense cosine-similarity search returning top-K ScoredChunk results.
    - Parent fetch by ID (used after re-ranking to get full context).

Collections:
    <QDRANT_COLLECTION>          child chunks  — searched
    <QDRANT_COLLECTION>_parents  parent chunks — fetched by ID

Public API:
    upsert_children(children, embeddings)
    upsert_parents(parents)
    search(query_vector, top_k)      -> list[ScoredChunk]
    fetch_parents(parent_ids)        -> list[dict]
    collection_info()                -> dict
    delete_collection()
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from evaluation.latency_tracker import track

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_HOST       = os.getenv("QDRANT_HOST",       "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT",   "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_rag")
EMBED_DIM         = 384      # must match embedder output

_PARENTS_COLLECTION = f"{QDRANT_COLLECTION}_parents"

# ── Qdrant client (lazy) ──────────────────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("qdrant-client required. Run: pip install qdrant-client")
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"Qdrant client connected: {QDRANT_HOST}:{QDRANT_PORT}")
    return _client


# ── Collection bootstrap ──────────────────────────────────────────────────────

def _ensure_collection(name: str, vector_size: int) -> None:
    """Create collection if it doesn't already exist."""
    from qdrant_client.models import (
        VectorParams,
        Distance,
    )
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection '{name}' (dim={vector_size})")
    else:
        logger.debug(f"Collection '{name}' already exists.")


def _ensure_parent_collection() -> None:
    """
    Parent collection stores no vectors — only payloads.
    We use a dummy 1-dim vector as Qdrant requires a vector config.
    """
    _ensure_collection(_PARENTS_COLLECTION, vector_size=1)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ScoredChunk:
    """A retrieved child chunk with its similarity score."""
    id:        str
    score:     float
    content:   str
    metadata:  dict[str, Any] = field(default_factory=dict)

    @property
    def parent_id(self) -> str | None:
        return self.metadata.get("parent_id")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_to_point(chunk: dict[str, Any], vector: list[float]):
    """Convert a chunk dict + embedding to a Qdrant PointStruct."""
    from qdrant_client.models import PointStruct
    # Use the chunk's own deterministic ID if available, else generate UUID
    point_id = chunk.get("id") or str(uuid.uuid4())
    # Qdrant point IDs must be UUID or unsigned int; convert hex SHA-1 → UUID
    try:
        uid = str(uuid.UUID(point_id))
    except ValueError:
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))

    payload = {
        "content":  chunk.get("content", ""),
        "chunk_id": chunk.get("id", ""),
        **chunk.get("metadata", {}),
    }
    return PointStruct(id=uid, vector=vector, payload=payload)


def _parent_to_point(parent: dict[str, Any]):
    """Convert a parent chunk to a Qdrant point with dummy vector."""
    from qdrant_client.models import PointStruct
    point_id = parent.get("id") or str(uuid.uuid4())
    try:
        uid = str(uuid.UUID(point_id))
    except ValueError:
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))

    payload = {
        "content":  parent.get("content", ""),
        "chunk_id": parent.get("id", ""),
        **parent.get("metadata", {}),
    }
    return PointStruct(id=uid, vector=[0.0], payload=payload)


def _batch(iterable, size: int):
    """Yield successive fixed-size slices of *iterable*."""
    for i in range(0, len(iterable), size):
        yield iterable[i: i + size]


# ── Public API ────────────────────────────────────────────────────────────────

UPSERT_BATCH = int(os.getenv("QDRANT_UPSERT_BATCH", "128"))


@track("qdrant_upsert_children")
def upsert_children(
    children:   list[dict[str, Any]],
    embeddings: np.ndarray,
) -> int:
    """
    Upsert child chunks and their embeddings into the vector collection.

    Args:
        children:   list of child chunk dicts from chunker.py.
        embeddings: float32 array of shape (len(children), EMBED_DIM).

    Returns:
        Number of points upserted.
    """
    if len(children) != len(embeddings):
        raise ValueError(
            f"children ({len(children)}) and embeddings ({len(embeddings)}) must match."
        )

    _ensure_collection(QDRANT_COLLECTION, EMBED_DIM)
    client = _get_client()

    points = [
        _chunk_to_point(chunk, embeddings[i].tolist())
        for i, chunk in enumerate(children)
    ]

    upserted = 0
    for batch in _batch(points, UPSERT_BATCH):
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        upserted += len(batch)
        logger.debug(f"Upserted {upserted}/{len(points)} child points…")

    logger.info(f"Qdrant: upserted {upserted} child chunks into '{QDRANT_COLLECTION}'")
    return upserted


@track("qdrant_upsert_parents")
def upsert_parents(parents: list[dict[str, Any]]) -> int:
    """
    Store parent chunks (payload-only) for later context hydration.

    Args:
        parents: list of parent chunk dicts from chunker.py.

    Returns:
        Number of points upserted.
    """
    _ensure_parent_collection()
    client = _get_client()

    points = [_parent_to_point(p) for p in parents]
    upserted = 0
    for batch in _batch(points, UPSERT_BATCH):
        client.upsert(collection_name=_PARENTS_COLLECTION, points=batch)
        upserted += len(batch)

    logger.info(f"Qdrant: upserted {upserted} parent chunks into '{_PARENTS_COLLECTION}'")
    return upserted


@track("qdrant_search")
def search(
    query_vector: np.ndarray,
    top_k: int = 20,
    filters: dict[str, Any] | None = None,
) -> list[ScoredChunk]:
    """
    Dense cosine-similarity search over child chunk embeddings.

    Args:
        query_vector: Shape (EMBED_DIM,) float32 — from embedder.embed_query().
        top_k:        Number of candidates to return (default 20).
        filters:      Optional Qdrant filter dict for metadata pre-filtering.
                      e.g. {"file_type": "pdf"} — wrapped into a MatchValue filter.

    Returns:
        List of ScoredChunk sorted by score descending.
    """
    _ensure_collection(QDRANT_COLLECTION, EMBED_DIM)
    client = _get_client()

    qdrant_filter = None
    if filters:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector.tolist(),
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    scored = [
        ScoredChunk(
            id=str(hit.id),
            score=float(hit.score),
            content=hit.payload.get("content", ""),
            metadata={k: v for k, v in hit.payload.items() if k != "content"},
        )
        for hit in results
    ]

    logger.debug(f"Qdrant search → {len(scored)} hits (top score: {scored[0].score:.4f})"
                 if scored else "Qdrant search → 0 hits")
    return scored


@track("qdrant_fetch_parents")
def fetch_parents(parent_ids: list[str]) -> list[dict[str, Any]]:
    """
    Retrieve full parent chunks by their chunk IDs.
    Called after re-ranking to hydrate the LLM context window.

    Args:
        parent_ids: List of parent chunk_id strings (SHA-1 hex from chunker).

    Returns:
        List of parent dicts with 'content' and 'metadata' keys.
        Order matches parent_ids; missing IDs produce None entries.
    """
    if not parent_ids:
        return []

    _ensure_parent_collection()
    client = _get_client()

    # Convert chunk IDs to Qdrant UUIDs (same logic as _parent_to_point)
    id_map: dict[str, str] = {}   # qdrant_uuid → original chunk_id
    qdrant_ids: list[str] = []
    for cid in parent_ids:
        try:
            uid = str(uuid.UUID(cid))
        except ValueError:
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, cid))
        id_map[uid] = cid
        qdrant_ids.append(uid)

    records = client.retrieve(
        collection_name=_PARENTS_COLLECTION,
        ids=qdrant_ids,
        with_payload=True,
    )

    # Index by qdrant UUID for ordered output
    by_uid = {str(r.id): r for r in records}
    parents: list[dict[str, Any]] = []
    for uid in qdrant_ids:
        rec = by_uid.get(uid)
        if rec is None:
            logger.warning(f"Parent not found in Qdrant: chunk_id={id_map.get(uid)}")
            parents.append({"content": "", "metadata": {}})
        else:
            payload = rec.payload or {}
            parents.append({
                "content":  payload.pop("content", ""),
                "metadata": payload,
            })

    return parents


def collection_info() -> dict[str, Any]:
    """
    Return a summary of both collections (point counts, vector config).
    Useful for health checks and the Streamlit metrics tab.
    """
    client = _get_client()
    info: dict[str, Any] = {}

    for name in [QDRANT_COLLECTION, _PARENTS_COLLECTION]:
        try:
            ci = client.get_collection(name)
            info[name] = {
                "status":       ci.status,
                "points_count": ci.points_count,
                "vectors_count": ci.vectors_count,
            }
        except Exception:
            info[name] = {"status": "not_found"}

    return info


def delete_collection(confirm: bool = False) -> None:
    """
    Delete both collections. Requires confirm=True as a safety guard.
    Used by integration tests and full re-ingestion workflows.
    """
    if not confirm:
        raise RuntimeError("Pass confirm=True to delete Qdrant collections.")
    client = _get_client()
    for name in [QDRANT_COLLECTION, _PARENTS_COLLECTION]:
        try:
            client.delete_collection(name)
            logger.warning(f"Deleted Qdrant collection '{name}'")
        except Exception as e:
            logger.warning(f"Could not delete '{name}': {e}")