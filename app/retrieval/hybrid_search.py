"""
Hybrid Search — async parallel dense + sparse retrieval with RRF fusion.

Flow:
    1. HyDE-expand the query (Groq LLM → hypothetical passage).
    2. Embed the expanded query (BGE → 384-dim vector).
    3. Fire dense search (Qdrant) and sparse search (BM25) concurrently.
    4. Merge both result lists with Reciprocal Rank Fusion.
    5. Return the top-K deduplicated ScoredChunk list.

RRF formula:  score(d) = 1/(k + rank_dense) + 1/(k + rank_sparse)
              k = 60  (standard constant that dampens outlier ranks)

Public API:
    async hybrid_search(query, top_k, filters)  -> list[ScoredChunk]
          hybrid_search_sync(...)               -> list[ScoredChunk]   (sync wrapper)
"""

import asyncio
import os
from typing import Any

from loguru import logger

from evaluation.latency_tracker import track
from retrieval.qdrant_store     import ScoredChunk, search   as qdrant_search
from retrieval.bm25_store       import bm25_search
from retrieval.embedder         import embed_query
from retrieval.hyde             import rewrite_query

# ── Config ────────────────────────────────────────────────────────────────────

RRF_K             = int(os.getenv("RRF_K",              "60"))
DENSE_TOP_K       = int(os.getenv("DENSE_TOP_K",        "20"))
SPARSE_TOP_K      = int(os.getenv("SPARSE_TOP_K",       "20"))
HYBRID_TOP_K      = int(os.getenv("TOP_K_RERANK",       "10"))  # output to reranker
HYDE_ENABLED      = os.getenv("HYDE_ENABLED", "true").lower() != "false"


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(
    dense_hits:  list[ScoredChunk],
    sparse_hits: list[ScoredChunk],
    k: int = RRF_K,
    top_k: int = HYBRID_TOP_K,
) -> list[ScoredChunk]:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.

    Each document receives a score from each list it appears in:
        score += 1 / (k + rank)
    Ranks are 1-indexed. Documents absent from a list contribute 0.

    Args:
        dense_hits:  Ordered results from Qdrant (rank 1 = best).
        sparse_hits: Ordered results from BM25  (rank 1 = best).
        k:           RRF constant (default 60).
        top_k:       Number of merged results to return.

    Returns:
        Deduplicated list of ScoredChunk sorted by RRF score descending.
    """
    rrf_scores: dict[str, float]      = {}
    chunks_by_id: dict[str, ScoredChunk] = {}

    # Accumulate RRF scores from dense list
    for rank, chunk in enumerate(dense_hits, start=1):
        rrf_scores[chunk.id]  = rrf_scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        chunks_by_id[chunk.id] = chunk

    # Accumulate RRF scores from sparse list
    for rank, chunk in enumerate(sparse_hits, start=1):
        rrf_scores[chunk.id]  = rrf_scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        if chunk.id not in chunks_by_id:
            chunks_by_id[chunk.id] = chunk

    # Sort by fused score and return top_k
    sorted_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]

    fused = []
    for doc_id in sorted_ids:
        chunk = chunks_by_id[doc_id]
        fused.append(ScoredChunk(
            id=chunk.id,
            score=round(rrf_scores[doc_id], 8),
            content=chunk.content,
            metadata=chunk.metadata,
        ))

    logger.debug(
        f"RRF fusion: {len(dense_hits)} dense + {len(sparse_hits)} sparse "
        f"→ {len(fused)} merged (k={k})"
    )
    return fused


# ── Async retrieval helpers ───────────────────────────────────────────────────

async def _dense_search_async(
    query_vector: "np.ndarray",   # noqa: F821
    top_k: int,
    filters: dict[str, Any] | None,
) -> list[ScoredChunk]:
    """Run Qdrant dense search in a thread so it doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: qdrant_search(query_vector, top_k=top_k, filters=filters)
    )


async def _sparse_search_async(
    query: str,
    top_k: int,
) -> list[ScoredChunk]:
    """Run BM25 sparse search in a thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: bm25_search(query, top_k=top_k)
    )


# ── Query preparation ─────────────────────────────────────────────────────────

def _prepare_query(query: str) -> tuple[str, "np.ndarray"]:   # noqa: F821
    """
    Optionally HyDE-expand then embed the query.

    Returns:
        (expanded_query_text, query_vector)
        expanded_query_text is used for BM25 (token-level match).
        query_vector is used for Qdrant (semantic match).
    """
    expanded = rewrite_query(query) if HYDE_ENABLED else query
    vector   = embed_query(expanded)
    return expanded, vector


# ── Public API ────────────────────────────────────────────────────────────────

@track("hybrid_search")
async def hybrid_search(
    query:   str,
    top_k:   int = HYBRID_TOP_K,
    filters: dict[str, Any] | None = None,
) -> list[ScoredChunk]:
    """
    Async hybrid search: HyDE → embed → dense ∥ sparse → RRF fusion.

    Args:
        query:   Raw user query string.
        top_k:   Number of fused results to return (fed to reranker).
        filters: Optional metadata filter forwarded to Qdrant only
                 e.g. {"file_type": "pdf"}.

    Returns:
        Deduplicated, RRF-fused list of ScoredChunk, length ≤ top_k.
    """
    if not query or not query.strip():
        raise ValueError("hybrid_search received an empty query.")

    query = query.strip()
    logger.info(f"Hybrid search: '{query[:80]}'")

    # Step 1 — HyDE expansion + embedding (sync, in calling thread)
    loop = asyncio.get_running_loop()
    expanded, query_vector = await loop.run_in_executor(
        None, lambda: _prepare_query(query)
    )

    # Step 2 — fire dense and sparse concurrently
    dense_hits, sparse_hits = await asyncio.gather(
        _dense_search_async(query_vector, top_k=DENSE_TOP_K, filters=filters),
        _sparse_search_async(expanded,    top_k=SPARSE_TOP_K),
    )

    logger.debug(f"Dense hits: {len(dense_hits)}  Sparse hits: {len(sparse_hits)}")

    # Step 3 — RRF fusion
    fused = _rrf_fuse(dense_hits, sparse_hits, k=RRF_K, top_k=top_k)

    if not fused:
        logger.warning("Hybrid search returned 0 results after RRF fusion.")

    return fused


def hybrid_search_sync(
    query:   str,
    top_k:   int = HYBRID_TOP_K,
    filters: dict[str, Any] | None = None,
) -> list[ScoredChunk]:
    """
    Synchronous wrapper around hybrid_search().
    Useful for scripts, tests, and non-async call sites.
    """
    try:
        loop = asyncio.get_running_loop()
        # Already inside an event loop (e.g. Jupyter) — use run_until_complete
        # via a new thread to avoid "cannot run nested event loop" errors.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                hybrid_search(query, top_k=top_k, filters=filters),
            )
            return future.result()
    except RuntimeError:
        # No running event loop — safe to use asyncio.run directly
        return asyncio.run(hybrid_search(query, top_k=top_k, filters=filters))


# ── RRF unit helper (exported for tests) ─────────────────────────────────────

def rrf_fuse(
    dense_hits:  list[ScoredChunk],
    sparse_hits: list[ScoredChunk],
    k: int = RRF_K,
    top_k: int = HYBRID_TOP_K,
) -> list[ScoredChunk]:
    """Public alias for _rrf_fuse — used in unit tests."""
    return _rrf_fuse(dense_hits, sparse_hits, k=k, top_k=top_k)