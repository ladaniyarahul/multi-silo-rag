"""
Reranker — cross-encoder re-ranking with ms-marco-MiniLM.

Takes the top-K RRF-fused ScoredChunks and scores each (query, passage)
pair jointly using a cross-encoder model. This is a second quality gate
that fixes cases where a relevant passage ranked low in both dense and
sparse retrieval — something bi-encoder RRF cannot correct.

Model : cross-encoder/ms-marco-MiniLM-L-6-v2  (~22 MB, fast on CPU)
Input : top-10 RRF chunks + original query
Output: top-5 re-ranked ScoredChunks with updated scores

After re-ranking, fetch_parents() is called to swap each child chunk
for its full 1000-token parent context before passing to the LLM.

Public API:
    rerank(query, chunks, top_k)              -> list[RankedResult]
    rerank_and_fetch(query, chunks, top_k)    -> list[RankedResult]
"""

import os
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from evaluation.latency_tracker import track
from retrieval.qdrant_store     import ScoredChunk, fetch_parents

# ── Config ────────────────────────────────────────────────────────────────────

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
TOP_K_FINAL    = int(os.getenv("TOP_K_FINAL",  "5"))
TOP_K_RERANK   = int(os.getenv("TOP_K_RERANK", "10"))   # max input to reranker

# ── Model singleton ───────────────────────────────────────────────────────────

_cross_encoder = None


def _get_model():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Run: pip install sentence-transformers"
            )
        logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("Cross-encoder ready.")
    return _cross_encoder


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RankedResult:
    """
    A re-ranked retrieval result, optionally hydrated with its parent context.
    """
    # Child chunk fields (from retrieval)
    chunk_id:      str
    rerank_score:  float
    content:       str                       # child chunk text
    metadata:      dict[str, Any] = field(default_factory=dict)

    # Parent hydration (populated by rerank_and_fetch)
    parent_id:     str | None  = None
    parent_content: str        = ""          # full 1000-tok context for LLM
    parent_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_for_llm(self) -> str:
        """Return the richest available context: parent if hydrated, else child."""
        return self.parent_content if self.parent_content else self.content

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")


# ── Core reranking ────────────────────────────────────────────────────────────

@track("reranker")
def rerank(
    query:  str,
    chunks: list[ScoredChunk],
    top_k:  int = TOP_K_FINAL,
) -> list[RankedResult]:
    """
    Score (query, passage) pairs with the cross-encoder and return top-k.

    Args:
        query:  Original user query (NOT the HyDE expansion — we score
                against the real question for precision).
        chunks: RRF-fused ScoredChunks from hybrid_search (≤ TOP_K_RERANK).
        top_k:  Number of results to return after re-ranking.

    Returns:
        List of RankedResult sorted by rerank_score descending.
        parent_content is empty — call rerank_and_fetch() to hydrate.
    """
    if not chunks:
        logger.warning("rerank() called with empty chunks list.")
        return []

    # Limit input to TOP_K_RERANK candidates
    candidates = chunks[:TOP_K_RERANK]

    model = _get_model()
    pairs = [(query, c.content) for c in candidates]

    raw_scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()

    # Pair scores with chunks and sort
    scored = sorted(
        zip(raw_scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = [
        RankedResult(
            chunk_id=chunk.id,
            rerank_score=round(float(score), 6),
            content=chunk.content,
            metadata=chunk.metadata,
            parent_id=chunk.parent_id,
        )
        for score, chunk in scored[:top_k]
    ]

    logger.debug(
        f"Reranker: {len(candidates)} → {len(results)} "
        f"(top score: {results[0].rerank_score:.4f})"
        if results else "Reranker: no results"
    )
    return results


@track("rerank_and_fetch")
def rerank_and_fetch(
    query:  str,
    chunks: list[ScoredChunk],
    top_k:  int = TOP_K_FINAL,
) -> list[RankedResult]:
    """
    Re-rank chunks, then hydrate each result with its full parent context.

    This is the primary entry point called by api/main.py.

    Args:
        query:  Original user query.
        chunks: RRF-fused ScoredChunks from hybrid_search.
        top_k:  Number of final results.

    Returns:
        List of RankedResult with parent_content populated for LLM consumption.
    """
    ranked = rerank(query, chunks, top_k=top_k)

    if not ranked:
        return []

    # Collect parent IDs (some may be None if chunk is already a parent)
    parent_ids = [
        r.parent_id for r in ranked if r.parent_id
    ]

    if not parent_ids:
        logger.debug("No parent IDs to fetch — using child content directly.")
        return ranked

    parents = fetch_parents(parent_ids)

    # Map parent chunk_id → parent dict for O(1) lookup
    parent_map: dict[str, dict[str, Any]] = {}
    for pid, parent in zip(parent_ids, parents):
        parent_map[pid] = parent

    # Hydrate each ranked result
    for result in ranked:
        if result.parent_id and result.parent_id in parent_map:
            p = parent_map[result.parent_id]
            result.parent_content  = p.get("content", "")
            result.parent_metadata = p.get("metadata", {})
        else:
            # Fall back to child content — retrieval still works
            result.parent_content = result.content

    logger.info(
        f"rerank_and_fetch: returned {len(ranked)} results, "
        f"{sum(1 for r in ranked if r.parent_content)} hydrated with parent context"
    )
    return ranked


# ── Context builder (used by api/main.py) ─────────────────────────────────────

def build_context_string(results: list[RankedResult], separator: str = "\n\n---\n\n") -> str:
    """
    Concatenate the context_for_llm fields of all results into a single
    string ready to be injected into the LLM system prompt.

    Args:
        results:   Output of rerank_and_fetch().
        separator: String inserted between each context block.

    Returns:
        Multi-block context string with source citations embedded.
    """
    blocks = []
    for i, r in enumerate(results, start=1):
        header = f"[Source {i}: {r.source} — {r.section}]"
        blocks.append(f"{header}\n{r.context_for_llm}")

    return separator.join(blocks)