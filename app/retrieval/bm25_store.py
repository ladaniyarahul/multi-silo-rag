"""
BM25 Store — sparse keyword index for hybrid retrieval.

Uses rank_bm25 (BM25Okapi variant) held in memory during a server session.
The index is serialised to disk as a pickle after every ingestion run so
it survives restarts without reprocessing documents.

Index schema (pickled):
    {
        "corpus_ids":   list[str],    # chunk IDs parallel to tokenised corpus
        "corpus_texts": list[str],    # raw content strings (for payload return)
        "corpus_meta":  list[dict],   # metadata dicts parallel to corpus
        "bm25":         BM25Okapi,    # the live index object
    }

Public API:
    build_index(children)               -> None   (build + persist)
    load_index()                        -> None   (load from disk)
    bm25_search(query, top_k)           -> list[ScoredChunk]
    index_info()                        -> dict
    clear_index()                       -> None
"""

import os
import pickle
import re
import string
from pathlib import Path
from typing import Any

from loguru import logger

from evaluation.latency_tracker import track
from retrieval.qdrant_store import ScoredChunk   # reuse same dataclass

# ── Config ────────────────────────────────────────────────────────────────────

BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", ".cache/bm25_index.pkl"))
BM25_K1         = float(os.getenv("BM25_K1", "1.5"))   # term frequency saturation
BM25_B          = float(os.getenv("BM25_B",  "0.75"))  # length normalisation

# ── In-memory state ───────────────────────────────────────────────────────────

_state: dict[str, Any] | None = None   # populated by build_index() or load_index()


# ── Tokeniser ─────────────────────────────────────────────────────────────────

_PUNCT_RE   = re.compile(r"[" + re.escape(string.punctuation) + r"]")
_WHITESPACE = re.compile(r"\s+")

# Minimal English stopword list — keeps the index lean without NLTK dependency
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it", "its",
    "as", "if", "not", "no", "so", "than", "then", "into", "up", "out",
    "about", "which", "who", "what", "how", "when", "where", "there",
})


def _tokenise(text: str) -> list[str]:
    """
    Lowercase → strip punctuation → split → remove stopwords + single chars.
    Intentionally simple: no stemming, no lemmatisation — keeps it fast and
    predictable for exact-keyword matching use cases.
    """
    lowered = text.lower()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    tokens = _WHITESPACE.split(no_punct.strip())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


# ── Build & persist ───────────────────────────────────────────────────────────

@track("bm25_build")
def build_index(children: list[dict[str, Any]]) -> None:
    """
    Build a BM25 index from child chunks and persist it to disk.

    Args:
        children: list of child chunk dicts from chunker.py.
                  Each must have 'id', 'content', and 'metadata' keys.
    """
    global _state

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("rank-bm25 required. Run: pip install rank-bm25")

    if not children:
        logger.warning("build_index called with empty children list — skipping.")
        return

    logger.info(f"Building BM25 index over {len(children)} child chunks…")

    corpus_ids:   list[str]        = []
    corpus_texts: list[str]        = []
    corpus_meta:  list[dict]       = []
    tokenised:    list[list[str]]  = []

    for chunk in children:
        content = chunk.get("content", "").strip()
        if not content:
            continue
        corpus_ids.append(chunk.get("id", ""))
        corpus_texts.append(content)
        corpus_meta.append(chunk.get("metadata", {}))
        tokenised.append(_tokenise(content))

    bm25 = BM25Okapi(tokenised, k1=BM25_K1, b=BM25_B)

    _state = {
        "corpus_ids":   corpus_ids,
        "corpus_texts": corpus_texts,
        "corpus_meta":  corpus_meta,
        "bm25":         bm25,
    }

    _persist()
    logger.info(f"BM25 index built: {len(corpus_ids)} documents, "
                f"avg tokens/doc: "
                f"{sum(len(t) for t in tokenised) // max(len(tokenised), 1)}")


def _persist() -> None:
    """Serialise current _state to disk."""
    if _state is None:
        return
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with BM25_INDEX_PATH.open("wb") as fh:
            pickle.dump(_state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        size_kb = BM25_INDEX_PATH.stat().st_size // 1024
        logger.info(f"BM25 index persisted to '{BM25_INDEX_PATH}' ({size_kb} KB)")
    except Exception as e:
        logger.error(f"Failed to persist BM25 index: {e}")


# ── Load ──────────────────────────────────────────────────────────────────────

def load_index(path: Path | None = None) -> None:
    """
    Load a previously persisted BM25 index from disk into memory.
    Called once at API startup if the index file exists.

    Args:
        path: Override the default BM25_INDEX_PATH.
    """
    global _state
    target = path or BM25_INDEX_PATH
    if not target.exists():
        logger.warning(f"BM25 index not found at '{target}'. Run ingestion first.")
        return
    try:
        with target.open("rb") as fh:
            _state = pickle.load(fh)
        n = len(_state.get("corpus_ids", []))
        logger.info(f"BM25 index loaded from '{target}' ({n} documents)")
    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")
        _state = None


def _ensure_loaded() -> None:
    """Auto-load from disk if the in-memory state is empty."""
    global _state
    if _state is None:
        load_index()
    if _state is None:
        raise RuntimeError(
            "BM25 index is not loaded. "
            "Run ingestion first or call load_index() explicitly."
        )


# ── Search ────────────────────────────────────────────────────────────────────

@track("bm25_search")
def bm25_search(
    query: str,
    top_k: int = 20,
) -> list[ScoredChunk]:
    """
    Keyword-frequency search using the in-memory BM25 index.

    Args:
        query: Raw or HyDE-expanded query string.
        top_k: Number of results to return.

    Returns:
        List of ScoredChunk sorted by BM25 score descending.
        Scores are normalised to [0, 1] relative to the top result.
    """
    _ensure_loaded()
    assert _state is not None   # narrowing for type checkers

    tokens = _tokenise(query)
    if not tokens:
        logger.warning("BM25 query tokenised to empty list — returning no results.")
        return []

    bm25:         "BM25Okapi" = _state["bm25"]       # noqa: F821
    corpus_ids:   list[str]   = _state["corpus_ids"]
    corpus_texts: list[str]   = _state["corpus_texts"]
    corpus_meta:  list[dict]  = _state["corpus_meta"]

    raw_scores: list[float] = bm25.get_scores(tokens).tolist()

    # Get indices of top_k highest scores
    top_indices = sorted(
        range(len(raw_scores)),
        key=lambda i: raw_scores[i],
        reverse=True,
    )[:top_k]

    # Filter zero-score results (no term overlap)
    top_indices = [i for i in top_indices if raw_scores[i] > 0.0]

    if not top_indices:
        logger.debug("BM25 search returned 0 results with non-zero score.")
        return []

    # Normalise scores to [0, 1]
    max_score = raw_scores[top_indices[0]]
    results = [
        ScoredChunk(
            id=corpus_ids[i],
            score=round(raw_scores[i] / max_score, 6) if max_score > 0 else 0.0,
            content=corpus_texts[i],
            metadata=corpus_meta[i],
        )
        for i in top_indices
    ]

    logger.debug(
        f"BM25 search → {len(results)} hits "
        f"(top score: {results[0].score:.4f}, tokens: {tokens[:6]})"
    )
    return results


# ── Introspection ─────────────────────────────────────────────────────────────

def index_info() -> dict[str, Any]:
    """
    Return a summary of the current index state.
    Used by health checks and the Streamlit metrics tab.
    """
    if _state is None:
        on_disk = BM25_INDEX_PATH.exists()
        return {
            "loaded":     False,
            "on_disk":    on_disk,
            "index_path": str(BM25_INDEX_PATH),
        }

    corpus_ids = _state.get("corpus_ids", [])
    tokenised  = [_tokenise(t) for t in _state.get("corpus_texts", [])]
    avg_tokens = (
        sum(len(t) for t in tokenised) // len(tokenised)
        if tokenised else 0
    )
    size_kb = BM25_INDEX_PATH.stat().st_size // 1024 if BM25_INDEX_PATH.exists() else 0

    return {
        "loaded":       True,
        "doc_count":    len(corpus_ids),
        "avg_tokens":   avg_tokens,
        "index_path":   str(BM25_INDEX_PATH),
        "index_size_kb": size_kb,
        "k1":           BM25_K1,
        "b":            BM25_B,
    }


def clear_index(confirm: bool = False) -> None:
    """
    Wipe the in-memory index and delete the pickle file.
    Requires confirm=True as a safety guard.
    """
    global _state
    if not confirm:
        raise RuntimeError("Pass confirm=True to clear the BM25 index.")
    _state = None
    if BM25_INDEX_PATH.exists():
        BM25_INDEX_PATH.unlink()
        logger.warning(f"Deleted BM25 index file: '{BM25_INDEX_PATH}'")
    logger.warning("BM25 in-memory index cleared.")