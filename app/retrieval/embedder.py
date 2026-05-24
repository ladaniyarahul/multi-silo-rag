"""
Embedder — BGE sentence embedding with on-disk cache.

Model : BAAI/bge-small-en-v1.5  (384-dim, CPU-friendly, ~130 MB)
Cache : embeddings are stored as numpy .npy files keyed by SHA-1 of the
        input text, so re-ingesting unchanged documents skips recomputation.

Public API:
    embed_texts(texts)          -> np.ndarray  shape (N, 384)
    embed_query(text)           -> np.ndarray  shape (384,)
    warmup()                    -> None        (pre-loads model at startup)

All calls are wrapped with @track so latency is captured automatically.
"""

import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from loguru import logger

from evaluation.latency_tracker import track

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL     = os.getenv("EMBED_MODEL",     "BAAI/bge-small-en-v1.5")
EMBED_CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", ".cache/embeddings"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_DIM        = 384          # fixed for bge-small-en-v1.5

# BGE models expect this instruction prefix on passages (not on queries)
_PASSAGE_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIX   = "Represent this question for searching relevant passages: "

# ── Model singleton ───────────────────────────────────────────────────────────

_model = None          # SentenceTransformer instance, loaded lazily


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Run: pip install sentence-transformers"
            )
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
        logger.info("Embedding model ready.")
    return _model


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(text: str) -> str:
    """Deterministic SHA-1 key for a single text string."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    # Two-level sharding: .cache/embeddings/ab/abcdef1234…npy
    return EMBED_CACHE_DIR / key[:2] / f"{key}.npy"


def _load_cached(key: str) -> np.ndarray | None:
    p = _cache_path(key)
    if p.exists():
        try:
            return np.load(str(p))
        except Exception as e:
            logger.warning(f"Cache read failed for {key}: {e}")
    return None


def _save_cached(key: str, vec: np.ndarray) -> None:
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(str(p), vec)
    except Exception as e:
        logger.warning(f"Cache write failed for {key}: {e}")


def _cache_stats(keys: list[str]) -> tuple[int, int]:
    """Return (hits, misses) for a list of cache keys."""
    hits = sum(1 for k in keys if _cache_path(k).exists())
    return hits, len(keys) - hits


# ── Batch encode (internal) ───────────────────────────────────────────────────

def _encode_batch(texts: list[str], is_query: bool = False) -> np.ndarray:
    """
    Encode a list of texts with the BGE prefix convention.
    Returns float32 array of shape (len(texts), EMBED_DIM).
    """
    prefix = _QUERY_PREFIX if is_query else _PASSAGE_PREFIX
    prefixed = [prefix + t for t in texts]

    model = _get_model()
    vecs  = model.encode(
        prefixed,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,   # cosine similarity → dot product
        show_progress_bar=False,
    )
    return np.array(vecs, dtype=np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

@track("embedder")
def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """
    Embed a list of passage texts, using the on-disk cache where possible.

    Args:
        texts: Any sequence of non-empty strings.

    Returns:
        np.ndarray of shape (len(texts), EMBED_DIM), dtype float32.
    """
    texts = list(texts)
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    keys      = [_cache_key(t) for t in texts]
    hits, misses = _cache_stats(keys)
    logger.debug(f"Embed cache — hits: {hits}, misses: {misses}, total: {len(texts)}")

    results: dict[int, np.ndarray] = {}

    # Load cached vectors
    miss_indices: list[int] = []
    for i, (text, key) in enumerate(zip(texts, keys)):
        cached = _load_cached(key)
        if cached is not None:
            results[i] = cached
        else:
            miss_indices.append(i)

    # Encode cache misses in batches
    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        new_vecs   = _encode_batch(miss_texts, is_query=False)
        for j, idx in enumerate(miss_indices):
            vec = new_vecs[j]
            results[idx] = vec
            _save_cached(keys[idx], vec)

    # Reassemble in original order
    matrix = np.stack([results[i] for i in range(len(texts))], axis=0)
    logger.debug(f"embed_texts → shape {matrix.shape}")
    return matrix


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string (query prefix, no caching — queries are ephemeral).

    Args:
        text: The user query or HyDE-expanded query.

    Returns:
        np.ndarray of shape (EMBED_DIM,), dtype float32.
    """
    if not text or not text.strip():
        raise ValueError("embed_query received an empty string.")

    vecs = _encode_batch([text], is_query=True)
    return vecs[0]


def warmup() -> None:
    """
    Pre-load the model into memory.
    Call this once at API startup to avoid cold-start latency on the first query.
    """
    logger.info("Warming up embedding model…")
    _get_model()
    # Run a tiny encode to JIT-compile the inference path
    _encode_batch(["warmup"], is_query=False)
    logger.info("Embedding model warm.")


def clear_cache(confirm: bool = False) -> int:
    """
    Delete all cached embedding files.

    Args:
        confirm: Safety flag — must be True to actually delete.

    Returns:
        Number of files deleted.
    """
    if not confirm:
        raise RuntimeError("Pass confirm=True to clear the embedding cache.")
    if not EMBED_CACHE_DIR.exists():
        return 0
    files = list(EMBED_CACHE_DIR.rglob("*.npy"))
    for f in files:
        f.unlink(missing_ok=True)
    logger.warning(f"Cleared {len(files)} cached embedding file(s).")
    return len(files)