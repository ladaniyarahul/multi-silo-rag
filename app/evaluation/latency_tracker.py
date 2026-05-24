"""
Per-component latency tracker.
Usage:
    from evaluation.latency_tracker import track

    @track("pdf_parser")
    async def parse(...): ...

    # or sync
    @track("chunker")
    def chunk(...): ...
"""

import time
import functools
import asyncio
import json
from pathlib import Path
from typing import Callable, Any
from loguru import logger


# In-memory store for the current request
_current: dict[str, float] = {}


def track(component: str):
    """Decorator — works on both sync and async functions."""
    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs) -> Any:
                start = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    return result
                finally:
                    elapsed = round(time.perf_counter() - start, 4)
                    _current[component] = elapsed
                    logger.debug(f"[latency] {component}: {elapsed:.3f}s")
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs) -> Any:
                start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    return result
                finally:
                    elapsed = round(time.perf_counter() - start, 4)
                    _current[component] = elapsed
                    logger.debug(f"[latency] {component}: {elapsed:.3f}s")
            return sync_wrapper
    return decorator


def get_current() -> dict[str, float]:
    """Return latency snapshot for the current request."""
    return dict(_current)


def reset():
    """Clear latency store — call at the start of each request."""
    _current.clear()


def log_to_file(path: str = "logs/latency.jsonl"):
    """Append current snapshot to a JSONL file for trend analysis."""
    record = {"ts": time.time(), **_current}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")