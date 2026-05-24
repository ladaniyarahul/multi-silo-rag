"""
Ingestion Pipeline — async orchestrator for the full ingestion flow.

Responsibilities:
  1. Discover all supported files under a given directory.
  2. Run the three parsers concurrently (asyncio + ThreadPoolExecutor,
     because all parsers are CPU-bound / sync).
  3. Pass combined raw chunks through the parent-child chunker.
  4. Return (parents, children) ready for dual indexing.

Supported extensions:
    PDF     : .pdf
    Table   : .csv  .xlsx  .xls
    Markdown: .md   .txt   .rst

Usage (CLI):
    python -m ingestion.pipeline --dir ./data

Usage (import):
    from ingestion.pipeline import run_ingestion
    parents, children = await run_ingestion(Path("./data"))
"""

import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

from ingestions.pdf_parser         import parse_pdf
from ingestions.spreadsheet_parser import parse_spreadsheet
from ingestions.markdown_parser    import parse_markdown
from ingestions.chunker            import chunk_documents
from evaluation.latency_tracker   import track, reset, get_current, log_to_file


# ── Extension routing ─────────────────────────────────────────────────────────

_PARSER_MAP: dict[str, Any] = {
    ".pdf":  parse_pdf,
    ".csv":  parse_spreadsheet,
    ".xlsx": parse_spreadsheet,
    ".xls":  parse_spreadsheet,
    ".md":   parse_markdown,
    ".txt":  parse_markdown,
    ".rst":  parse_markdown,
}


def _discover_files(directory: Path) -> list[Path]:
    """Recursively find all supported files under *directory*."""
    found = []
    for ext in _PARSER_MAP:
        found.extend(directory.rglob(f"*{ext}"))
    found = sorted(set(found))
    logger.info(f"Discovered {len(found)} file(s) in '{directory}'")
    return found


# ── Sync parse worker (runs in thread pool) ───────────────────────────────────

def _parse_file(path: Path) -> list[dict]:
    """Dispatch to the correct parser and return raw chunks."""
    parser = _PARSER_MAP.get(path.suffix.lower())
    if parser is None:
        logger.warning(f"No parser for {path.name} — skipping.")
        return []
    try:
        return parser(path)
    except Exception as e:
        logger.error(f"Parser failed for '{path.name}': {e}")
        return []


# ── Async orchestration ───────────────────────────────────────────────────────

async def _parse_all_async(
    files: list[Path],
    max_workers: int = 4,
) -> list[dict]:
    """
    Run all parsers concurrently using a thread pool.
    Returns the combined flat list of raw chunks.
    """
    loop = asyncio.get_running_loop()
    raw_all: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            loop.run_in_executor(executor, _parse_file, path): path
            for path in files
        }
        for coro in asyncio.as_completed(futures.keys()):
            try:
                chunks = await coro
                raw_all.extend(chunks)
            except Exception as e:
                logger.error(f"Unexpected error during async parse: {e}")

    logger.info(f"All parsers done — {len(raw_all)} total raw chunks collected")
    return raw_all


# ── Public API ────────────────────────────────────────────────────────────────

@track("pipeline_total")
async def run_ingestion(
    directory: Path,
    max_workers: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Full ingestion pipeline.

    Args:
        directory:   Path to the folder containing source documents.
        max_workers: Thread pool size for parallel parsing.

    Returns:
        (parents, children) — chunked documents ready for dual indexing.
    """
    reset()   # clear latency counters for this run

    files = _discover_files(directory)
    if not files:
        logger.warning(f"No supported files found in '{directory}'.")
        return [], []

    raw_chunks = await _parse_all_async(files, max_workers=max_workers)

    if not raw_chunks:
        logger.warning("Parsers returned no chunks — check input files.")
        return [], []

    parents, children = chunk_documents(raw_chunks)

    # Log latency snapshot for trend analysis
    log_to_file("logs/ingestion_latency.jsonl")
    logger.info(
        f"Ingestion complete — {len(parents)} parents, {len(children)} children. "
        f"Latency: {get_current()}"
    )

    return parents, children


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(
    parents: list[dict],
    children: list[dict],
) -> None:
    from collections import Counter

    p_by_type: Counter = Counter(
        p["metadata"].get("file_type", "unknown") for p in parents
    )
    c_by_type: Counter = Counter(
        c["metadata"].get("file_type", "unknown") for c in children
    )

    print("\n── Ingestion Summary ─────────────────────────────")
    print(f"  Parents  : {len(parents):>6}")
    for ft, n in sorted(p_by_type.items()):
        print(f"    {ft:<12}: {n}")
    print(f"  Children : {len(children):>6}")
    for ft, n in sorted(c_by_type.items()):
        print(f"    {ft:<12}: {n}")
    print("──────────────────────────────────────────────────\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ingestion pipeline against a document directory."
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Path to the directory containing source documents.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel parser threads (default: 4).",
    )
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"Error: '{args.dir}' is not a directory.")
        raise SystemExit(1)

    parents, children = asyncio.run(
        run_ingestion(args.dir, max_workers=args.workers)
    )
    _print_summary(parents, children)


if __name__ == "__main__":
    _cli()