"""
Eval Dataset — ground-truth Q&A pair manager for RAGAS evaluation.

Each entry in eval_dataset.jsonl contains:
    {
        "question":        str,   # natural-language query
        "ground_truth":    str,   # expected answer
        "source_document": str,   # filename the answer lives in
        "context_keywords": list[str]  # optional — key terms that must appear in retrieved context
    }

Commands:
    python -m evaluation.eval_dataset --list
    python -m evaluation.eval_dataset --add
    python -m evaluation.eval_dataset --validate
    python -m evaluation.eval_dataset --export --out eval_export.json
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

from loguru import logger

# ── Default path (override via env) ──────────────────────────────────────────

import os

DATASET_PATH = Path(os.getenv("EVAL_DATASET_PATH", "evaluation/eval_dataset.jsonl"))

# ── Required fields ───────────────────────────────────────────────────────────

_REQUIRED = {"question", "ground_truth", "source_document"}
_OPTIONAL = {"context_keywords"}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load all entries from the JSONL file. Returns [] if file doesn't exist."""
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {lineno}: {e}")
    return entries


def _save(entries: list[dict[str, Any]], path: Path = DATASET_PATH) -> None:
    """Overwrite the JSONL file with *entries*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(entries)} entries to '{path}'")


def _append(entry: dict[str, Any], path: Path = DATASET_PATH) -> None:
    """Append a single entry without rewriting the whole file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_entry(entry: dict[str, Any], idx: int) -> list[str]:
    """Return a list of error strings for one entry (empty = valid)."""
    errors: list[str] = []

    for field in _REQUIRED:
        if field not in entry:
            errors.append(f"[{idx}] Missing required field: '{field}'")
        elif not str(entry[field]).strip():
            errors.append(f"[{idx}] Field '{field}' is empty")

    if "context_keywords" in entry:
        kw = entry["context_keywords"]
        if not isinstance(kw, list):
            errors.append(f"[{idx}] 'context_keywords' must be a list, got {type(kw).__name__}")
        elif not all(isinstance(k, str) for k in kw):
            errors.append(f"[{idx}] All 'context_keywords' must be strings")

    unknown = set(entry.keys()) - _REQUIRED - _OPTIONAL
    if unknown:
        errors.append(f"[{idx}] Unknown fields: {sorted(unknown)}")

    return errors


def validate_dataset(path: Path = DATASET_PATH) -> tuple[bool, list[str]]:
    """
    Validate every entry in the dataset.

    Returns:
        (ok, errors) — ok is True iff there are zero errors.
    """
    entries = _load(path)
    if not entries:
        return False, [f"Dataset is empty or not found at '{path}'"]

    all_errors: list[str] = []
    questions_seen: set[str] = set()

    for i, entry in enumerate(entries):
        all_errors.extend(_validate_entry(entry, i))

        # Duplicate question check
        q = entry.get("question", "").strip().lower()
        if q in questions_seen:
            all_errors.append(f"[{i}] Duplicate question: '{entry.get('question')}'")
        questions_seen.add(q)

    return len(all_errors) == 0, all_errors


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """
    Load and validate the dataset.  Raises ValueError if validation fails.
    Returns the list of valid entries.
    """
    ok, errors = validate_dataset(path)
    if not ok:
        joined = "\n  ".join(errors)
        raise ValueError(f"Dataset validation failed:\n  {joined}")
    return _load(path)


def add_entry(
    question: str,
    ground_truth: str,
    source_document: str,
    context_keywords: list[str] | None = None,
    path: Path = DATASET_PATH,
) -> dict[str, Any]:
    """
    Add a single Q&A entry to the dataset.
    Raises ValueError if the entry is invalid or the question already exists.
    """
    entry: dict[str, Any] = {
        "question":        question.strip(),
        "ground_truth":    ground_truth.strip(),
        "source_document": source_document.strip(),
    }
    if context_keywords:
        entry["context_keywords"] = [k.strip() for k in context_keywords if k.strip()]

    errors = _validate_entry(entry, -1)
    if errors:
        raise ValueError(f"Invalid entry: {errors}")

    # Duplicate guard
    existing = _load(path)
    existing_qs = {e["question"].strip().lower() for e in existing}
    if question.strip().lower() in existing_qs:
        raise ValueError(f"Question already exists: '{question}'")

    _append(entry, path)
    logger.info(f"Added entry #{len(existing) + 1}: '{question[:60]}...'")
    return entry


def get_questions(path: Path = DATASET_PATH) -> list[str]:
    """Return just the question strings — useful for batch RAGAS eval."""
    return [e["question"] for e in _load(path)]


def get_ground_truths(path: Path = DATASET_PATH) -> list[str]:
    """Return just the ground-truth strings — parallel to get_questions()."""
    return [e["ground_truth"] for e in _load(path)]


def export_dataset(
    out_path: Path,
    path: Path = DATASET_PATH,
) -> None:
    """Export dataset as a pretty-printed JSON file (e.g. for RAGAS)."""
    entries = _load(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    logger.info(f"Exported {len(entries)} entries to '{out_path}'")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cmd_list(path: Path) -> None:
    entries = _load(path)
    if not entries:
        print(f"No entries found in '{path}'.")
        return
    print(f"\n{'#':<4}  {'Question':<55}  Source")
    print("─" * 80)
    for i, e in enumerate(entries, start=1):
        q = textwrap.shorten(e.get("question", ""), width=54, placeholder="…")
        src = e.get("source_document", "")
        print(f"{i:<4}  {q:<55}  {src}")
    print(f"\nTotal: {len(entries)} entries\n")


def _cmd_add(path: Path) -> None:
    print("\nAdd a new evaluation entry (Ctrl-C to cancel)\n")
    try:
        question        = input("Question        : ").strip()
        ground_truth    = input("Ground truth    : ").strip()
        source_document = input("Source document : ").strip()
        kw_raw          = input("Keywords (comma-separated, optional): ").strip()
        keywords        = [k.strip() for k in kw_raw.split(",") if k.strip()] or None

        entry = add_entry(question, ground_truth, source_document, keywords, path)
        print(f"\n✓ Entry added: {json.dumps(entry, indent=2)}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")


def _cmd_validate(path: Path) -> None:
    ok, errors = validate_dataset(path)
    if ok:
        entries = _load(path)
        print(f"\n✓ Dataset valid — {len(entries)} entries, no issues found.\n")
    else:
        print(f"\n✗ Validation failed ({len(errors)} error(s)):\n")
        for err in errors:
            print(f"  {err}")
        print()
        sys.exit(1)


def _cmd_export(path: Path, out: Path) -> None:
    export_dataset(out, path)
    print(f"\n✓ Exported to '{out}'\n")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the RAGAS ground-truth evaluation dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH,
                        help="Path to the JSONL dataset file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list",     action="store_true", help="List all entries.")
    group.add_argument("--add",      action="store_true", help="Interactively add an entry.")
    group.add_argument("--validate", action="store_true", help="Validate all entries.")
    group.add_argument("--export",   action="store_true", help="Export to JSON.")
    parser.add_argument("--out", type=Path, default=Path("eval_export.json"),
                        help="Output path for --export.")
    args = parser.parse_args()

    if args.list:
        _cmd_list(args.dataset)
    elif args.add:
        _cmd_add(args.dataset)
    elif args.validate:
        _cmd_validate(args.dataset)
    elif args.export:
        _cmd_export(args.dataset, args.out)


if __name__ == "__main__":
    _cli()