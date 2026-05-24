"""
Markdown Parser — extracts structured text blocks from .md / .txt files.

Strategy:
  - Split on ATX headings (# / ## / ###) to honour document structure.
  - Each heading + its following content becomes one block.
  - document_topic is taken from the nearest heading above the block.
  - Date is parsed from:
      1. YAML/TOML frontmatter  (date: / updated:)
      2. Filename pattern        report_2024-06-30.md
      3. Falls back to file last-modified date.

Returns a list of dicts matching the standard chunk schema:
  {
    "content": str,
    "metadata": {
        "file_type": "markdown",
        "source": str,
        "last_modified": str,     # ISO date YYYY-MM-DD
        "section": str,           # nearest heading
        "element_type": str,      # NarrativeText | CodeBlock | ListBlock
        "document_topic": str,    # top-level H1, or filename stem
    }
  }
"""

import os
import re
from datetime import datetime
from pathlib import Path

from loguru import logger

from evaluation.latency_tracker import track


# ── helpers ──────────────────────────────────────────────────────────────────

def _last_modified(path: Path) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(?P<yaml>.+?)\n---\s*\n|"
    r"^\+\+\+\s*\n(?P<toml>.+?)\n\+\+\+\s*\n",
    re.DOTALL,
)
_DATE_FIELD_RE = re.compile(
    r"^(?:date|updated|created)\s*[:=]\s*['\"]?(\d{4}-\d{2}-\d{2})",
    re.MULTILINE | re.IGNORECASE,
)
_FILENAME_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_LIST_LINE_RE = re.compile(r"^\s*[-*+]\s+|^\s*\d+\.\s+", re.MULTILINE)


def _strip_frontmatter(text: str) -> tuple[str, str | None]:
    """
    Remove YAML/TOML frontmatter and return (clean_text, date_str | None).
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return text, None

    fm_body = m.group("yaml") or m.group("toml") or ""
    clean = text[m.end():]

    date_m = _DATE_FIELD_RE.search(fm_body)
    date_str = date_m.group(1) if date_m else None
    return clean, date_str


def _date_from_filename(path: Path) -> str | None:
    m = _FILENAME_DATE_RE.search(path.stem)
    return m.group(1) if m else None


def _element_type(block: str) -> str:
    """Classify a text block as CodeBlock, ListBlock, or NarrativeText."""
    stripped = block.strip()
    if stripped.startswith("```"):
        return "CodeBlock"
    if _LIST_LINE_RE.match(stripped):
        return "ListBlock"
    return "NarrativeText"


# ── section splitter ─────────────────────────────────────────────────────────

def _split_sections(text: str) -> list[tuple[str, str, int]]:
    """
    Split markdown into (heading_title, body, heading_level) tuples.
    Content before the first heading is yielded under an empty heading.
    """
    sections: list[tuple[str, str, int]] = []
    heading_matches = list(_HEADING_RE.finditer(text))

    if not heading_matches:
        # No headings — treat entire document as one section
        return [("", text.strip(), 0)]

    # Text before first heading
    preamble = text[: heading_matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble, 0))

    for i, match in enumerate(heading_matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body, level))

    return sections


# ── sub-block splitter ────────────────────────────────────────────────────────

def _split_blocks(body: str) -> list[str]:
    """
    Within a section body, split on blank lines so code fences and
    narrative paragraphs become separate indexable blocks.
    Keeps code fences intact.
    """
    # Temporarily replace code fences with placeholders
    fences: list[str] = []

    def _save_fence(m: re.Match) -> str:
        fences.append(m.group(0))
        return f"\x00FENCE{len(fences) - 1}\x00"

    protected = _CODE_FENCE_RE.sub(_save_fence, body)

    # Split on 2+ blank lines
    raw_blocks = re.split(r"\n{2,}", protected)

    # Restore fences and filter empties
    blocks = []
    for b in raw_blocks:
        restored = re.sub(
            r"\x00FENCE(\d+)\x00",
            lambda m: fences[int(m.group(1))],
            b,
        ).strip()
        if restored and len(restored) >= 20:
            blocks.append(restored)

    return blocks if blocks else [body.strip()]


# ── main entry point ──────────────────────────────────────────────────────────

@track("markdown_parser")
def parse_markdown(file_path: str | Path) -> list[dict]:
    """
    Main entry point. Returns structured chunks from a Markdown or text file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    logger.info(f"Parsing markdown: {path.name}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    text, fm_date = _strip_frontmatter(raw)

    # Resolve publication date (priority: frontmatter > filename > mtime)
    pub_date = (
        fm_date
        or _date_from_filename(path)
        or _last_modified(path)
    )

    sections = _split_sections(text)

    # The document topic is the first H1 found, falling back to filename stem
    document_topic = next(
        (title for title, _, level in sections if level == 1 and title),
        path.stem.replace("-", " ").replace("_", " ").title(),
    )

    chunks: list[dict] = []
    current_section = document_topic

    for heading, body, level in sections:
        if heading:
            current_section = heading

        if not body:
            continue

        for block in _split_blocks(body):
            chunks.append({
                "content": block,
                "metadata": {
                    "file_type": "markdown",
                    "source": path.name,
                    "last_modified": pub_date,
                    "section": current_section,
                    "element_type": _element_type(block),
                    "document_topic": document_topic,
                },
            })

    logger.info(f"Markdown '{path.name}' → {len(chunks)} blocks across {len(sections)} sections")
    return chunks