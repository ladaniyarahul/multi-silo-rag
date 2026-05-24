"""
PDF Parser — extracts structured text + tables from PDFs.

Strategy:
  - unstructured: section-aware text extraction (headers, lists, narrative)
  - pdfplumber:   clean table extraction (converts to semantic NL rows)

Returns a list of dicts:
  {
    "content": str,
    "metadata": {
        "file_type": "pdf",
        "source": str,          # filename
        "last_modified": str,   # ISO date
        "section": str,         # nearest header above this block
        "element_type": str,    # NarrativeText | Table | Title | etc.
    }
  }
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pdfplumber
from loguru import logger
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Title, NarrativeText, ListItem

from evaluation.latency_tracker import track


def _last_modified(path: Path) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _table_to_nl(table_data: list[list[Any]]) -> str:
    """Convert a pdfplumber table (list of rows) into NL sentences."""
    if not table_data or len(table_data) < 2:
        return ""

    headers = [str(h).strip() if h else f"Col{i}" for i, h in enumerate(table_data[0])]
    sentences = []

    for row_idx, row in enumerate(table_data[1:], start=1):
        parts = []
        for col_idx, cell in enumerate(row):
            val = str(cell).strip() if cell else ""
            if val:
                parts.append(f"{headers[col_idx]} is '{val}'")
        if parts:
            sentences.append(f"Row {row_idx}: " + ", ".join(parts) + ".")

    return " ".join(sentences)


def _extract_tables_pdfplumber(pdf_path: Path) -> list[dict]:
    """Extract all tables from PDF as NL-converted chunks."""
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    nl = _table_to_nl(table)
                    if nl:
                        results.append({
                            "content": nl,
                            "metadata": {
                                "file_type": "pdf",
                                "source": pdf_path.name,
                                "last_modified": _last_modified(pdf_path),
                                "section": f"Page {page_num} Table {t_idx + 1}",
                                "element_type": "Table",
                            }
                        })
    except Exception as e:
        logger.warning(f"pdfplumber table extraction failed for {pdf_path.name}: {e}")
    return results


def _extract_text_unstructured(pdf_path: Path) -> list[dict]:
    """Extract narrative text and headings using unstructured."""
    results = []
    current_section = "Introduction"

    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",
            infer_table_structure=False,   # tables handled by pdfplumber
        )
    except Exception as e:
        logger.error(f"unstructured failed for {pdf_path.name}: {e}")
        return results

    for el in elements:
        el_type = type(el).__name__

        # Track current section from Title elements
        if isinstance(el, Title):
            current_section = str(el).strip()
            continue

        text = str(el).strip()
        if not text or len(text) < 20:
            continue

        # Skip elements that are just numbers or page markers
        if re.match(r"^\d+$", text):
            continue

        results.append({
            "content": text,
            "metadata": {
                "file_type": "pdf",
                "source": pdf_path.name,
                "last_modified": _last_modified(pdf_path),
                "section": current_section,
                "element_type": el_type,
            }
        })

    return results


@track("pdf_parser")
def parse_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Main entry point. Returns combined text + table chunks from a PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info(f"Parsing PDF: {path.name}")

    text_chunks = _extract_text_unstructured(path)
    table_chunks = _extract_tables_pdfplumber(path)

    all_chunks = text_chunks + table_chunks
    logger.info(f"PDF '{path.name}' → {len(text_chunks)} text + {len(table_chunks)} table blocks")

    return all_chunks