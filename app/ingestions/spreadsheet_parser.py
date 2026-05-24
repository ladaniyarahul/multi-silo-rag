"""
Spreadsheet Parser — converts CSV/XLSX rows into semantic NL sentences.

Why not chunk line-by-line?
  Raw CSV lines lose column context completely.
  "Delayed,Sarah,X,2024-06-30" tells the LLM nothing.
  "Project X has status Delayed, managed by Sarah, due 2024-06-30" is searchable.

Returns a list of dicts matching the standard chunk schema.
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from evaluation.latency_tracker import track


def _last_modified(path: Path) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _row_to_nl(row: pd.Series, row_idx: int) -> str:
    """
    Convert a single DataFrame row into a natural-language sentence.
    Example:
      headers: [Project, Status, Manager, Due Date]
      values:  [X,       Delayed, Sarah,  2024-06-30]
      output:  "Row 2: Project is 'X', Status is 'Delayed',
                Manager is 'Sarah', Due Date is '2024-06-30'."
    """
    parts = []
    for col, val in row.items():
        val_str = str(val).strip()
        if val_str and val_str.lower() not in ("nan", "none", ""):
            parts.append(f"{col} is '{val_str}'")
    if not parts:
        return ""
    return f"Row {row_idx}: " + ", ".join(parts) + "."


def _load_file(path: Path) -> pd.DataFrame:
    """Load CSV or XLSX into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Try UTF-8 first, fall back to latin-1
        try:
            return pd.read_csv(path, dtype=str)
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype=str, encoding="latin-1")
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


@track("spreadsheet_parser")
def parse_spreadsheet(file_path: str | Path) -> list[dict]:
    """
    Main entry point.
    Each row → one NL chunk with full metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Spreadsheet not found: {path}")

    logger.info(f"Parsing spreadsheet: {path.name}")

    df = _load_file(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop completely empty rows
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    last_mod = _last_modified(path)
    file_type = "csv" if path.suffix.lower() == ".csv" else "xlsx"

    chunks = []
    for idx, row in df.iterrows():
        nl = _row_to_nl(row, int(idx) + 2)  # +2: 1-indexed + header row
        if not nl:
            continue
        chunks.append({
            "content": nl,
            "metadata": {
                "file_type": file_type,
                "source": path.name,
                "last_modified": last_mod,
                "section": "Table Data",
                "element_type": "TableRow",
                "row_index": int(idx) + 2,
                "columns": list(df.columns),
            }
        })

    logger.info(f"Spreadsheet '{path.name}' → {len(chunks)} NL row chunks")
    return chunks