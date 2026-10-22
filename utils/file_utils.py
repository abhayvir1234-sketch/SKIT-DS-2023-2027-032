"""
utils/file_utils.py — Safe filesystem and path utilities.

Author: Karam Singh Bana — Backend integration / file management.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output directory management
# ---------------------------------------------------------------------------

def ensure_output_dir(path: Path | str) -> Path:
    """Create *path* and all parents if they do not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_output_filename(prefix: str = "plate", ext: str = ".jpg") -> str:
    """Generate a unique filename using a timestamp and UUID fragment.

    Args:
        prefix: Filename prefix (e.g. "plate", "annotated").
        ext: File extension including the leading dot.

    Returns:
        A filename string like "plate_20261022_191200_a3f2.jpg".
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts}_{uid}{ext}"


# ---------------------------------------------------------------------------
# JSON / CSV helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Path | str) -> None:
    """Serialize *data* to a JSON file at *path*.

    Args:
        data: JSON-serialisable Python object.
        path: Destination file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    logger.debug("Saved JSON to %s", p)


def save_csv(rows: List[Dict[str, Any]], path: Path | str) -> None:
    """Write a list of dicts to a CSV file.

    Args:
        rows: List of dictionaries (each dict is a row; keys are column names).
        path: Destination file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.debug("Saved CSV to %s", p)


def rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """Convert a list of dicts to CSV-encoded bytes (for Streamlit download).

    Args:
        rows: List of dicts representing rows.

    Returns:
        UTF-8 encoded CSV bytes.
    """
    import io

    if not rows:
        return b""
    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def validate_upload(
    filename: str,
    file_bytes: bytes,
    allowed_extensions: frozenset[str],
    max_bytes: int,
) -> Optional[str]:
    """Validate an uploaded file and return an error message or None.

    Args:
        filename: Original filename from the uploader.
        file_bytes: Raw bytes of the uploaded file.
        allowed_extensions: Set of permitted lowercase extensions (e.g. {".jpg"}).
        max_bytes: Maximum allowed file size in bytes.

    Returns:
        A human-readable error string, or *None* if the file is valid.
    """
    if not filename:
        return "No filename provided."

    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        allowed_str = ", ".join(sorted(allowed_extensions))
        return f"Unsupported file type '{suffix}'. Allowed: {allowed_str}"

    if len(file_bytes) == 0:
        return "The uploaded file is empty."

    if len(file_bytes) > max_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        return f"File is too large ({size_mb:.1f} MB). Limit is {limit_mb:.0f} MB."

    return None
