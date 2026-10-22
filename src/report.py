"""
src/report.py — Recognition result report generation.

Author: Karam Singh Bana (Team Member 2) — Recognition result display and export.

Builds structured JSON and CSV reports from pipeline results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def build_recognition_report(
    filename: str,
    plate_count: int,
    recognized_text: str,
    detector_confidence: Optional[float],
    ocr_confidence: Optional[float],
    validation_passed: bool,
    validation_message: str,
    processing_time_sec: Optional[float] = None,
    error_message: str = "",
) -> Dict[str, Any]:
    """Build a structured recognition report dictionary.

    Args:
        filename: Name of the uploaded image file.
        plate_count: Number of plates detected.
        recognized_text: Best normalized registration text (may be empty).
        detector_confidence: YOLO confidence for the best plate (0–1 or None).
        ocr_confidence: OCR engine confidence (0–1 or None).
        validation_passed: Whether the text passed format validation.
        validation_message: Human-readable validation outcome message.
        processing_time_sec: Time taken by the pipeline in seconds (optional).
        error_message: Error description if the pipeline failed.

    Returns:
        A dictionary ready for JSON serialization or DataFrame construction.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "plates_detected": plate_count,
        "recognized_text": recognized_text,
        "detector_confidence": (
            round(detector_confidence, 4) if detector_confidence is not None else None
        ),
        "ocr_confidence": (
            round(ocr_confidence, 4) if ocr_confidence is not None else None
        ),
        "validation_passed": validation_passed,
        "validation_message": validation_message,
        "processing_time_sec": (
            round(processing_time_sec, 3) if processing_time_sec is not None else None
        ),
        "error": error_message,
    }


def report_to_json_bytes(report: Dict[str, Any]) -> bytes:
    """Serialize a report dict to indented UTF-8 JSON bytes.

    Args:
        report: Report dictionary from :func:`build_recognition_report`.

    Returns:
        UTF-8 encoded JSON bytes suitable for Streamlit download.
    """
    return json.dumps(report, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def report_to_csv_bytes(report: Dict[str, Any]) -> bytes:
    """Serialize a single report dict to CSV bytes.

    Args:
        report: Report dictionary.

    Returns:
        UTF-8 encoded CSV bytes.
    """
    df = pd.DataFrame([report])
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------


def append_history(
    history: List[Dict[str, Any]],
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Append *report* to *history* and return the updated list.

    Args:
        history: Existing session history list.
        report: New report to append.

    Returns:
        Updated history list.
    """
    return history + [report]


def history_to_csv_bytes(history: List[Dict[str, Any]]) -> bytes:
    """Serialize the session history list to CSV bytes.

    Args:
        history: List of report dicts.

    Returns:
        UTF-8 encoded CSV bytes (empty bytes if history is empty).
    """
    if not history:
        return b""
    df = pd.DataFrame(history)
    return df.to_csv(index=False).encode("utf-8")


def history_to_dataframe(history: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert session history to a Pandas DataFrame for display.

    Args:
        history: List of report dicts.

    Returns:
        DataFrame with columns from the report schema.
    """
    if not history:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "filename",
                "plates_detected",
                "recognized_text",
                "detector_confidence",
                "ocr_confidence",
                "validation_passed",
                "error",
            ]
        )
    return pd.DataFrame(history)
