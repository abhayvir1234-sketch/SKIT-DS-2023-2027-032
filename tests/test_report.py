"""
tests/test_report.py — Tests for src/report.py

Verifies recognition report generation, JSON/CSV serialization, and
session history utilities.
"""

import json
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_REPORT = {
    "timestamp": "2026-10-22T19:00:00",
    "filename": "test.jpg",
    "plates_detected": 1,
    "recognized_text": "MH12AB1234",
    "detector_confidence": 0.8765,
    "ocr_confidence": 0.9123,
    "validation_passed": True,
    "validation_message": "Accepted (lenient validation mode).",
    "processing_time_sec": 1.234,
    "error": "",
}


# ---------------------------------------------------------------------------
# Tests: build_recognition_report
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_all_keys_present(self):
        from src.report import build_recognition_report
        report = build_recognition_report(
            filename="test.jpg",
            plate_count=1,
            recognized_text="MH12AB1234",
            detector_confidence=0.87,
            ocr_confidence=0.91,
            validation_passed=True,
            validation_message="Accepted.",
            processing_time_sec=1.0,
        )
        for key in ("timestamp", "filename", "plates_detected", "recognized_text",
                    "detector_confidence", "ocr_confidence", "validation_passed",
                    "processing_time_sec", "error"):
            assert key in report, f"Missing key: {key}"

    def test_none_confidence_preserved(self):
        from src.report import build_recognition_report
        report = build_recognition_report(
            filename="no_plate.jpg",
            plate_count=0,
            recognized_text="",
            detector_confidence=None,
            ocr_confidence=None,
            validation_passed=False,
            validation_message="No detection.",
        )
        assert report["detector_confidence"] is None
        assert report["ocr_confidence"] is None

    def test_error_message_stored(self):
        from src.report import build_recognition_report
        report = build_recognition_report(
            filename="bad.jpg",
            plate_count=0,
            recognized_text="",
            detector_confidence=None,
            ocr_confidence=None,
            validation_passed=False,
            validation_message="",
            error_message="Model weights not found.",
        )
        assert "Model weights" in report["error"]


# ---------------------------------------------------------------------------
# Tests: report_to_json_bytes
# ---------------------------------------------------------------------------

class TestReportToJSON:
    def test_output_is_valid_json(self):
        from src.report import report_to_json_bytes
        data = report_to_json_bytes(SAMPLE_REPORT)
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["filename"] == "test.jpg"
        assert parsed["recognized_text"] == "MH12AB1234"

    def test_output_is_bytes(self):
        from src.report import report_to_json_bytes
        assert isinstance(report_to_json_bytes(SAMPLE_REPORT), bytes)


# ---------------------------------------------------------------------------
# Tests: report_to_csv_bytes
# ---------------------------------------------------------------------------

class TestReportToCSV:
    def test_csv_contains_recognized_text(self):
        from src.report import report_to_csv_bytes
        csv_bytes = report_to_csv_bytes(SAMPLE_REPORT)
        content = csv_bytes.decode("utf-8")
        assert "MH12AB1234" in content

    def test_csv_has_header_row(self):
        from src.report import report_to_csv_bytes
        csv_bytes = report_to_csv_bytes(SAMPLE_REPORT)
        lines = csv_bytes.decode("utf-8").strip().splitlines()
        assert len(lines) >= 2  # header + data


# ---------------------------------------------------------------------------
# Tests: session history
# ---------------------------------------------------------------------------

class TestSessionHistory:
    def test_append_history_increases_length(self):
        from src.report import append_history
        history = []
        history = append_history(history, SAMPLE_REPORT)
        assert len(history) == 1
        history = append_history(history, SAMPLE_REPORT)
        assert len(history) == 2

    def test_history_to_csv_bytes_empty(self):
        from src.report import history_to_csv_bytes
        result = history_to_csv_bytes([])
        assert result == b""

    def test_history_to_csv_bytes_nonempty(self):
        from src.report import history_to_csv_bytes
        csv_bytes = history_to_csv_bytes([SAMPLE_REPORT, SAMPLE_REPORT])
        lines = csv_bytes.decode("utf-8").strip().splitlines()
        assert len(lines) == 3  # 1 header + 2 data rows

    def test_history_to_dataframe(self):
        from src.report import history_to_dataframe
        df = history_to_dataframe([SAMPLE_REPORT])
        assert len(df) == 1
        assert "recognized_text" in df.columns

    def test_empty_history_returns_empty_dataframe(self):
        from src.report import history_to_dataframe
        df = history_to_dataframe([])
        assert len(df) == 0
