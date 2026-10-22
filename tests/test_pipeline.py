"""
tests/test_pipeline.py — Integration-style tests for src/pipeline.py

All external model calls are mocked.  These tests verify pipeline logic,
error handling, and state transitions without requiring actual YOLO weights
or OCR network downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import io
from PIL import Image


# ---------------------------------------------------------------------------
# Helper: create synthetic JPEG bytes
# ---------------------------------------------------------------------------

def make_image_bytes(w: int = 200, h: int = 150) -> bytes:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_detector():
    from src.detector import DetectionResult, PlateDetection, PlateDetector
    detector = MagicMock(spec=PlateDetector)
    detector.is_loaded = True
    detector.conf_threshold = 0.25
    # Default: one detection
    plate = PlateDetection(bbox=(10, 10, 100, 50), confidence=0.85, class_name="license_plate")
    detector.detect.return_value = DetectionResult(plates=[plate])
    return detector


@pytest.fixture
def mock_ocr_engine():
    from src.ocr_engine import EasyOCREngine, OCRResult
    engine = MagicMock(spec=EasyOCREngine)
    engine.is_initialized = True
    engine.recognize.return_value = OCRResult(
        raw_text="MH 12 AB 1234",
        normalized_text="MH 12 AB 1234",
        confidence=0.91,
        confidence_source="EasyOCR average word confidence",
        success=True,
    )
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineSuccess:
    def test_successful_recognition(self, mock_detector, mock_ocr_engine):
        """Happy path: plate detected, OCR succeeds."""
        from src.pipeline import ALPRPipeline
        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        result = pipeline.process_bytes(make_image_bytes(), "test.jpg")
        assert result.plate_count == 1
        assert result.recognized_text == "MH 12 AB 1234"
        assert result.success is True
        assert result.detector_confidence == pytest.approx(0.85)
        assert result.ocr_confidence == pytest.approx(0.91)

    def test_annotated_image_generated(self, mock_detector, mock_ocr_engine):
        from src.pipeline import ALPRPipeline
        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        result = pipeline.process_bytes(make_image_bytes(), "test.jpg")
        assert result.annotated_image is not None
        assert isinstance(result.annotated_image, np.ndarray)

    def test_report_populated(self, mock_detector, mock_ocr_engine):
        from src.pipeline import ALPRPipeline
        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        result = pipeline.process_bytes(make_image_bytes(), "test.jpg")
        assert result.report
        assert result.report["recognized_text"] == "MH 12 AB 1234"
        assert result.report["filename"] == "test.jpg"

    def test_processing_time_recorded(self, mock_detector, mock_ocr_engine):
        from src.pipeline import ALPRPipeline
        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        result = pipeline.process_bytes(make_image_bytes(), "test.jpg")
        assert result.processing_time_sec >= 0.0


class TestPipelineNoDetection:
    def test_no_plate_detected_message(self, mock_detector, mock_ocr_engine):
        """When no plate is found, user_message should say so."""
        from src.detector import DetectionResult
        from src.pipeline import ALPRPipeline
        mock_detector.detect.return_value = DetectionResult(plates=[])

        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        result = pipeline.process_bytes(make_image_bytes(), "empty.jpg")
        assert result.plate_count == 0
        assert result.success is False
        assert "no number plate" in result.user_message.lower()

    def test_no_plate_annotated_image_still_produced(self, mock_detector, mock_ocr_engine):
        from src.detector import DetectionResult
        from src.pipeline import ALPRPipeline
        mock_detector.detect.return_value = DetectionResult(plates=[])

        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        result = pipeline.process_bytes(make_image_bytes(), "empty.jpg")
        assert result.annotated_image is not None


class TestPipelineDetectionFailure:
    def test_detector_config_error_returns_graceful_result(self, mock_ocr_engine):
        """Missing model weights should produce a friendly error, not a crash."""
        from src.detector import ConfigurationError, DetectionResult, PlateDetector
        from src.pipeline import ALPRPipeline

        bad_detector = MagicMock(spec=PlateDetector)
        bad_detector.detect.return_value = DetectionResult(
            success=False,
            error_message="Model weight file not found: models/plate_detector.pt",
        )

        pipeline = ALPRPipeline(bad_detector, mock_ocr_engine)
        result = pipeline.process_bytes(make_image_bytes(), "test.jpg")
        assert result.success is False
        assert result.error_message != ""


class TestPipelineOCRFailure:
    def test_ocr_failure_produces_advisory_message(self, mock_detector, mock_ocr_engine):
        """When OCR fails, the pipeline should inform the user without crashing."""
        from src.ocr_engine import OCRResult
        from src.pipeline import ALPRPipeline
        mock_ocr_engine.recognize.return_value = OCRResult(
            success=False,
            message="OCR ran but could not extract readable text.",
        )

        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        result = pipeline.process_bytes(make_image_bytes(), "test.jpg")
        # plate was detected but OCR failed
        assert result.plate_count == 1
        assert result.recognized_text == ""
        assert "recognized" in result.user_message.lower() or "detected" in result.user_message.lower()


class TestPipelineMultiplePlates:
    def test_multiple_plates_detected_count_correct(self, mock_detector, mock_ocr_engine):
        from src.detector import DetectionResult, PlateDetection
        from src.pipeline import ALPRPipeline

        plates = [
            PlateDetection(bbox=(10, 10, 100, 50), confidence=0.90),
            PlateDetection(bbox=(200, 20, 310, 60), confidence=0.75),
        ]
        mock_detector.detect.return_value = DetectionResult(plates=plates)

        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        result = pipeline.process_bytes(make_image_bytes(), "multi.jpg")
        assert result.plate_count == 2


class TestPipelineCorruptedInput:
    def test_corrupted_bytes_returns_error(self, mock_detector, mock_ocr_engine):
        from src.pipeline import ALPRPipeline
        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        corrupted = b"\xff\xfe\x00\x01garbage"
        result = pipeline.process_bytes(corrupted, "bad.jpg")
        assert result.success is False
        assert result.error_message != ""

    def test_empty_bytes_returns_error(self, mock_detector, mock_ocr_engine):
        from src.pipeline import ALPRPipeline
        pipeline = ALPRPipeline(mock_detector, mock_ocr_engine)
        result = pipeline.process_bytes(b"", "empty.jpg")
        assert result.success is False
