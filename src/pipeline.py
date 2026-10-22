"""
src/pipeline.py — ALPR end-to-end recognition pipeline.

Author: Abhayvir Singh (Team Lead) — Integration and coordination.

Orchestrates: validation → preprocessing → detection → crop → OCR →
text validation → annotation → report.

Each stage is isolated in a try/except block so that a failure in one stage
does not silently corrupt results shown to the user.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from src.annotation import annotate_image, annotate_no_detection
from src.detector import DetectionResult, PlateDetection, PlateDetector
from src.ocr_engine import EasyOCREngine, OCRResult
from src.plate_extractor import extract_plate_crop, get_ocr_variants
from src.report import build_recognition_report
from src.validation import ValidationResult, validate_registration
from utils.image_utils import bytes_to_bgr, pil_from_bytes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Complete result of one ALPR pipeline run."""

    # --- Input ---
    filename: str = ""

    # --- Detection ---
    detection_result: Optional[DetectionResult] = None
    plate_count: int = 0

    # --- Best plate ---
    best_plate: Optional[PlateDetection] = None
    plate_crop: Optional[np.ndarray] = field(default=None, repr=False)

    # --- OCR ---
    ocr_result: Optional[OCRResult] = None
    recognized_text: str = ""

    # --- Validation ---
    validation_result: Optional[ValidationResult] = None

    # --- Output images ---
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Timing ---
    processing_time_sec: float = 0.0

    # --- Status ---
    success: bool = False
    error_message: str = ""
    user_message: str = ""

    # --- Report ---
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def detector_confidence(self) -> Optional[float]:
        return self.best_plate.confidence if self.best_plate else None

    @property
    def ocr_confidence(self) -> Optional[float]:
        return self.ocr_result.confidence if self.ocr_result else None


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class ALPRPipeline:
    """End-to-end Automatic Licence Plate Recognition pipeline.

    Args:
        detector: Configured :class:`~src.detector.PlateDetector` instance.
        ocr_engine: Configured :class:`~src.ocr_engine.EasyOCREngine` instance.
    """

    def __init__(
        self,
        detector: PlateDetector,
        ocr_engine: EasyOCREngine,
    ) -> None:
        self.detector = detector
        self.ocr_engine = ocr_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_bytes(
        self,
        image_bytes: bytes,
        filename: str = "upload",
    ) -> PipelineResult:
        """Run the full pipeline on raw image bytes.

        This is the primary entry point from the Streamlit app.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            filename: Original filename (for reporting).

        Returns:
            A fully populated :class:`PipelineResult`.
        """
        result = PipelineResult(filename=filename)
        t_start = time.monotonic()

        # 1. Decode image bytes
        try:
            bgr_image = bytes_to_bgr(image_bytes)
        except ValueError as exc:
            result.error_message = str(exc)
            result.user_message = (
                "The uploaded file could not be read as an image. "
                "Please ensure it is a valid JPG or PNG file."
            )
            result.processing_time_sec = time.monotonic() - t_start
            result.report = self._build_report(result)
            return result

        # 2. Plate detection
        detection_result = self._run_detection(bgr_image, result)
        if detection_result is None:
            result.processing_time_sec = time.monotonic() - t_start
            result.report = self._build_report(result)
            return result

        result.detection_result = detection_result
        result.plate_count = detection_result.plate_count

        # 3. No plate found
        if detection_result.plate_count == 0:
            result.user_message = (
                "No number plate detected. "
                "Please upload a clearer vehicle image with a visible plate."
            )
            result.annotated_image = annotate_no_detection(bgr_image)
            result.processing_time_sec = time.monotonic() - t_start
            result.report = self._build_report(result)
            return result

        # 4. Extract crop for the best plate
        best_plate = detection_result.best_plate
        result.best_plate = best_plate
        crop = extract_plate_crop(bgr_image, best_plate.bbox)
        result.plate_crop = crop

        # 5. OCR
        ocr_result: Optional[OCRResult] = None
        if crop is not None:
            ocr_result = self._run_ocr(crop, result)
        else:
            result.error_message = "Plate crop extraction produced an empty image."

        result.ocr_result = ocr_result

        # 6. Text extraction and validation
        if ocr_result and ocr_result.success:
            result.recognized_text = ocr_result.normalized_text
            val = validate_registration(ocr_result.normalized_text)
            result.validation_result = val
            if val.is_valid:
                result.user_message = f"Registration recognized: {result.recognized_text}"
                result.success = True
            else:
                result.user_message = (
                    f"Plate text read as '{result.recognized_text}', "
                    f"but format validation note: {val.message}"
                )
                result.success = True  # OCR succeeded; validation is advisory
        else:
            result.user_message = (
                "Number plate detected, but the registration text could not "
                "be recognized confidently."
            )

        # 7. Annotate
        ocr_texts = [result.recognized_text] if result.recognized_text else [""]
        result.annotated_image = annotate_image(bgr_image, [best_plate], ocr_texts)

        result.processing_time_sec = time.monotonic() - t_start
        result.report = self._build_report(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_detection(
        self,
        bgr_image: np.ndarray,
        result: PipelineResult,
    ) -> Optional[DetectionResult]:
        """Run detection and populate error state on failure."""
        try:
            detection_result = self.detector.detect(bgr_image)
        except Exception as exc:
            logger.exception("Unexpected error during detection: %s", exc)
            result.error_message = f"Detection error: {exc}"
            result.user_message = (
                "An unexpected error occurred during plate detection. "
                "Check that the model weights are configured correctly."
            )
            return None

        if not detection_result.success:
            result.error_message = detection_result.error_message
            result.user_message = (
                f"Plate detection failed: {detection_result.error_message}"
            )
            return None

        return detection_result

    def _run_ocr(
        self,
        crop: np.ndarray,
        result: PipelineResult,
    ) -> Optional[OCRResult]:
        """Run OCR on all variants of *crop*."""
        try:
            variants = get_ocr_variants(crop)
            if not variants:
                result.error_message = "No valid OCR input variants generated."
                return None
            ocr_result = self.ocr_engine.recognize(variants)
            return ocr_result
        except Exception as exc:
            logger.exception("Unexpected OCR error: %s", exc)
            result.error_message = f"OCR error: {exc}"
            return None

    def _build_report(self, result: PipelineResult) -> Dict[str, Any]:
        """Build the recognition report dictionary."""
        val_passed = (
            result.validation_result.is_valid
            if result.validation_result
            else False
        )
        val_msg = (
            result.validation_result.message
            if result.validation_result
            else ""
        )
        return build_recognition_report(
            filename=result.filename,
            plate_count=result.plate_count,
            recognized_text=result.recognized_text,
            detector_confidence=result.detector_confidence,
            ocr_confidence=result.ocr_confidence,
            validation_passed=val_passed,
            validation_message=val_msg,
            processing_time_sec=result.processing_time_sec,
            error_message=result.error_message,
        )
