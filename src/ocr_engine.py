"""
src/ocr_engine.py — OCR integration for number plate text extraction.

Author: Himank Devnani (Team Member 1) — OCR integration and text recognition.

Supports EasyOCR. Each recognized result carries the raw text, normalized
text, and confidence score. Multiple preprocessing variants are tried;
the result with the highest aggregate confidence is selected.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class OCRResult:
    """Text recognized from a single plate image."""

    #: Raw, unmodified text as returned by the OCR engine.
    raw_text: str = ""

    #: Normalized registration number (uppercase, spaces removed, artifacts stripped).
    normalized_text: str = ""

    #: Aggregate confidence score (0.0–1.0), or None when unavailable.
    confidence: Optional[float] = None

    #: Human-readable label for the confidence source.
    confidence_source: str = ""

    #: True when OCR succeeded and produced non-empty normalized text.
    success: bool = False

    #: Additional diagnostic message (error description or OCR note).
    message: str = ""

    #: Which preprocessing variant index produced this result.
    variant_index: int = 0


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Characters that are unambiguously illegal in any registration number
_ILLEGAL_CHARS = re.compile(r"[^A-Z0-9 \-]")

# Repeated internal spaces / hyphens
_MULTI_SPACE = re.compile(r"[\s\-]+")


def normalize_plate_text(text: str) -> str:
    """Normalize raw OCR text into a clean registration-number candidate.

    Steps applied:
        1. Unicode NFC normalization.
        2. Convert to uppercase.
        3. Remove characters that are not alphanumeric, space, or hyphen.
        4. Collapse consecutive spaces/hyphens to a single space.
        5. Strip leading/trailing whitespace.

    Deliberate non-corrections:
        O → 0, I → 1 substitutions are NOT applied automatically because they
        can corrupt genuinely correct characters.  They may be enabled via a
        future configurable rule set.

    Args:
        text: Raw string from OCR engine.

    Returns:
        Cleaned registration-number candidate (may be empty).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.upper()
    text = _ILLEGAL_CHARS.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# EasyOCR engine
# ---------------------------------------------------------------------------


class EasyOCREngine:
    """OCR engine backed by EasyOCR.

    The ``easyocr.Reader`` object is expensive to create (it loads neural
    network weights). This class keeps a single instance and exposes a lazy
    :meth:`initialize` method.  Callers should initialize once (ideally at
    Streamlit startup via ``@st.cache_resource``) and reuse the same engine.

    Args:
        languages: List of EasyOCR language codes (default: ["en"]).
        use_gpu: If True, request GPU inference.  Falls back to CPU
            automatically if CUDA is unavailable.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
    ) -> None:
        self.languages = languages or config.EASYOCR_LANGUAGES
        self.use_gpu = use_gpu
        self._reader = None  # type: ignore[assignment]

    @property
    def is_initialized(self) -> bool:
        return self._reader is not None

    def initialize(self) -> None:
        """Load the EasyOCR model (downloads weights on first run).

        Raises:
            RuntimeError: If EasyOCR cannot be imported or the reader fails to
                initialise.
        """
        try:
            import easyocr  # noqa: PLC0415

            logger.info(
                "Initializing EasyOCR (languages=%s, gpu=%s)",
                self.languages,
                self.use_gpu,
            )
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
            )
            logger.info("EasyOCR initialized successfully.")
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is not installed. Run: pip install easyocr"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"EasyOCR initialization failed: {exc}"
            ) from exc

    def ensure_initialized(self) -> None:
        if not self.is_initialized:
            self.initialize()

    def _run_on_array(
        self,
        image: np.ndarray,
        min_conf: float,
    ) -> tuple[str, Optional[float]]:
        """Run OCR on a single numpy array and return (raw_text, confidence).

        EasyOCR returns a list of (bbox, text, prob) tuples.  We concatenate
        all text fragments that exceed *min_conf* and compute an average
        confidence.

        Args:
            image: BGR or grayscale numpy array of the plate region.
            min_conf: Minimum per-word confidence to include.

        Returns:
            Tuple of (concatenated raw text, average confidence or None).
        """
        try:
            detections = self._reader.readtext(image, detail=1)
        except Exception as exc:
            logger.warning("EasyOCR readtext failed: %s", exc)
            return "", None

        if not detections:
            return "", None

        words: list[str] = []
        confidences: list[float] = []
        for _bbox, text, prob in detections:
            if prob >= min_conf:
                words.append(text)
                confidences.append(prob)

        if not words:
            return "", None

        raw = " ".join(words)
        avg_conf = sum(confidences) / len(confidences)
        return raw, avg_conf

    def recognize(
        self,
        plate_variants: List[np.ndarray],
        min_conf: float = config.OCR_MIN_CONFIDENCE,
    ) -> OCRResult:
        """Try OCR on each variant and return the best result.

        "Best" is defined as the variant whose normalized text is non-empty
        and has the highest average per-word confidence.

        Args:
            plate_variants: Preprocessed plate images (from plate_extractor).
            min_conf: Minimum per-word confidence threshold.

        Returns:
            An :class:`OCRResult`.  On failure, ``success`` is False and
            ``message`` explains why.
        """
        try:
            self.ensure_initialized()
        except RuntimeError as exc:
            return OCRResult(
                success=False,
                message=str(exc),
                confidence_source="unavailable",
            )

        if not plate_variants:
            return OCRResult(
                success=False,
                message="No plate image variants provided.",
            )

        best: Optional[OCRResult] = None

        for idx, variant in enumerate(plate_variants):
            if variant is None or variant.size == 0:
                continue

            raw, conf = self._run_on_array(variant, min_conf)
            normalized = normalize_plate_text(raw)

            if not normalized:
                continue

            candidate = OCRResult(
                raw_text=raw,
                normalized_text=normalized,
                confidence=conf,
                confidence_source="EasyOCR average word confidence",
                success=True,
                variant_index=idx,
            )

            if best is None:
                best = candidate
            elif (
                conf is not None
                and best.confidence is not None
                and conf > best.confidence
            ):
                best = candidate
            elif conf is not None and best.confidence is None:
                best = candidate

        if best is None:
            return OCRResult(
                success=False,
                message=(
                    "OCR ran but could not extract readable text. "
                    "The plate image may be too blurry, dark, or small."
                ),
                confidence_source="EasyOCR average word confidence",
            )

        logger.info(
            "OCR result: '%s' (conf=%.3f, variant=%d)",
            best.normalized_text,
            best.confidence or 0.0,
            best.variant_index,
        )
        return best


# ---------------------------------------------------------------------------
# Factory / singleton helper
# ---------------------------------------------------------------------------

_engine_instance: Optional[EasyOCREngine] = None


def get_ocr_engine(use_gpu: bool = False) -> EasyOCREngine:
    """Return the shared :class:`EasyOCREngine` instance (created lazily).

    This function is used outside of Streamlit (e.g. in tests) where
    ``@st.cache_resource`` is not available.

    Args:
        use_gpu: Forward to :class:`EasyOCREngine` constructor.

    Returns:
        Initialized engine instance.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EasyOCREngine(use_gpu=use_gpu)
    return _engine_instance
