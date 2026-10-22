"""
src/annotation.py — Draw detection annotations on images.

Author: Karam Singh Bana (Team Member 2) — Annotated output image display.

Produces a copy of the original image with bounding boxes, labels, and
confidence text drawn.  The original numpy array is never modified.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from src.detector import PlateDetection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core annotation function
# ---------------------------------------------------------------------------


def annotate_image(
    image: np.ndarray,
    plates: List[PlateDetection],
    ocr_texts: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw bounding boxes and labels for all detected plates.

    Args:
        image: Original BGR numpy array (will NOT be modified; a copy is made).
        plates: List of :class:`~src.detector.PlateDetection` objects.
        ocr_texts: Optional list of recognized text strings, one per plate.
                   When provided, the recognized text is shown alongside
                   the detector confidence.  Must have the same length as
                   *plates* if provided.

    Returns:
        A new BGR numpy array with annotations drawn.
    """
    annotated = image.copy()

    if ocr_texts is None:
        ocr_texts = [""] * len(plates)

    if len(ocr_texts) < len(plates):
        ocr_texts = list(ocr_texts) + [""] * (len(plates) - len(ocr_texts))

    for i, plate in enumerate(plates):
        x1, y1, x2, y2 = plate.bbox
        box_color = config.ANNOTATION_BOX_COLOR
        thickness = config.ANNOTATION_THICKNESS

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        # Build label text
        conf_str = f"{plate.confidence:.2f}"
        recognized = ocr_texts[i]
        if recognized:
            label = f"{recognized}  [{conf_str}]"
        else:
            label = f"Plate  [{conf_str}]"

        # Compute text size to draw background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = config.ANNOTATION_FONT_SCALE
        font_thick = max(1, thickness - 1)
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        # Position label above the box (or inside if the box is at the top)
        label_y = y1 - baseline - 4
        if label_y - th < 0:
            label_y = y2 + th + baseline + 4

        # Background rectangle for readability
        bg_x1 = x1
        bg_y1 = label_y - th - baseline
        bg_x2 = x1 + tw + 4
        bg_y2 = label_y + baseline

        cv2.rectangle(
            annotated,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            config.ANNOTATION_BG_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y),
            font,
            font_scale,
            config.ANNOTATION_TEXT_COLOR,
            font_thick,
            cv2.LINE_AA,
        )

    return annotated


def annotate_no_detection(image: np.ndarray) -> np.ndarray:
    """Return an annotated copy with a 'No plate detected' overlay.

    Args:
        image: Original BGR numpy array.

    Returns:
        Annotated copy with overlay text.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    text = "No number plate detected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(w, h) / 500.0
    font_scale = max(0.5, min(font_scale, 1.5))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    x = (w - tw) // 2
    y = h - 30
    # Semi-transparent overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
    cv2.putText(annotated, text, (x, y), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
    return annotated
