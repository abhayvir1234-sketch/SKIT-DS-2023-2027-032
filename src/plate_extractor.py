"""
src/plate_extractor.py — Safe plate crop extraction and enhancement.

Author: Abhayvir Singh (Team Lead) — Plate region extraction and cropping.

Given a bounding box from the detector, this module extracts the plate crop
from the original image, adds configurable padding, and rejects degenerate
crops before passing them to the OCR engine.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from src.preprocessing import preprocess_plate_for_ocr, validate_image_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_plate_crop(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = config.CROP_PADDING_PX,
) -> Optional[np.ndarray]:
    """Crop the plate region from *image* using *bbox*, with padding.

    Bounding box coordinates are clipped to the image boundaries so that
    padding never reads outside the array.

    Args:
        image: Full-frame BGR numpy array.
        bbox: Plate bounding box (x1, y1, x2, y2) in pixel coordinates.
        padding: Pixels of context to add on each side of the detected region.

    Returns:
        Cropped BGR numpy array, or *None* if the crop would be degenerate.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Apply padding while staying inside image bounds
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(w, x2 + padding)
    y2p = min(h, y2 + padding)

    if x2p <= x1p or y2p <= y1p:
        logger.warning("Degenerate plate crop after padding: %s", bbox)
        return None

    crop = image[y1p:y2p, x1p:x2p].copy()

    error = validate_image_array(crop)
    if error:
        logger.warning("Invalid crop array: %s", error)
        return None

    return crop


def extract_all_crops(
    image: np.ndarray,
    bboxes: list[Tuple[int, int, int, int]],
    padding: int = config.CROP_PADDING_PX,
) -> list[Optional[np.ndarray]]:
    """Extract crop arrays for a list of bounding boxes.

    Args:
        image: Full-frame BGR numpy array.
        bboxes: List of (x1, y1, x2, y2) bounding boxes.
        padding: Crop padding in pixels.

    Returns:
        List of BGR crop arrays (element is *None* if extraction failed).
    """
    return [extract_plate_crop(image, bbox, padding) for bbox in bboxes]


# ---------------------------------------------------------------------------
# Enhancement for OCR
# ---------------------------------------------------------------------------


def get_ocr_variants(
    plate_crop: np.ndarray,
    target_height: int = 64,
) -> List[np.ndarray]:
    """Return pre-processed plate variants ready for OCR.

    Delegates to :func:`src.preprocessing.preprocess_plate_for_ocr`.  The
    caller should try each variant with the OCR engine and select the result
    with the highest confidence.

    Args:
        plate_crop: Raw plate crop (BGR numpy array).
        target_height: Height to normalise all variants to.

    Returns:
        List of numpy arrays (variant images).  May be empty if *plate_crop*
        is invalid.
    """
    error = validate_image_array(plate_crop)
    if error:
        logger.warning("Cannot generate OCR variants: %s", error)
        return []
    return preprocess_plate_for_ocr(plate_crop, target_height=target_height)


# ---------------------------------------------------------------------------
# Saving crops (optional, for diagnostics)
# ---------------------------------------------------------------------------


def save_crop(crop: np.ndarray, path: str) -> bool:
    """Save a plate crop image to *path*.

    Args:
        crop: BGR numpy array.
        path: Destination filepath (e.g. "outputs/plate_001.jpg").

    Returns:
        True on success, False on failure.
    """
    try:
        success = cv2.imwrite(path, crop)
        if not success:
            logger.warning("cv2.imwrite returned False for '%s'.", path)
        return success
    except Exception as exc:
        logger.error("Could not save crop to '%s': %s", path, exc)
        return False
