"""
src/preprocessing.py — Image preprocessing utilities.

Author: Abhayvir Singh (Team Lead) — Image preprocessing.

These utilities prepare raw images and plate crops for YOLO inference and OCR.
They are intentionally modular: callers select which steps to apply rather than
blindly running every enhancement.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grayscale helpers
# ---------------------------------------------------------------------------


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale image to a single-channel grayscale array.

    Args:
        image: Input numpy array (BGR or already grayscale).

    Returns:
        Single-channel uint8 grayscale array.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Noise reduction
# ---------------------------------------------------------------------------


def denoise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """Apply mild Gaussian blur to suppress sensor noise.

    Args:
        image: Grayscale or BGR numpy array.
        strength: Blur kernel size (odd integer; higher = more smoothing).

    Returns:
        Blurred image array (same shape and dtype as input).
    """
    k = strength if strength % 2 == 1 else strength + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def denoise_nlm(gray: np.ndarray, h: int = 10) -> np.ndarray:
    """Non-local means denoising for grayscale images.

    Slower than Gaussian blur but preserves edges better.  Use on plates where
    noise is heavy and character edges matter.

    Args:
        gray: Single-channel grayscale array.
        h: Filter strength (higher removes more noise but blurs edges).

    Returns:
        Denoised grayscale array.
    """
    return cv2.fastNlMeansDenoising(gray, h=h)


# ---------------------------------------------------------------------------
# Contrast enhancement
# ---------------------------------------------------------------------------


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    Args:
        gray: Single-channel grayscale array.
        clip_limit: Threshold for contrast limiting.
        tile_size: Tile grid size for local histogram computation.

    Returns:
        Contrast-enhanced grayscale array.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Apply Otsu's global binarisation.

    Args:
        gray: Single-channel grayscale array.

    Returns:
        Binary thresholded array (0 or 255).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(gray: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """Apply adaptive (local) Gaussian thresholding.

    Useful for plates with uneven illumination.

    Args:
        gray: Single-channel grayscale array.
        block_size: Neighbourhood size for local mean computation (odd, >= 3).
        c: Constant subtracted from the computed mean.

    Returns:
        Binary thresholded array (0 or 255).
    """
    bs = block_size if block_size % 2 == 1 else block_size + 1
    bs = max(bs, 3)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bs, c,
    )


# ---------------------------------------------------------------------------
# Sharpening
# ---------------------------------------------------------------------------


def sharpen(image: np.ndarray) -> np.ndarray:
    """Unsharp-mask sharpening to enhance text character edges.

    Args:
        image: BGR or grayscale array.

    Returns:
        Sharpened array, clamped to uint8 range.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


# ---------------------------------------------------------------------------
# Resize helpers
# ---------------------------------------------------------------------------


def resize_to_height(image: np.ndarray, target_height: int = 64) -> np.ndarray:
    """Resize an image to *target_height* while preserving aspect ratio.

    Args:
        image: Input numpy array (any shape).
        target_height: Desired height in pixels.

    Returns:
        Resized image array.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)


def resize_keeping_aspect(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Downscale an image so its longest side is at most *max_side* pixels.

    Upscaling is never performed.

    Args:
        image: BGR numpy array.
        max_side: Maximum allowed length of the longest dimension.

    Returns:
        Resized (or unchanged) numpy array.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Composite preprocessing chains
# ---------------------------------------------------------------------------


def preprocess_for_detection(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Light preprocessing to pass a full-frame image into the YOLO detector.

    No colour normalisation is applied; YOLO expects a natural-colour BGR frame.

    Args:
        image: Raw uploaded image as BGR numpy array.
        max_side: Limit the longest side to avoid GPU OOM on huge images.

    Returns:
        BGR array ready for YOLO inference.
    """
    image = resize_keeping_aspect(image, max_side=max_side)
    return image


def preprocess_plate_for_ocr(
    plate_bgr: np.ndarray,
    target_height: int = 64,
) -> list[np.ndarray]:
    """Generate a set of preprocessed plate variants for OCR.

    Multiple variants are returned so the OCR engine can try each and select
    the result with the highest confidence.  The caller is responsible for
    selection.

    Variants produced:
        0 — Original plate crop, resized to target height (colour).
        1 — Grayscale + CLAHE.
        2 — Grayscale + CLAHE + Otsu binarisation.
        3 — Grayscale + CLAHE + adaptive thresholding.
        4 — Sharpened grayscale + CLAHE.

    Args:
        plate_bgr: Cropped licence-plate region as BGR numpy array.
        target_height: Height (pixels) to normalise all variants to.

    Returns:
        List of numpy arrays (one per variant). Each variant is 3-channel BGR
        or grayscale depending on the operation applied.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return []

    resized = resize_to_height(plate_bgr, target_height)
    gray = to_grayscale(resized)
    clahe_img = apply_clahe(gray)

    variants: list[np.ndarray] = [
        resized,                                     # 0: colour
        clahe_img,                                   # 1: CLAHE
        otsu_threshold(clahe_img),                   # 2: CLAHE + Otsu
        adaptive_threshold(clahe_img),               # 3: CLAHE + adaptive
        apply_clahe(to_grayscale(sharpen(resized))), # 4: sharpened + CLAHE
    ]
    return variants


def validate_image_array(image: np.ndarray) -> Optional[str]:
    """Check that a numpy array represents a non-trivial image.

    Args:
        image: Candidate numpy array.

    Returns:
        An error message string if the image is invalid, otherwise *None*.
    """
    if image is None:
        return "Image array is None."
    if not isinstance(image, np.ndarray):
        return f"Expected numpy ndarray, got {type(image).__name__}."
    if image.size == 0:
        return "Image array is empty (zero elements)."
    if image.ndim not in (2, 3):
        return f"Unexpected array dimensions: {image.ndim}."
    h, w = image.shape[:2]
    if h < 4 or w < 4:
        return f"Image is too small ({w}×{h} px)."
    return None
