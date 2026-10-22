"""
utils/image_utils.py — Shared image helper functions.

Author: Abhayvir Singh (Team Lead) — Image processing utilities.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV BGR numpy array.

    Args:
        image: A PIL Image object (any mode; will be converted to RGB first).

    Returns:
        A numpy array in BGR channel order, dtype=uint8.
    """
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(array: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR numpy array to a PIL Image (RGB).

    Args:
        array: A numpy array in BGR channel order.

    Returns:
        A PIL Image in RGB mode.
    """
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to an OpenCV BGR numpy array.

    Args:
        image_bytes: Raw bytes of a JPG, PNG, or similar image file.

    Returns:
        A numpy array in BGR channel order.

    Raises:
        ValueError: If the bytes cannot be decoded as a valid image.
    """
    if not image_bytes:
        raise ValueError("Could not decode image bytes — empty byte string provided.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"Could not decode image bytes — {exc}") from exc
    if img is None:
        raise ValueError("Could not decode image bytes — file may be corrupted or unsupported.")
    return img


def bgr_to_bytes(array: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode an OpenCV BGR array to image bytes.

    Args:
        array: BGR numpy array.
        ext: File extension that determines the codec (e.g. ".jpg", ".png").

    Returns:
        Encoded image as bytes.

    Raises:
        ValueError: If encoding fails.
    """
    success, buf = cv2.imencode(ext, array)
    if not success:
        raise ValueError(f"Failed to encode image as {ext}")
    return buf.tobytes()


def pil_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load a PIL Image from raw bytes.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        A PIL Image in RGB mode.

    Raises:
        ValueError: If the bytes do not represent a valid image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()                    # Raises on corrupt file
        # Re-open after verify() (PIL closes the stream after verify)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return img
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot load image: {exc}") from exc


# ---------------------------------------------------------------------------
# Resize helpers
# ---------------------------------------------------------------------------

def resize_keeping_aspect(
    image: np.ndarray,
    max_side: int = 1280,
) -> np.ndarray:
    """Downscale an image so its longest side is at most *max_side* pixels.

    Upscaling is never performed.  If the image is already within bounds,
    it is returned unchanged (no copy is made).

    Args:
        image: BGR numpy array.
        max_side: Maximum allowed length of the longest dimension.

    Returns:
        Resized (or unchanged) BGR numpy array.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_image_info(image: np.ndarray) -> dict:
    """Return a dict of basic image metadata.

    Args:
        image: BGR numpy array.

    Returns:
        Dictionary with keys: height, width, channels, dtype.
    """
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(image.dtype),
    }
