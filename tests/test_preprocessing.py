"""
tests/test_preprocessing.py — Unit tests for src/preprocessing.py

These tests verify preprocessing logic using synthetic numpy arrays.
No external model weights or network access are required.
"""

import numpy as np
import pytest
import cv2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb_image(h: int = 100, w: int = 200) -> np.ndarray:
    """Create a random BGR image array for testing."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_gray_image(h: int = 100, w: int = 200) -> np.ndarray:
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests: to_grayscale
# ---------------------------------------------------------------------------

class TestToGrayscale:
    def test_bgr_converts_to_single_channel(self):
        from src.preprocessing import to_grayscale
        img = make_rgb_image()
        gray = to_grayscale(img)
        assert gray.ndim == 2, "Expected single-channel output."

    def test_already_gray_returns_unchanged(self):
        from src.preprocessing import to_grayscale
        img = make_gray_image()
        gray = to_grayscale(img)
        assert gray.ndim == 2
        assert gray.shape == img.shape

    def test_output_dtype_uint8(self):
        from src.preprocessing import to_grayscale
        img = make_rgb_image()
        gray = to_grayscale(img)
        assert gray.dtype == np.uint8


# ---------------------------------------------------------------------------
# Tests: resize_keeping_aspect
# ---------------------------------------------------------------------------

class TestResizeKeepingAspect:
    def test_large_image_is_downscaled(self):
        from src.preprocessing import resize_keeping_aspect
        img = make_rgb_image(2000, 3000)
        result = resize_keeping_aspect(img, max_side=1280)
        assert max(result.shape[:2]) <= 1280

    def test_small_image_is_not_upscaled(self):
        from src.preprocessing import resize_keeping_aspect
        img = make_rgb_image(100, 150)
        result = resize_keeping_aspect(img, max_side=1280)
        assert result.shape == img.shape

    def test_aspect_ratio_preserved(self):
        from src.preprocessing import resize_keeping_aspect
        img = make_rgb_image(400, 800)   # 2:1 aspect ratio
        result = resize_keeping_aspect(img, max_side=640)
        h, w = result.shape[:2]
        assert abs(w / h - 2.0) < 0.1, "Aspect ratio should be approximately 2:1."

    def test_unusual_dimensions(self):
        """Very tall or very wide images should not raise."""
        from src.preprocessing import resize_keeping_aspect
        tall = make_rgb_image(4000, 50)
        result = resize_keeping_aspect(tall, max_side=640)
        assert max(result.shape[:2]) <= 640

        wide = make_rgb_image(50, 4000)
        result = resize_keeping_aspect(wide, max_side=640)
        assert max(result.shape[:2]) <= 640


# ---------------------------------------------------------------------------
# Tests: apply_clahe
# ---------------------------------------------------------------------------

class TestCLAHE:
    def test_clahe_output_shape(self):
        from src.preprocessing import apply_clahe
        gray = make_gray_image()
        enhanced = apply_clahe(gray)
        assert enhanced.shape == gray.shape
        assert enhanced.dtype == np.uint8

    def test_clahe_on_uniform_image(self):
        """CLAHE on a fully gray image should not raise."""
        from src.preprocessing import apply_clahe
        gray = np.full((100, 200), 128, dtype=np.uint8)
        enhanced = apply_clahe(gray)
        assert enhanced is not None


# ---------------------------------------------------------------------------
# Tests: validate_image_array
# ---------------------------------------------------------------------------

class TestValidateImageArray:
    def test_none_returns_error(self):
        from src.preprocessing import validate_image_array
        assert validate_image_array(None) is not None

    def test_empty_array_returns_error(self):
        from src.preprocessing import validate_image_array
        assert validate_image_array(np.array([])) is not None

    def test_valid_image_returns_none(self):
        from src.preprocessing import validate_image_array
        img = make_rgb_image(50, 100)
        assert validate_image_array(img) is None

    def test_tiny_image_returns_error(self):
        from src.preprocessing import validate_image_array
        tiny = np.zeros((2, 2, 3), dtype=np.uint8)
        assert validate_image_array(tiny) is not None


# ---------------------------------------------------------------------------
# Tests: preprocess_plate_for_ocr
# ---------------------------------------------------------------------------

class TestPreprocessPlateForOCR:
    def test_returns_multiple_variants(self):
        from src.preprocessing import preprocess_plate_for_ocr
        plate = make_rgb_image(32, 120)
        variants = preprocess_plate_for_ocr(plate)
        assert len(variants) >= 3, "Should return at least 3 variants."

    def test_empty_plate_returns_empty_list(self):
        from src.preprocessing import preprocess_plate_for_ocr
        variants = preprocess_plate_for_ocr(None)
        assert variants == []
