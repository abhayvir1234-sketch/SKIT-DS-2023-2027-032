"""
tests/test_upload_validation.py — Tests for file upload validation utilities.

Verifies that the upload validator correctly accepts and rejects files based
on extension, size, and content validity.
"""

import pytest


# ---------------------------------------------------------------------------
# Tests: validate_upload
# ---------------------------------------------------------------------------

class TestValidateUpload:
    _ALLOWED = frozenset({".jpg", ".jpeg", ".png"})
    _MAX = 5 * 1024 * 1024  # 5 MB

    def _valid_jpg_bytes(self) -> bytes:
        """Minimal JPEG-like bytes (real header, tiny content for test)."""
        import io
        import numpy as np
        from PIL import Image
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        return buf.getvalue()

    def test_valid_jpg_accepted(self):
        from utils.file_utils import validate_upload
        data = self._valid_jpg_bytes()
        error = validate_upload("vehicle.jpg", data, self._ALLOWED, self._MAX)
        assert error is None

    def test_valid_png_accepted(self):
        from utils.file_utils import validate_upload
        import io, numpy as np
        from PIL import Image
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        data = buf.getvalue()
        error = validate_upload("vehicle.png", data, self._ALLOWED, self._MAX)
        assert error is None

    def test_unsupported_extension_rejected(self):
        from utils.file_utils import validate_upload
        error = validate_upload("document.pdf", b"fake content", self._ALLOWED, self._MAX)
        assert error is not None
        assert "pdf" in error.lower() or "type" in error.lower() or "allowed" in error.lower()

    def test_empty_file_rejected(self):
        from utils.file_utils import validate_upload
        error = validate_upload("empty.jpg", b"", self._ALLOWED, self._MAX)
        assert error is not None

    def test_oversized_file_rejected(self):
        from utils.file_utils import validate_upload
        big = b"x" * (self._MAX + 1)
        error = validate_upload("huge.jpg", big, self._ALLOWED, self._MAX)
        assert error is not None
        assert "large" in error.lower() or "size" in error.lower() or "mb" in error.lower()

    def test_no_filename_rejected(self):
        from utils.file_utils import validate_upload
        error = validate_upload("", b"data", self._ALLOWED, self._MAX)
        assert error is not None

    def test_bmp_extension_rejected(self):
        from utils.file_utils import validate_upload
        error = validate_upload("photo.bmp", b"fake", self._ALLOWED, self._MAX)
        assert error is not None


# ---------------------------------------------------------------------------
# Tests: bytes_to_bgr (image decode)
# ---------------------------------------------------------------------------

class TestBytesToBGR:
    def test_valid_jpeg_decoded(self):
        import io, numpy as np
        from PIL import Image
        from utils.image_utils import bytes_to_bgr
        arr = np.zeros((50, 80, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="JPEG")
        bgr = bytes_to_bgr(buf.getvalue())
        assert bgr.shape == (50, 80, 3)

    def test_corrupted_bytes_raise_value_error(self):
        from utils.image_utils import bytes_to_bgr
        with pytest.raises(ValueError):
            bytes_to_bgr(b"\x00\x01\x02corrupted garbage")

    def test_empty_bytes_raise_value_error(self):
        from utils.image_utils import bytes_to_bgr
        with pytest.raises(ValueError):
            bytes_to_bgr(b"")
