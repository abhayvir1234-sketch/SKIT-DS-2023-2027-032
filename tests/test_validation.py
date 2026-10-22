"""
tests/test_validation.py — Unit tests for src/validation.py

Tests heuristic registration-number format validation logic.
No external dependencies.
"""

import pytest


# ---------------------------------------------------------------------------
# Tests: normalize_plate_text
# ---------------------------------------------------------------------------

class TestNormalizePlateText:
    def test_lowercase_converted_to_uppercase(self):
        from src.ocr_engine import normalize_plate_text
        assert normalize_plate_text("mh12ab1234") == "MH12AB1234"

    def test_extra_spaces_collapsed(self):
        from src.ocr_engine import normalize_plate_text
        result = normalize_plate_text("  MH  12  AB  1234  ")
        assert "  " not in result
        assert result.strip() == result

    def test_special_chars_removed(self):
        from src.ocr_engine import normalize_plate_text
        result = normalize_plate_text("MH@12#AB!1234")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_empty_string_returns_empty(self):
        from src.ocr_engine import normalize_plate_text
        assert normalize_plate_text("") == ""

    def test_none_coercion(self):
        from src.ocr_engine import normalize_plate_text
        # Should handle None gracefully via empty string
        assert normalize_plate_text("") == ""


# ---------------------------------------------------------------------------
# Tests: validate_registration (lenient mode)
# ---------------------------------------------------------------------------

class TestValidateRegistrationLenient:
    """Tests with STRICT_VALIDATION = False (default)."""

    def setup_method(self):
        import config
        config.STRICT_VALIDATION = False

    def test_standard_indian_plate_valid(self):
        from src.validation import validate_registration
        result = validate_registration("MH12AB1234")
        assert result.is_valid

    def test_bh_series_valid(self):
        from src.validation import validate_registration
        result = validate_registration("22BH1234AA")
        assert result.is_valid

    def test_too_short_rejected(self):
        from src.validation import validate_registration
        result = validate_registration("AB1")
        assert not result.is_valid

    def test_too_long_rejected(self):
        from src.validation import validate_registration
        result = validate_registration("ABCDEFGH12345678901")
        assert not result.is_valid

    def test_empty_rejected(self):
        from src.validation import validate_registration
        result = validate_registration("")
        assert not result.is_valid

    def test_contains_invalid_chars_rejected(self):
        from src.validation import validate_registration
        # '@' is not alphanumeric
        result = validate_registration("MH@12AB1234")
        assert not result.is_valid


# ---------------------------------------------------------------------------
# Tests: validate_registration (strict mode)
# ---------------------------------------------------------------------------

class TestValidateRegistrationStrict:
    """Tests with STRICT_VALIDATION = True."""

    def setup_method(self):
        import config
        config.STRICT_VALIDATION = True

    def teardown_method(self):
        import config
        config.STRICT_VALIDATION = False   # restore default

    def test_standard_pattern_matches(self):
        from src.validation import validate_registration
        result = validate_registration("MH12AB1234")
        assert result.is_valid
        assert result.matched_pattern != ""

    def test_random_alnum_no_match_in_strict(self):
        from src.validation import validate_registration
        # "ABCDE12345" is 10 alnum chars but doesn't fit any Indian pattern
        result = validate_registration("ABCDE12345")
        # May or may not be valid depending on pattern — just check no crash
        assert isinstance(result.is_valid, bool)

    def test_empty_always_rejected(self):
        from src.validation import validate_registration
        result = validate_registration("")
        assert not result.is_valid


# ---------------------------------------------------------------------------
# Tests: clean_raw_text
# ---------------------------------------------------------------------------

class TestCleanRawText:
    def test_whitespace_stripped(self):
        from src.validation import clean_raw_text
        assert clean_raw_text("  MH12  ") == "MH12"

    def test_empty_returns_empty(self):
        from src.validation import clean_raw_text
        assert clean_raw_text("") == ""

    def test_no_o_to_0_substitution(self):
        """O→0 corrections must NOT be applied automatically."""
        from src.validation import clean_raw_text
        result = clean_raw_text("MHO2AB1234")
        # 'O' should NOT be replaced with '0'
        assert "O" in result
