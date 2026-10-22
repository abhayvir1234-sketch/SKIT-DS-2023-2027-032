"""
src/validation.py — Registration number format validation.

Author: Himank Devnani (Team Member 1) — OCR output cleanup and validation.

This module applies HEURISTIC validation only. It does not verify whether a
registration number is legally issued or belongs to any specific vehicle.
Validation is configurable — see config.STRICT_VALIDATION.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import config

# ---------------------------------------------------------------------------
# Indian vehicle registration formats
# ---------------------------------------------------------------------------
# Standard format: XX 00 XX 0000
# e.g., MH 12 AB 1234, DL 01 CAB 1234 (older long codes also exist)
# BH series: 22 BH 0000 AA (Bharat series, 2021+)
# Two-letter state codes are a 2-char alpha prefix.

_INDIAN_PATTERNS = [
    # Standard: 2 alpha + 2 digit + 1-3 alpha + 1-4 digit (spaces/hyphens optional)
    re.compile(
        r"^[A-Z]{2}\s*\d{2}\s*[A-Z]{1,3}\s*\d{1,4}$",
        re.IGNORECASE,
    ),
    # Bharat (BH) series: 2 digit + BH + 4 digit + 2 alpha
    re.compile(
        r"^\d{2}\s*BH\s*\d{4}\s*[A-Z]{2}$",
        re.IGNORECASE,
    ),
    # Old format: single alpha series (some states), e.g. AP 1234
    re.compile(
        r"^[A-Z]{2}\s*\d{1,4}$",
        re.IGNORECASE,
    ),
    # Temporary / transit plates may vary — accept any 5-12 alnum sequence
    re.compile(
        r"^[A-Z0-9]{5,12}$",
        re.IGNORECASE,
    ),
]

# Minimum / maximum characters in a valid plate (after whitespace removal)
_MIN_CHARS = 4
_MAX_CHARS = 16


@dataclass
class ValidationResult:
    """Result of validating a candidate registration string."""

    #: Validated / normalised registration text.
    text: str

    #: True when *text* passes at least one pattern in the selected mode.
    is_valid: bool

    #: Human-readable reason why validation passed or failed.
    message: str

    #: Name of the pattern that matched (or "" if none matched).
    matched_pattern: str = ""


def validate_registration(text: str) -> ValidationResult:
    """Validate a candidate registration number string.

    When :data:`config.STRICT_VALIDATION` is ``True``, the text must match
    one of the documented Indian plate patterns.  When ``False``, any
    non-empty alphanumeric sequence of plausible length is accepted.

    Args:
        text: Normalized registration-number candidate (uppercase, no leading
              or trailing whitespace).

    Returns:
        A :class:`ValidationResult`.
    """
    if not text:
        return ValidationResult(
            text=text,
            is_valid=False,
            message="Empty string — no text to validate.",
        )

    # Remove spaces and hyphens for length check
    compact = re.sub(r"[\s\-]", "", text)

    if len(compact) < _MIN_CHARS:
        return ValidationResult(
            text=text,
            is_valid=False,
            message=f"Too short ({len(compact)} chars; minimum {_MIN_CHARS}).",
        )

    if len(compact) > _MAX_CHARS:
        return ValidationResult(
            text=text,
            is_valid=False,
            message=f"Too long ({len(compact)} chars; maximum {_MAX_CHARS}).",
        )

    if not config.STRICT_VALIDATION:
        # Lenient: accept any non-empty alphanumeric sequence in range
        if re.fullmatch(r"[A-Z0-9\s\-]+", text, re.IGNORECASE):
            return ValidationResult(
                text=text,
                is_valid=True,
                message="Accepted (lenient validation mode).",
                matched_pattern="lenient",
            )
        return ValidationResult(
            text=text,
            is_valid=False,
            message="Contains invalid characters for a registration number.",
        )

    # Strict mode: test against known Indian patterns
    for idx, pattern in enumerate(_INDIAN_PATTERNS):
        if pattern.fullmatch(text):
            return ValidationResult(
                text=text,
                is_valid=True,
                message="Matches known Indian registration format.",
                matched_pattern=f"pattern_{idx}",
            )

    return ValidationResult(
        text=text,
        is_valid=False,
        message=(
            "Does not match a known Indian registration format. "
            "Result may still be the correct plate text — enable lenient mode "
            "to accept it."
        ),
    )


def clean_raw_text(raw: str) -> str:
    """Light cleaning of raw OCR output before normalization.

    - Removes leading/trailing whitespace.
    - Replaces common OCR confusion characters only when doing so follows a
      documented rule (none applied by default — explicit, auditable changes
      only).

    Args:
        raw: Raw string from OCR engine.

    Returns:
        Lightly cleaned string.
    """
    return raw.strip()
