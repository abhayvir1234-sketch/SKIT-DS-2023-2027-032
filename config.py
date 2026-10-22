"""
config.py — Centralized configuration for the ALPR system.

All tuneable parameters are read from environment variables with documented
defaults. Import this module wherever configuration is needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present (project root)
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).parent.resolve()
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
DATA_DIR: Path = BASE_DIR / "data"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------
#: Filesystem path to the YOLOv8 license-plate detector weights.
#: Set via ALPR_MODEL_PATH environment variable or .env file.
#: If the path does not exist the detector raises a clear ConfigurationError.
MODEL_PATH: str = os.getenv(
    "ALPR_MODEL_PATH",
    str(MODELS_DIR / "plate_detector.pt"),
)

#: Minimum detector confidence score (0–1) to accept a detection.
DETECTION_CONFIDENCE: float = float(os.getenv("ALPR_DETECT_CONF", "0.25"))

#: Image size (pixels) passed to YOLO inference.  Must be a multiple of 32.
INFERENCE_IMG_SIZE: int = int(os.getenv("ALPR_INFER_SIZE", "640"))

#: Device string: "cpu" or "cuda" (or "cuda:0", "mps" for Apple Silicon).
DEVICE: str = os.getenv("ALPR_DEVICE", "cpu")

# ---------------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------------
#: Supported values: "easyocr"
OCR_ENGINE: str = os.getenv("ALPR_OCR_ENGINE", "easyocr")

#: Language list for EasyOCR.
EASYOCR_LANGUAGES: list[str] = os.getenv("ALPR_EASYOCR_LANGS", "en").split(",")

#: Minimum OCR character confidence (0–1) to include a word in the result.
OCR_MIN_CONFIDENCE: float = float(os.getenv("ALPR_OCR_MIN_CONF", "0.1"))

# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
#: Maximum allowed upload size in megabytes.
MAX_UPLOAD_MB: float = float(os.getenv("ALPR_MAX_UPLOAD_MB", "10.0"))
MAX_UPLOAD_BYTES: int = int(MAX_UPLOAD_MB * 1024 * 1024)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

# ---------------------------------------------------------------------------
# Plate crop
# ---------------------------------------------------------------------------
#: Pixels of padding added around a detected plate crop.
CROP_PADDING_PX: int = int(os.getenv("ALPR_CROP_PADDING", "8"))

# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
ANNOTATION_BOX_COLOR: tuple[int, int, int] = (0, 255, 0)   # BGR green
ANNOTATION_TEXT_COLOR: tuple[int, int, int] = (0, 0, 0)    # BGR black
ANNOTATION_BG_COLOR: tuple[int, int, int] = (0, 255, 0)    # BGR green bg
ANNOTATION_FONT_SCALE: float = 0.7
ANNOTATION_THICKNESS: int = 2

# ---------------------------------------------------------------------------
# Validation heuristics
# ---------------------------------------------------------------------------
#: When True, the validation module rejects text that does not match any
#: known Indian registration pattern.  Set False to accept any non-empty text.
STRICT_VALIDATION: bool = os.getenv("ALPR_STRICT_VALIDATION", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Logging / debug
# ---------------------------------------------------------------------------
DEBUG: bool = os.getenv("ALPR_DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Streamlit-specific
# ---------------------------------------------------------------------------
APP_TITLE: str = "Vehicle Number Plate Recognition System"
APP_SUBTITLE: str = "AI-powered vehicle number plate detection and character recognition"
APP_ICON: str = "🚗"
