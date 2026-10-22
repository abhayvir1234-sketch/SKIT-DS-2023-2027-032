# Vehicle Number Plate Recognition System (ALPR)

> **AI-powered vehicle number plate detection and character recognition**  
> College Engineering Project | Computer Vision & Machine Learning  
> Implementation Date: 22 October 2026

---

## Abstract

This project implements an Automatic License Plate Recognition (ALPR) system that detects vehicle number plates in uploaded images using a YOLOv8 deep-learning object detector and extracts registration text using EasyOCR optical character recognition. The system provides a polished Streamlit web interface supporting image upload, plate visualization, annotated output, confidence reporting, session history, and downloadable reports. All processing runs locally on CPU without requiring internet access after initial model download.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [System Workflow](#system-workflow)
5. [Architecture](#architecture)
6. [Technology Stack](#technology-stack)
7. [Project Structure](#project-structure)
8. [Installation — Windows](#installation--windows)
9. [Installation — Linux / macOS](#installation--linux--macos)
10. [Model Weight Configuration](#model-weight-configuration)
11. [Running the Application](#running-the-application)
12. [Usage Instructions](#usage-instructions)
13. [Testing](#testing)
14. [Limitations](#limitations)
15. [Future Scope](#future-scope)
16. [Team Member Responsibilities](#team-member-responsibilities)
17. [Troubleshooting](#troubleshooting)
18. [License & Attribution](#license--attribution)

---

## Introduction

Automatic License Plate Recognition (ALPR) is a computer vision technique that combines object detection, image preprocessing, and optical character recognition to read vehicle registration numbers without manual intervention. ALPR is used in traffic surveillance, toll collection, parking management, and law enforcement.

This project demonstrates a complete end-to-end ALPR pipeline implemented in Python using modern deep learning and OCR frameworks, wrapped in a user-friendly Streamlit web interface.

---

## Problem Statement

Manual reading of vehicle license plates is error-prone, slow, and impractical in high-traffic or surveillance scenarios. A software system that can automatically detect plate regions in images and extract the alphanumeric registration text enables numerous practical applications while serving as a demonstration of applied computer vision techniques.

---

## Objectives

1. Detect licence-plate regions in uploaded vehicle images using a YOLOv8 model trained specifically for number plates.
2. Extract and enhance the plate crop for OCR processing.
3. Recognize the alphanumeric registration number using EasyOCR with multi-variant preprocessing.
4. Display annotated images, confidence scores, and downloadable recognition reports.
5. Provide a clean, professional Streamlit interface suitable for a college project demonstration.

---

## System Workflow

```
Upload Image (JPG/PNG)
        ↓
File Validation (type, size, readability)
        ↓
Image Decode (OpenCV BGR)
        ↓
Preprocessing (resize for YOLO)
        ↓
YOLOv8 Plate Detection
        ↓
Bounding Box Clipping & Sorting (confidence order)
        ↓
Plate Crop Extraction (with padding, boundary-safe)
        ↓
OCR Variant Generation (5 preprocessing variants)
        ↓
EasyOCR Recognition (best variant selected by confidence)
        ↓
Text Normalization & Format Validation (heuristic)
        ↓
Image Annotation (bounding box + label drawn)
        ↓
Result Display + Downloads (JSON / CSV / annotated image)
        ↓
Session History Update
```

---

## Architecture

The system is organized into independent, testable modules:

| Module | File | Author | Responsibility |
|--------|------|--------|---------------|
| Pipeline | `src/pipeline.py` | Abhayvir Singh | Orchestration |
| Detection | `src/detector.py` | Abhayvir Singh | YOLOv8 inference |
| Preprocessing | `src/preprocessing.py` | Abhayvir Singh | Image enhancement |
| Plate Extractor | `src/plate_extractor.py` | Abhayvir Singh | Safe crop extraction |
| OCR Engine | `src/ocr_engine.py` | Himank Devnani | EasyOCR integration |
| Validation | `src/validation.py` | Himank Devnani | Format heuristics |
| Annotation | `src/annotation.py` | Karam Singh Bana | Result visualization |
| Report | `src/report.py` | Karam Singh Bana | JSON/CSV export |
| App UI | `app.py` | Karam Singh Bana | Streamlit interface |

---

## Technology Stack

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Object Detection | `ultralytics` | ≥8.0 | YOLOv8 plate detection |
| OCR | `easyocr` | ≥1.7 | Text recognition |
| Web UI | `streamlit` | ≥1.28 | Application interface |
| Image Processing | `opencv-python-headless` | ≥4.8 | CV operations |
| Image I/O | `Pillow` | ≥10.0 | Format conversion |
| Numerics | `numpy` | ≥1.24 | Array operations |
| Data | `pandas` | ≥2.0 | Reports/history |
| Testing | `pytest`, `pytest-mock` | ≥7.4 | Automated tests |
| Config | `python-dotenv` | ≥1.0 | Environment variables |

---

## Project Structure

```
vehicle-number-plate-recognition/
├── app.py                    # Main Streamlit application
├── config.py                 # Centralized configuration
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Pytest configuration
├── README.md                 # This file
├── .env.example              # Configuration template
├── .gitignore                # Git exclusion rules
├── run.bat                   # Windows launcher
├── run.sh                    # Linux/macOS launcher
│
├── src/                      # Core pipeline modules
│   ├── __init__.py
│   ├── pipeline.py           # End-to-end orchestrator
│   ├── detector.py           # YOLOv8 plate detector
│   ├── preprocessing.py      # Image preprocessing utilities
│   ├── plate_extractor.py    # Plate crop extraction
│   ├── ocr_engine.py         # EasyOCR integration
│   ├── validation.py         # Registration format validation
│   ├── annotation.py         # Bounding box annotation
│   └── report.py             # Report generation
│
├── utils/                    # Shared utilities
│   ├── __init__.py
│   ├── image_utils.py        # Format conversion helpers
│   └── file_utils.py         # File/path utilities
│
├── tests/                    # Automated test suite
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   ├── test_pipeline.py
│   ├── test_upload_validation.py
│   └── test_report.py
│
├── docs/                     # Project documentation
│   ├── project_overview.md
│   ├── methodology.md
│   ├── architecture.md
│   ├── testing_report.md
│   ├── team_contributions.md
│   └── viva_questions.md
│
├── models/
│   └── README.md             # Weight download instructions
│
├── data/
│   └── README.md             # Dataset documentation
│
└── outputs/                  # Auto-generated outputs
    └── .gitkeep
```

---

## Installation — Windows

### Prerequisites
- Python 3.10 or newer — https://python.org
- Git (optional, for cloning)
- 4 GB free disk space (for model weights and OCR models)

### Step 1: Clone / Download the Project
```cmd
git clone <repository-url> vehicle-number-plate-recognition
cd vehicle-number-plate-recognition
```
Or download and extract the ZIP.

### Step 2: Create a Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```cmd
pip install -r requirements.txt
```

> **Note on PyTorch / GPU**: The above installs CPU PyTorch automatically. For GPU support, install CUDA-enabled PyTorch first:
> ```cmd
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

### Step 4: Configure Environment
```cmd
copy .env.example .env
```
Edit `.env` and set `ALPR_MODEL_PATH` to your downloaded weights file.

---

## Installation — Linux / macOS

```bash
git clone <repository-url> vehicle-number-plate-recognition
cd vehicle-number-plate-recognition
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set ALPR_MODEL_PATH
```

---

## Model Weight Configuration

> ⚠️ **A licence-plate-specific YOLOv8 model is required.** A generic COCO detector cannot detect number plates.

See [`models/README.md`](models/README.md) for complete download instructions.

**Quick setup:**

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="keremberke/license-plate-detection",
    filename="best.pt",
    local_dir="models"
)
import shutil, pathlib
shutil.copy(path, "models/plate_detector.pt")
print("Model saved to models/plate_detector.pt")
EOF
```

Then verify your `.env`:
```
ALPR_MODEL_PATH=models/plate_detector.pt
```

---

## Running the Application

### Quick Start (Windows)
```cmd
run.bat
```

### Manual Start
```cmd
venv\Scripts\activate        # Windows
# source venv/bin/activate  # Linux/macOS
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage Instructions

1. Navigate to the **Recognition** page in the sidebar.
2. Click **Browse files** and select a JPG or PNG vehicle image (max 10 MB).
3. Adjust the **detection confidence threshold** slider if needed.
4. Click **Run Recognition**.
5. View results:
   - **Metrics**: number of plates detected, recognized text, confidence scores.
   - **Original / Plate Crop / Annotated** image tabs.
   - Download buttons for the annotated image and JSON/CSV reports.
6. Check the **History** page to review past recognitions in the current session.

---

## Testing

### Run all unit tests
```cmd
pytest tests/ -v
```

### Run a specific test file
```cmd
pytest tests/test_preprocessing.py -v
pytest tests/test_validation.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_upload_validation.py -v
pytest tests/test_report.py -v
```

### Syntax check all modules
```cmd
python -m py_compile src/pipeline.py src/detector.py src/preprocessing.py src/plate_extractor.py src/ocr_engine.py src/validation.py src/annotation.py src/report.py config.py app.py utils/image_utils.py utils/file_utils.py
```

### Import check
```cmd
python -c "from src.pipeline import ALPRPipeline; from src.ocr_engine import EasyOCREngine; print('Imports OK')"
```

> **Note**: Tests in `test_pipeline.py` mock all model calls. They pass without model weights.  
> Real-model integration testing requires a downloaded `plate_detector.pt` and internet access for EasyOCR weights.

---

## Limitations

1. **Model weights required**: The application cannot detect plates without a trained YOLOv8 weight file. See `models/README.md`.
2. **OCR accuracy**: Heavily blurred, occluded, or very small plates may not be recognized correctly.
3. **Indian plates**: Strict validation is tuned for common Indian formats. International formats may show validation warnings (this does not discard the OCR result).
4. **Single best plate**: Only the highest-confidence plate is shown in the detail view; all plates are annotated on the output image.
5. **CPU inference speed**: Expect 2–10 seconds per image on CPU-only hardware.
6. **Static images only**: Real-time video is not supported in this version.

---

## Future Scope

- Real-time webcam/video stream processing.
- Multi-plate detailed reporting (one result per plate).
- Indian state-code validation lookup table.
- On-device model optimization (ONNX export, TensorRT).
- Mobile-responsive deployment via Streamlit Community Cloud.
- Vehicle type classification alongside plate detection.
- Multi-language plate support (Hindi script, Arabic numerals).

---

## Team Member Responsibilities

### Abhayvir Singh — Team Lead (AI/ML & ALPR)
| Task | Files |
|------|-------|
| ALPR setup & configuration | `config.py`, `models/` |
| Image preprocessing | `src/preprocessing.py` |
| YOLOv8 integration | `src/detector.py` |
| Plate detection pipeline | `src/detector.py` |
| Plate region extraction | `src/plate_extractor.py` |
| Pipeline integration | `src/pipeline.py` |

### Himank Devnani — OCR & Computer Vision
| Task | Files |
|------|-------|
| OCR engine integration | `src/ocr_engine.py` |
| Character recognition | `src/ocr_engine.py` |
| Text normalization | `src/ocr_engine.py` |
| Format validation | `src/validation.py` |
| OCR testing | `tests/test_validation.py` |

### Karam Singh Bana — UI, Python & Backend
| Task | Files |
|------|-------|
| Streamlit UI | `app.py` |
| Upload validation | `utils/file_utils.py` |
| Result display | `app.py` |
| Annotation | `src/annotation.py` |
| Report generation | `src/report.py` |
| Image utilities | `utils/image_utils.py` |
| System testing | `tests/test_report.py`, `tests/test_upload_validation.py` |

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| `ConfigurationError: Model weight file not found` | `plate_detector.pt` missing | Follow `models/README.md` to download weights |
| `EasyOCR is not installed` | Missing package | `pip install easyocr` |
| `ModuleNotFoundError: No module named 'ultralytics'` | Missing package | `pip install ultralytics` |
| Streamlit shows blank page | Port conflict | Try `streamlit run app.py --server.port 8502` |
| OCR returns empty text | Plate too blurry/small | Try a higher-resolution image |
| Very slow inference | CPU-only machine | Normal; GPU not required. Inference may take 5–10 s |
| `No plate detected` on a clear image | Confidence threshold too high | Lower the slider below 0.25 |
| Tests fail with `ModuleNotFoundError` | Wrong working directory | Run `pytest` from the project root |

---

## License & Attribution

- **Ultralytics YOLOv8**: AGPL-3.0 (research and educational use).
- **EasyOCR**: Apache 2.0 License.
- **OpenCV**: Apache 2.0 License.
- **Streamlit**: Apache 2.0 License.
- **Model weights**: Check the specific model card for license details (see `models/README.md`).

This project was built for **educational purposes** as part of a college engineering curriculum. It does not connect to any vehicle registration database and cannot verify vehicle ownership.
