# System Architecture — Vehicle Number Plate Recognition System

---

## Architecture Overview

The ALPR system is built as a set of loosely coupled Python modules. The Streamlit
application (`app.py`) acts as the presentation layer; the `src/` package contains
the processing pipeline; `utils/` provides shared helpers; `config.py` centralizes
all tunable parameters.

---

## Module Interaction Flowchart

```mermaid
flowchart TD
    A([User Uploads Image\nJPG / PNG]) --> B[File Validation\nutils/file_utils.py]
    B -->|Invalid| ERR1([Error Message to User])
    B -->|Valid| C[Image Decode\nutils/image_utils.py\nbytes_to_bgr]
    C -->|Corrupted| ERR2([Decode Error to User])
    C -->|OK| D[Preprocessing\nsrc/preprocessing.py\nresize for YOLO]
    D --> E[YOLOv8 Detection\nsrc/detector.py\nPlateDetector.detect]
    E -->|No weights| ERR3([ConfigurationError to User])
    E -->|No plate| F1[annotate_no_detection\nsrc/annotation.py]
    E -->|Plate found| G[BBox Clipping\nConfidence Sort]
    G --> H[Plate Crop Extraction\nsrc/plate_extractor.py\npadding + boundary-safe]
    H --> I[OCR Variant Generation\n5 preprocessing variants\nsrc/plate_extractor.py]
    I --> J[EasyOCR Recognition\nsrc/ocr_engine.py\nbest variant selected]
    J -->|OCR fails| K1[Advisory message\nshown to user]
    J -->|OCR OK| K[Text Normalization\nsrc/ocr_engine.py\nnormalize_plate_text]
    K --> L[Format Validation\nsrc/validation.py\nvalidate_registration]
    L --> M[Image Annotation\nsrc/annotation.py\nannotate_image]
    M --> N[Build Report\nsrc/report.py\nbuild_recognition_report]
    N --> O([Result Display\napp.py\nMetrics + Images + Downloads])
    F1 --> O
    K1 --> M
```

---

## Module Responsibility Table

| File | Author | Function |
|------|--------|---------|
| `config.py` | Abhayvir Singh | Centralized configuration from env vars |
| `src/pipeline.py` | Abhayvir Singh | End-to-end orchestration |
| `src/detector.py` | Abhayvir Singh | YOLOv8 inference, PlateDetection dataclass |
| `src/preprocessing.py` | Abhayvir Singh | CLAHE, thresholding, resize, sharpening |
| `src/plate_extractor.py` | Abhayvir Singh | Safe crop extraction with padding |
| `src/ocr_engine.py` | Himank Devnani | EasyOCR wrapper, multi-variant selection |
| `src/validation.py` | Himank Devnani | Indian plate format heuristics |
| `src/annotation.py` | Karam Singh Bana | Bounding box + label drawing |
| `src/report.py` | Karam Singh Bana | JSON/CSV report, session history |
| `app.py` | Karam Singh Bana | Streamlit multi-page UI |
| `utils/image_utils.py` | Karam Singh Bana | PIL↔BGR conversion, resize |
| `utils/file_utils.py` | Karam Singh Bana | Upload validation, CSV/JSON helpers |

---

## Data Flow

1. **Raw bytes** arrive from the Streamlit file uploader.
2. `validate_upload()` checks extension, size, and emptiness.
3. `bytes_to_bgr()` decodes the bytes into an OpenCV BGR array.
4. `preprocess_for_detection()` downsizes very large images to protect memory.
5. `PlateDetector.detect()` passes the array to YOLOv8; the result is a list of
   `PlateDetection` objects (bbox, confidence, class_name).
6. `extract_plate_crop()` slices the detected region with padding from the full image.
7. `get_ocr_variants()` generates 5 preprocessed versions of the crop.
8. `EasyOCREngine.recognize()` runs EasyOCR on each variant and returns the
   `OCRResult` with the highest average word confidence.
9. `normalize_plate_text()` uppercases the text and strips illegal characters.
10. `validate_registration()` applies heuristic Indian format matching.
11. `annotate_image()` draws bounding boxes and labels on a copy of the original.
12. `build_recognition_report()` packages all results into a dict.
13. Streamlit renders images, metrics, and download buttons.

---

## Configuration Architecture

All tuneable values are in `config.py`, which reads from environment variables
(set via `.env` or shell). No magic numbers are scattered through source files.

```
.env / environment
      ↓
  config.py
      ↓
  ┌───┴────┐
  detector  ocr_engine  pipeline  app
```

---

## Error Isolation Strategy

Each stage of the pipeline is wrapped in `try/except`. A failure at any stage
produces a descriptive `error_message` and a `user_message` in the
`PipelineResult`. The Streamlit UI reads these fields and shows appropriate
error banners. Failures never silently show a stale previous result.
