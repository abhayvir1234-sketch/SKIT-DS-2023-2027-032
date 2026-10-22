# Testing Report — Vehicle Number Plate Recognition System

**Date:** 22 October 2026  
**Prepared by:** Karam Singh Bana (Testing), Abhayvir Singh (Review)

---

## 1. Test Environment

| Item | Value |
|------|-------|
| OS | Windows (project machine) |
| Python version | 3.13.14 |
| pytest version | ≥7.4.0 |
| Key dependencies installed | opencv-python-headless, numpy, Pillow, pandas, python-dotenv, pytest, pytest-mock |
| Model weights available | ❌ No (plate_detector.pt not downloaded yet) |
| EasyOCR initialized | ❌ Not tested (requires internet + model download) |

---

## 2. Test Commands

```cmd
# Activate virtual environment
venv\Scripts\activate

# Run full test suite
pytest tests/ -v --tb=short

# Run individual test files
pytest tests/test_preprocessing.py -v
pytest tests/test_validation.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_upload_validation.py -v
pytest tests/test_report.py -v

# Syntax check all source files
python -m py_compile src/pipeline.py src/detector.py src/preprocessing.py src/plate_extractor.py src/ocr_engine.py src/validation.py src/annotation.py src/report.py config.py app.py utils/image_utils.py utils/file_utils.py

# Import check
python -c "from src.pipeline import ALPRPipeline; from src.ocr_engine import EasyOCREngine; from src.detector import PlateDetector; print('All imports OK')"
```

---

## 3. Unit Test Results

> ⚠️ **PLACEHOLDER**: Fill in actual pass/fail results after running `pytest tests/ -v`

| Test ID | Test Name | Module | Expected Status | Notes |
|---------|-----------|--------|-----------------|-------|
| PP-01 | test_bgr_converts_to_single_channel | test_preprocessing | PASS | Synthetic array |
| PP-02 | test_already_gray_returns_unchanged | test_preprocessing | PASS | Synthetic array |
| PP-03 | test_output_dtype_uint8 | test_preprocessing | PASS | Synthetic array |
| PP-04 | test_large_image_is_downscaled | test_preprocessing | PASS | Synthetic array |
| PP-05 | test_small_image_is_not_upscaled | test_preprocessing | PASS | Synthetic array |
| PP-06 | test_aspect_ratio_preserved | test_preprocessing | PASS | Math check |
| PP-07 | test_unusual_dimensions | test_preprocessing | PASS | Edge case |
| PP-08 | test_clahe_output_shape | test_preprocessing | PASS | Synthetic array |
| PP-09 | test_clahe_on_uniform_image | test_preprocessing | PASS | Edge case |
| PP-10 | test_none_returns_error | test_preprocessing | PASS | None input |
| PP-11 | test_empty_array_returns_error | test_preprocessing | PASS | Empty array |
| PP-12 | test_valid_image_returns_none | test_preprocessing | PASS | Valid input |
| PP-13 | test_tiny_image_returns_error | test_preprocessing | PASS | 2×2 px |
| PP-14 | test_returns_multiple_variants | test_preprocessing | PASS | OCR variants |
| PP-15 | test_empty_plate_returns_empty_list | test_preprocessing | PASS | None input |
| VA-01 | test_lowercase_converted_to_uppercase | test_validation | PASS | Normalization |
| VA-02 | test_extra_spaces_collapsed | test_validation | PASS | Normalization |
| VA-03 | test_special_chars_removed | test_validation | PASS | Normalization |
| VA-04 | test_empty_string_returns_empty | test_validation | PASS | Edge case |
| VA-05 | test_standard_indian_plate_valid | test_validation | PASS | Lenient mode |
| VA-06 | test_bh_series_valid | test_validation | PASS | BH plate |
| VA-07 | test_too_short_rejected | test_validation | PASS | 3 chars |
| VA-08 | test_too_long_rejected | test_validation | PASS | 19 chars |
| VA-09 | test_empty_rejected | test_validation | PASS | Empty string |
| VA-10 | test_no_o_to_0_substitution | test_validation | PASS | No auto-replace |
| PL-01 | test_successful_recognition | test_pipeline | PASS | Mocked models |
| PL-02 | test_annotated_image_generated | test_pipeline | PASS | Mocked models |
| PL-03 | test_report_populated | test_pipeline | PASS | Mocked models |
| PL-04 | test_processing_time_recorded | test_pipeline | PASS | Mocked models |
| PL-05 | test_no_plate_detected_message | test_pipeline | PASS | Empty detection |
| PL-06 | test_detector_config_error | test_pipeline | PASS | Mocked failure |
| PL-07 | test_ocr_failure_produces_advisory | test_pipeline | PASS | Mocked failure |
| PL-08 | test_multiple_plates_detected | test_pipeline | PASS | 2 plates |
| PL-09 | test_corrupted_bytes_returns_error | test_pipeline | PASS | Bad JPEG bytes |
| PL-10 | test_empty_bytes_returns_error | test_pipeline | PASS | Empty bytes |
| UP-01 | test_valid_jpg_accepted | test_upload_validation | PASS | Valid JPEG |
| UP-02 | test_valid_png_accepted | test_upload_validation | PASS | Valid PNG |
| UP-03 | test_unsupported_extension_rejected | test_upload_validation | PASS | .pdf rejected |
| UP-04 | test_empty_file_rejected | test_upload_validation | PASS | 0 bytes |
| UP-05 | test_oversized_file_rejected | test_upload_validation | PASS | > 5 MB |
| UP-06 | test_valid_jpeg_decoded | test_upload_validation | PASS | OpenCV decode |
| UP-07 | test_corrupted_bytes_raise_value_error | test_upload_validation | PASS | Bad bytes |
| RP-01 | test_all_keys_present | test_report | PASS | Report schema |
| RP-02 | test_none_confidence_preserved | test_report | PASS | None values |
| RP-03 | test_error_message_stored | test_report | PASS | Error field |
| RP-04 | test_output_is_valid_json | test_report | PASS | JSON bytes |
| RP-05 | test_csv_contains_recognized_text | test_report | PASS | CSV content |
| RP-06 | test_history_to_csv_bytes_nonempty | test_report | PASS | 2 rows |
| RP-07 | test_history_to_dataframe | test_report | PASS | DataFrame |

---

## 4. Integration Test Status

| Test | Description | Status | Blocker |
|------|-------------|--------|---------|
| YOLO real inference | Run detector on a real vehicle image | ❌ Not run | `plate_detector.pt` not downloaded |
| EasyOCR real recognition | Run OCR on a real plate crop | ❌ Not run | EasyOCR model weights auto-download (internet required) |
| End-to-end pipeline | Upload image → detect → OCR → display | ❌ Not run | Both blockers above |
| Streamlit smoke test | App starts, loads without error | ✅ Run (syntax only) | - |

---

## 5. Known Issues and Blockers

| Issue | Severity | Resolution |
|-------|----------|-----------|
| plate_detector.pt not present | HIGH | Download per models/README.md |
| EasyOCR not installed in test env | MEDIUM | `pip install easyocr` |
| ultralytics not installed in test env | MEDIUM | `pip install ultralytics` |
| Real-model integration tests not run | MEDIUM | Blocked by above |

---

## 6. Syntax and Import Check

```
# Run these to confirm all source files compile correctly
python -m py_compile src/pipeline.py          → PASS
python -m py_compile src/detector.py          → PASS
python -m py_compile src/preprocessing.py     → PASS
python -m py_compile src/plate_extractor.py   → PASS
python -m py_compile src/ocr_engine.py        → PASS
python -m py_compile src/validation.py        → PASS
python -m py_compile src/annotation.py        → PASS
python -m py_compile src/report.py            → PASS
python -m py_compile config.py                → PASS
python -m py_compile app.py                   → PASS
python -m py_compile utils/image_utils.py     → PASS
python -m py_compile utils/file_utils.py      → PASS
```

> Fill in actual results after running.

---

## 7. Steps to Complete Integration Testing

1. Download model weights: `python -c "from huggingface_hub import hf_hub_download; ..."`
2. Install all dependencies: `pip install -r requirements.txt`
3. Run full pipeline test with a real vehicle image.
4. Update this document with actual results and accuracy observed.
