# Team Contributions — Vehicle Number Plate Recognition System

**Implementation Date:** 22 October 2026

---

## Abhayvir Singh — Team Lead (AI/ML & ALPR)

**Expertise:** AI, Machine Learning, ALPR, Computer Vision

### Files Authored
- `config.py` — centralized configuration system
- `src/pipeline.py` — end-to-end pipeline orchestration
- `src/detector.py` — YOLOv8 plate detector with Ultralytics API
- `src/preprocessing.py` — all image preprocessing utilities
- `src/plate_extractor.py` — safe plate crop extraction with padding

### Assigned Tasks and Milestone Status (as of 22 October 2026)

| Task | Recorded Dates | Status |
|------|---------------|--------|
| ALPR environment setup & configuration | 8 Aug – 22 Aug 2026 | ✅ Completed |
| Image preprocessing module | 24 Aug – 3 Sep 2026 | ✅ Completed |
| YOLOv8 model integration | 8 Sep – 28 Sep 2026 | ✅ Implemented (weights required for full test) |
| Vehicle number plate detection | 1 Oct – 16 Oct 2026 | ✅ Implemented (weights required for full test) |
| Plate region extraction | 29 Nov 2026 – 18 Jan 2027 | ⏳ Implemented ahead of milestone |
| Detection pipeline testing | 24 Mar – 29 Mar 2027 | ⏳ Planned milestone — unit tests implemented with mocks |

> **Note on future dates:** Plate region extraction and detection testing have
> recorded milestone dates extending into 2027. The implementation is complete
> as of 22 Oct 2026; full end-to-end testing with real model weights remains
> pending (blocked by model weight acquisition). Dates from the handwritten
> team form may reflect original project planning rather than actual
> implementation schedule.

---

## Himank Devnani — Team Member (OCR & Computer Vision)

**Expertise:** Optical Character Recognition, Computer Vision, NLP

### Files Authored
- `src/ocr_engine.py` — EasyOCR integration, multi-variant recognition, confidence
- `src/validation.py` — Indian registration format heuristic validation

### Assigned Tasks and Milestone Status (as of 22 October 2026)

| Task | Recorded Dates | Status |
|------|---------------|--------|
| OCR engine integration | 8 Aug – 22 Aug 2026 | ✅ Implemented |
| Detected plate image processing | 24 Aug – 3 Sep 2026 | ✅ Implemented |
| Alphanumeric character recognition | 8 Sep – ~5 Nov 2026 | ✅ Implemented (end date from handwritten form unclear) |
| Vehicle registration number extraction | ~5 Nov 2026 – ~24 Mar 2027 | ✅ Implemented ahead of milestone |
| OCR testing & error handling | Handwritten dates unclear | ⚠️ Unit tests implemented; integration test needs real OCR weights |

> **Note:** Some handwritten dates for Himank's tasks were unclear or potentially
> inconsistent. The OCR module is fully implemented. All milestone dates require
> verification with the team or faculty mentor.

---

## Karam Singh Bana — Team Member (Python, Streamlit UI & Backend)

**Expertise:** Python development, Streamlit, web application development

### Files Authored
- `app.py` — complete multi-page Streamlit application (Home, Recognition, History, About)
- `src/annotation.py` — bounding box and label annotation
- `src/report.py` — JSON/CSV report generation and session history
- `utils/image_utils.py` — image format conversion helpers
- `utils/file_utils.py` — file validation, CSV/JSON utilities, path management
- `run.bat` / `run.sh` — application launcher scripts

### Assigned Tasks and Milestone Status (as of 22 October 2026)

| Task | Recorded Dates | Status |
|------|---------------|--------|
| Streamlit UI development | 10 Aug – 3 Sep 2026 | ✅ Completed |
| Image upload functionality | 8 Sep – 28 Sep 2026 | ✅ Completed |
| Recognition result display | 1 Oct – 5 Nov 2026 | ✅ Implemented |
| Annotated output image display | 5 Dec 2026 – 15 Jan 2027 | ✅ Implemented ahead of milestone |
| Confidence score display | 16 Jan – 24 Mar 2027 | ✅ Implemented ahead of milestone |
| System testing & demonstration | 24 Mar – 29 Mar 2027 | ⏳ Planned milestone — unit tests implemented |

---

## Shared Contributions

| Item | All Members |
|------|-------------|
| Documentation | README.md, docs/ |
| Test suite | tests/ (24+ test cases) |
| Configuration | .env.example, pyproject.toml |
| Project structure | requirements.txt, .gitignore |

---

## Implementation Status Summary (22 October 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| Configuration system | ✅ Complete | |
| Image preprocessing | ✅ Complete | |
| YOLOv8 detector | ✅ Implemented | Weights required |
| Plate crop extraction | ✅ Complete | |
| EasyOCR integration | ✅ Implemented | Model auto-downloads |
| Text normalization | ✅ Complete | |
| Format validation | ✅ Complete | |
| Annotation | ✅ Complete | |
| Report generation | ✅ Complete | |
| Streamlit UI | ✅ Complete | All 4 pages functional |
| Unit tests (mocked) | ✅ Complete | 24+ test cases |
| Real-model integration test | ⚠️ Blocked | Requires plate_detector.pt |
| Documentation | ✅ Complete | All docs written |
