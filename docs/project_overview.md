# Project Overview — Vehicle Number Plate Recognition System

**Team:** Abhayvir Singh (Lead), Himank Devnani (OCR), Karam Singh Bana (UI)  
**Date:** 22 October 2026

---

## Abstract

This project implements a Vehicle Number Plate Recognition (ALPR) system using deep learning and optical character recognition. A YOLOv8 model detects the licence-plate region in a vehicle image; the crop is passed through EasyOCR to extract the alphanumeric registration number. The system is presented as a Streamlit web application supporting image upload, result display, annotated output, confidence reporting, session history, and CSV/JSON export. The pipeline is modular, tested with pytest, and designed to run on a standard student laptop (CPU only).

---

## Introduction

Automatic Licence Plate Recognition (ALPR) combines object detection and optical character recognition to read vehicle registration numbers without human intervention. Applications include traffic surveillance, toll collection, parking enforcement, and fleet management. Modern ALPR systems use convolutional neural networks for both the detection and recognition stages, achieving accuracy comparable to human readers under good conditions.

This project demonstrates a practical ALPR pipeline using current open-source tools: YOLOv8 (Ultralytics, 2023) for detection and EasyOCR (JaidedAI, 2020) for text recognition.

---

## Problem Statement

Manual identification of vehicle number plates is slow, error-prone, and unable to scale to high-volume scenarios. An automated system is needed that can:
- Reliably locate the number plate region in diverse vehicle images.
- Extract legible registration text under varying illumination and image quality.
- Present results clearly to an end-user without requiring specialist knowledge.

---

## Objectives

1. Detect licence-plate bounding boxes in uploaded vehicle images using YOLOv8.
2. Extract and preprocess the plate crop for optimal OCR input.
3. Recognize the alphanumeric registration text using EasyOCR with multiple preprocessing variants.
4. Validate extracted text against common Indian registration formats (configurable).
5. Display annotated images, confidence scores, and downloadable reports via Streamlit.

---

## Proposed Solution

The solution is a Python web application with the following components:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Detection | YOLOv8 (Ultralytics) | Locate plate bounding box |
| Preprocessing | OpenCV | Enhance plate image for OCR |
| OCR | EasyOCR | Read alphanumeric text |
| UI | Streamlit | Upload, display, export |
| Config | python-dotenv | Runtime configuration |
| Tests | pytest + pytest-mock | Quality assurance |

---

## Technology Stack

| Library | Version | Role |
|---------|---------|------|
| ultralytics | ≥8.0 | YOLOv8 object detection |
| easyocr | ≥1.7 | OCR text recognition |
| opencv-python-headless | ≥4.8 | Image processing |
| streamlit | ≥1.28 | Web interface |
| Pillow | ≥10.0 | Image I/O |
| numpy | ≥1.24 | Array computation |
| pandas | ≥2.0 | Data export |
| python-dotenv | ≥1.0 | Environment config |
| pytest | ≥7.4 | Automated testing |

---

## Expected Outcomes

- A working ALPR pipeline that processes real uploaded images.
- A polished Streamlit interface suitable for college demonstration.
- A test suite with 24+ test cases covering all major pipeline paths.
- Complete documentation (README, methodology, architecture, viva Q&A).
- Honest reporting of limitations (model weights required, OCR limitations).
