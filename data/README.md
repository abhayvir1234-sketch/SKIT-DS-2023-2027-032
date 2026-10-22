# Dataset Information

> **Project:** Vehicle Number Plate Recognition System (ALPR)

---

## Overview

This directory is reserved for training/evaluation data. No actual dataset
images are included in the repository due to file size and privacy constraints.

---

## Recommended Datasets

### 1. UFPR-ALPR Dataset
- **Source**: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/
- **Content**: 4,500 images of Brazilian vehicles with plate annotations.
- **Format**: YOLO-compatible bounding boxes.
- **Access**: Free for academic use (request from authors).

### 2. OpenALPR Benchmark
- **Source**: https://github.com/openalpr/benchmark
- **Content**: US and EU plates, benchmark ground truth.
- **License**: Varies per file.

### 3. Roboflow Universe — License Plate Detection
- **Source**: https://universe.roboflow.com (search "license plate detection")
- **Indian plates**: Search "Indian number plate" for India-specific datasets.
- **Format**: Download in YOLOv8 format.

```bash
pip install roboflow
python - <<'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("license-plate-detection")
version = project.version(1)
version.download("yolov8", location="data/plate_dataset")
EOF
```

---

## Custom Indian Plate Dataset

Dedicated large-scale Indian plate datasets are limited in public availability.
We recommend:

1. **Collect images** of vehicles on Indian roads (with owner consent or from
   public-domain sources).
2. **Annotate** using [LabelImg](https://github.com/HumanSignal/labelImg)
   (free, local) or [Roboflow](https://roboflow.com) (cloud-based).
3. **Export** in YOLOv8 format.
4. **Augment** with flips, brightness variation, and blur to simulate real
   conditions.

Aim for:
- Minimum: 500 annotated images
- Recommended: 2,000+ for good generalization
- Split: 80% train / 10% val / 10% test

---

## Privacy and Legal Notes

> ⚠️ Vehicle registration numbers are personal data in many jurisdictions.

- Do **not** include real personal vehicle registration images in this repository.
- If collecting your own dataset, obtain appropriate consent or use images
  where number plates are legitimately visible (e.g., public road images where
  privacy expectations are lower).
- Do not upload data to any external service without verifying the terms of
  service and applicable data-protection regulations.
- This project does **not** connect to any vehicle registration database.
  OCR output is text only and does not reveal owner identity.

---

## Sample Test Images

If sample test images are bundled with this project (placed in `data/samples/`):
- Their source and usage rights must be documented here.
- Images sourced from public domain (e.g., Wikimedia Commons CC0) or
  created specifically for this project may be included.
- Images of real vehicles with readable plates collected without consent
  must NOT be committed to a public repository.

---

*Last updated: 22 October 2026*
