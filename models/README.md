# Model Weights — Setup Instructions

> **Project:** Vehicle Number Plate Recognition System (ALPR)
> **Team:** Abhayvir Singh (Lead/Detection), Himank Devnani (OCR), Karam Singh Bana (UI/Backend)
> **Date:** 22 October 2026

---

## Overview

This directory stores the trained model weight files required by the ALPR pipeline. The YOLOv8 detector used in this project must be a model **specifically trained for license plate detection**. The weights file is **not included in the repository** due to file size and licensing constraints — you must download or train the weights yourself using the instructions below.

---

## WARNING — Read Before Downloading Any Model

> Do NOT use a generic COCO-pretrained YOLOv8 model. The standard YOLOv8 weights (e.g., `yolov8n.pt`, `yolov8s.pt` downloaded directly from Ultralytics) are trained on the COCO dataset, which contains 80 common object classes such as cars, people, and animals. **License plates are NOT among those 80 classes.** Using a generic COCO model will result in zero detections and the pipeline will fail silently. You must use a model that has been fine-tuned specifically on a license plate detection dataset.

---

## 1. What Model Is Needed

The project requires a **YOLOv8 object detection model** (`.pt` format) fine-tuned to detect vehicle number plates (bounding box class: `license-plate` or similar). The model should be capable of:

- Detecting one or more license plates in a single image.
- Returning tight bounding boxes around the plate region.
- Functioning at a reasonable inference speed (real-time or near real-time on CPU/GPU).

### Recommended Pre-Trained Sources

| Source | Model / Link | Notes |
|---|---|---|
| **Hugging Face** | `keremberke/license-plate-detection` | YOLOv8-based, well-documented |
| **Roboflow Universe** | Search `license plate detection yolov8` | Multiple variants available |
| **Custom-trained** | See Section 6 | Best accuracy for Indian plates |

---

## 2. How to Download the Model

### Option A — Download from Hugging Face (Recommended)

The model `keremberke/license-plate-detection` is hosted on Hugging Face and can be downloaded programmatically using the `huggingface_hub` Python library.

**Step 1: Install the library**

```bash
pip install huggingface_hub
```

**Step 2: Run the download snippet**

```python
from huggingface_hub import hf_hub_download
import shutil
import os

# Download the YOLOv8 best.pt weights from the model repository
model_path = hf_hub_download(
    repo_id="keremberke/license-plate-detection",
    filename="best.pt",           # The fine-tuned YOLOv8 weights file
    repo_type="model",
    local_dir="./models"          # Save directly into the models/ directory
)

# Rename to the expected filename used by the ALPR pipeline
dest = os.path.join("models", "plate_detector.pt")
if not os.path.exists(dest):
    shutil.copy(model_path, dest)

print(f"Model downloaded to: {dest}")
```

> NOTE: The download may take a few minutes depending on your internet speed. The weights file is typically between 6 MB (nano) and 25 MB (small) depending on the model variant available in the repository. Check the model card on Hugging Face for the exact variant details.

**Step 3: Verify the download**

```python
from ultralytics import YOLO

model = YOLO("models/plate_detector.pt")
print(model.info())  # Should display model summary without errors
```

---

### Option B — Download from Roboflow Universe

Roboflow Universe hosts hundreds of community-trained license plate detection models. To find a suitable model:

1. Visit https://universe.roboflow.com
2. Search for **"license plate detection yolov8"**
3. Filter by **Model Type: YOLOv8** and **Task: Object Detection**
4. Select a model with high mAP (look for mAP@50 > 0.85 if available)
5. Click **"Deploy"** -> **"Download"** -> select **YOLOv8 PyTorch** format
6. Save the downloaded `.pt` file as `models/plate_detector.pt`

Alternatively, use the Roboflow Python SDK to download directly:

```bash
pip install roboflow
```

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("your-workspace").project("license-plate-detection")
version = project.version(1)

# Download model weights (YOLOv8 format)
version.download("yolov8")
```

> TIP: Sign up for a free Roboflow account at https://roboflow.com to get your API key. The free tier is sufficient for downloading public datasets and model weights.

---

## 3. Where to Place the File

Once downloaded, the weights file **must** be placed at the following path relative to the project root:

```
FinalProject/
+-- models/
    +-- plate_detector.pt    <- Place your downloaded .pt file here
```

**Full path on Windows:**

```
d:\FinalProject\models\plate_detector.pt
```

> IMPORTANT: The filename must be exactly `plate_detector.pt`. The pipeline code in `src/detector.py` and the configuration loader both reference this filename by default. If you use a different filename, you must update the `ALPR_MODEL_PATH` environment variable (see Section 4).

---

## 4. How to Configure the Model Path

The ALPR application reads the model path from an environment variable. This allows you to switch between different model versions without modifying source code.

**Step 1: Copy the example environment file**

```bash
cp .env.example .env
```

**Step 2: Edit `.env` and set the model path**

```ini
# .env — ALPR Configuration

# Path to the YOLOv8 license plate detection model weights
# Can be absolute or relative to the project root
ALPR_MODEL_PATH=models/plate_detector.pt

# Confidence threshold for the detector (0.0 to 1.0)
ALPR_CONFIDENCE_THRESHOLD=0.5

# IoU threshold for Non-Maximum Suppression
ALPR_IOU_THRESHOLD=0.45
```

**Step 3: Verify the configuration loads correctly**

```bash
python -c "from src.detector import PlateDetector; d = PlateDetector(); print('Model loaded OK')"
```

---

## 5. License Notes

### Ultralytics YOLOv8 Framework

The YOLOv8 framework is developed by Ultralytics (https://ultralytics.com/) and is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. Key implications:

- Free to use for educational and research purposes (this project).
- Modifications may be made, provided source code is disclosed.
- Commercial deployment requires a separate Ultralytics commercial license.
- Any software that uses YOLOv8 and is distributed must also be open-source under AGPL-3.0 (or a compatible license).

**Reference:** https://github.com/ultralytics/ultralytics/blob/main/LICENSE

### Model Weights (Fine-Tuned)

The license of the fine-tuned model **weights** (`.pt` file) may differ from the framework license. Always check the specific model card:

| Model Source | Where to Check License |
|---|---|
| `keremberke/license-plate-detection` | https://huggingface.co/keremberke/license-plate-detection — see Model Card |
| Roboflow Universe models | License displayed on the individual model/dataset page |
| Custom-trained weights | Your own choice; recommended: CC BY 4.0 or MIT for academic use |

> WARNING: This project is for **academic/college purposes only**. Do not deploy the system commercially without verifying and complying with the licenses of all model weights used.

---

## 6. How to Train Your Own Model

Training a custom model on Indian license plates is recommended for best accuracy, since most publicly available pre-trained models are optimized for Western-style plates (USA/EU formats). Follow these steps:

### Step 1 — Collect and Annotate a Dataset

- Collect images of Indian number plates (various states, angles, lighting conditions).
- Annotate with bounding boxes using LabelImg (https://github.com/HumanSignal/labelImg) or Roboflow (https://roboflow.com).
- Export annotations in **YOLOv8 format** (creates `images/` and `labels/` directories).
- Aim for a minimum of **500-1000 annotated images** for acceptable performance.

### Step 2 — Prepare the Dataset YAML

Create a file `data/plate_dataset.yaml`:

```yaml
path: data/plate_dataset    # Root directory of the dataset
train: images/train         # Training images subdirectory
val: images/val             # Validation images subdirectory
test: images/test           # Test images subdirectory (optional)

nc: 1                       # Number of classes
names: ['license-plate']    # Class names
```

### Step 3 — Train with Ultralytics CLI

```bash
# Install ultralytics if not already installed
pip install ultralytics

# Train YOLOv8 nano model (fastest; replace 'n' with 's', 'm', 'l' for larger models)
yolo detect train \
    model=yolov8n.pt \
    data=data/plate_dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=plate_detector_v1 \
    project=runs/train
```

### Step 4 — Evaluate and Export

```bash
# Evaluate on the validation set
yolo detect val model=runs/train/plate_detector_v1/weights/best.pt data=data/plate_dataset.yaml

# Copy the best weights to the models directory
cp runs/train/plate_detector_v1/weights/best.pt models/plate_detector.pt
```

### Step 5 — Update Configuration

Update your `.env` file to point to the newly trained weights:

```ini
ALPR_MODEL_PATH=models/plate_detector.pt
```

> TIP: Use data augmentation options in YOLOv8 (`hsv_h`, `hsv_s`, `flipud`, `mosaic`) to improve robustness against varying lighting and angles common in Indian road conditions. Refer to the Ultralytics training documentation at https://docs.ultralytics.com/modes/train/

---

## 7. Directory Structure (After Setup)

```
FinalProject/
+-- models/
|   +-- README.md              <- This file
|   +-- plate_detector.pt      <- Downloaded/trained weights (NOT in Git)
+-- .env                       <- Local config (NOT in Git)
+-- .env.example               <- Template config (in Git)
+-- .gitignore                 <- Ensures *.pt and .env are excluded
```

Make sure your `.gitignore` contains:

```gitignore
# Model weights (large binary files - download separately)
models/*.pt
models/*.onnx

# Environment configuration (contains API keys)
.env
```

---

## 8. Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| `FileNotFoundError: models/plate_detector.pt` | Weights not downloaded | Follow Section 2 to download |
| Model loads but detects nothing | Wrong model (COCO instead of plate-specific) | Re-download the correct fine-tuned model |
| `RuntimeError: CUDA out of memory` | GPU memory too small | Set `device='cpu'` in `src/detector.py` |
| Very low confidence scores (< 0.3) | Image quality poor or wrong model | Try preprocessing image; verify model source |
| `huggingface_hub` import error | Library not installed | Run `pip install huggingface_hub` |

---

*Document maintained by Abhayvir Singh. Last updated: 22 October 2026.*
