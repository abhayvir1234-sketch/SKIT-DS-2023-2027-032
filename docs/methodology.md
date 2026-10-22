# Methodology — Vehicle Number Plate Recognition System

---

## 1. Image Acquisition and Validation

The user uploads a JPEG or PNG image through the Streamlit file uploader.
Before any processing begins, the system validates:

- **Extension check**: Only `.jpg`, `.jpeg`, `.png` are accepted.
- **Size check**: Files over 10 MB are rejected to prevent memory issues.
- **Decode check**: OpenCV attempts to decode the bytes; corrupted files are
  caught here rather than mid-pipeline.

This validation layer prevents bad input from reaching model inference, which
is the most computationally expensive step.

---

## 2. Preprocessing Strategy

### Why preprocess before detection?

Raw images from cameras can be very large (4K+). Passing a 4000×3000 image
directly to YOLO would consume large amounts of GPU/CPU memory and slow inference.

**Step applied**: `resize_keeping_aspect(max_side=1280)` — the image is scaled
down so its longest side is at most 1280 px, with aspect ratio preserved. This
matches the typical YOLO inference range without losing plate details.

No colour normalisation is applied at this stage because YOLO models expect
natural-colour BGR frames.

### Why multiple preprocessing variants for OCR?

Different plates suffer from different degradations:
- Dark environments → CLAHE (local contrast enhancement) helps.
- Very bright environments → Otsu binarisation separates foreground from background.
- Camera motion blur → Sharpening partially recovers edges.
- Uneven lighting → Adaptive thresholding handles local brightness changes.

Rather than guessing which enhancement is best, the system generates 5 variants
and lets the OCR engine score each. The variant with the highest average word
confidence wins.

---

## 3. YOLOv8 Detection

### What is YOLOv8?

YOLOv8 (You Only Look Once, version 8) is a single-stage object detector
by Ultralytics (2023). Unlike two-stage detectors (e.g., Faster R-CNN) that
first propose regions then classify them, YOLO predicts bounding boxes and
class probabilities in a single forward pass.

**Key architecture characteristics:**
- **Backbone**: CSPDarknet (cross-stage partial network) for efficient feature extraction.
- **Neck**: PANet (path aggregation network) for multi-scale feature fusion.
- **Head**: Anchor-free detection head (YOLOv8 removes anchor boxes, predicting
  offsets directly from grid cell centres).
- **Loss**: Distribution focal loss (DFL) for box regression + binary cross-entropy
  for classification.

### Why YOLOv8 specifically?

- Anchor-free design is more robust to unusual aspect ratios (plates are wide rectangles).
- Ultralytics Python API is clean and well-documented.
- Supports CPU inference without any additional setup.
- Pre-trained checkpoints are readily fine-tunable on custom plate datasets.

### Confidence and filtering

Every detection includes a **confidence score** (0–1) indicating the model's
certainty that the bounding box contains a plate. Detections below the
configured threshold (`ALPR_DETECT_CONF`, default 0.25) are discarded.
The threshold is exposed as a slider in the UI so the user can tune it.

### Multiple detections

When multiple plates are detected, they are sorted by confidence (descending)
then left-to-right. The highest-confidence plate is used for the primary OCR
result; all detections are annotated on the output image.

---

## 4. Plate Crop Extraction

After detection, `extract_plate_crop()` slices the plate region from the
**original** (pre-detection-resize) image to preserve full resolution.

**Padding**: 8 px of context is added on all sides. This improves OCR because
characters near the plate border are sometimes clipped by tight bounding boxes.

**Boundary clipping**: Padded coordinates are clamped to `[0, image_width]` and
`[0, image_height]` so no out-of-bounds memory access occurs.

**Validation**: Degenerate crops (zero area, extremely small) are rejected before
being passed to OCR.

---

## 5. OCR Strategy

### Why EasyOCR over Tesseract?

| Feature | EasyOCR | Tesseract |
|---------|---------|-----------|
| Installation | `pip install easyocr` (simple) | Requires OS-level binary install |
| Deep learning | CRAFT + CRNN neural networks | Mostly classical + LSTM |
| Number plate accuracy | Generally higher | Requires significant tuning |
| Confidence output | Per-word float | Less reliable float |
| Windows support | Excellent | Requires GTesseract wrapper |

### How EasyOCR works

EasyOCR is a two-stage neural OCR system:
1. **CRAFT** (Character Region Awareness For Text detection): A fully
   convolutional network that finds character bounding boxes.
2. **CRNN** (Convolutional Recurrent Neural Network): Reads each detected text
   region left-to-right, outputting a sequence of characters.

### Variant selection

The system runs OCR on all 5 preprocessing variants. For each variant,
EasyOCR returns a list of `(bbox, text, confidence)` tuples. Words below the
minimum confidence threshold (`ALPR_OCR_MIN_CONF`, default 0.1) are discarded.
The variant whose accepted words have the highest **average confidence** is
selected as the final result.

---

## 6. Text Normalization and Validation

### Normalization (always applied)

1. Unicode NFC normalization.
2. Convert to uppercase.
3. Remove characters that are not alphanumeric, space, or hyphen.
4. Collapse consecutive spaces/hyphens to a single space.
5. Strip leading/trailing whitespace.

### What is deliberately NOT done

- **O → 0 / I → 1 substitution**: These are NOT applied automatically. While
  OCR engines sometimes confuse O with 0, blindly substituting can corrupt
  correct characters. No substitution is applied without explicit configuration.
- **Inventing missing characters**: If OCR cannot read a character, it is not
  guessed.

### Validation (configurable)

In **lenient mode** (default): any non-empty alphanumeric string of 4–16
characters is accepted.

In **strict mode**: the text must match one of the documented Indian plate formats:
- Standard: `XX 00 XX 0000` (e.g., MH 12 AB 1234)
- Bharat series: `00 BH 0000 XX` (e.g., 22 BH 1234 AA)

Validation failure does not discard the OCR result — it generates an advisory
note shown to the user.

---

## 7. Result Annotation

`annotate_image()` draws on a **copy** of the original image (never modifying
the original array). For each detected plate:
- A green bounding box is drawn.
- A label showing `{recognized_text} [{confidence}]` is drawn above the box
  with a filled background rectangle for readability.

When no plate is detected, `annotate_no_detection()` overlays a "No number
plate detected" message with a semi-transparent background.

---

## 8. Performance Considerations

- **Model caching**: `@st.cache_resource` ensures the YOLO model and EasyOCR
  reader are loaded exactly once per Streamlit session.
- **CPU support**: All operations run on CPU by default. GPU is optional.
- **Large image handling**: Images are downscaled before YOLO inference but
  the crop is extracted from the full-resolution original.
- **No duplicate OCR passes**: OCR is run once per crop with 5 variants; the
  results are compared in Python without additional model calls.
