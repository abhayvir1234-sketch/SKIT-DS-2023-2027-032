# Viva Questions & Answers — Vehicle Number Plate Recognition System

**Course:** Computer Vision & Machine Learning  
**Team:** Abhayvir Singh, Himank Devnani, Karam Singh Bana  
**Date:** 22 October 2026

---

## Section 1: ALPR Fundamentals

**Q1. What is ALPR and what are its real-world applications?**

ALPR (Automatic License Plate Recognition) is a computer vision technique that
automatically detects vehicle licence plates in images or video and extracts the
alphanumeric registration text. Applications include: traffic law enforcement
(speed cameras, red-light cameras), toll collection (electronic toll plazas),
parking management (automated entry/exit), border control, stolen vehicle
tracking, and fleet management.

---

**Q2. What are the two main stages of an ALPR pipeline?**

1. **Detection**: Localising the licence plate region within the image and
   producing a bounding box (usually done with a CNN-based object detector).
2. **Recognition**: Reading the alphanumeric characters from the cropped plate
   region (usually done with OCR — optical character recognition).

---

**Q3. What challenges make ALPR difficult in real-world conditions?**

- Varying illumination (bright sun, night, headlights).
- Motion blur (moving vehicles or camera shake).
- Partial occlusion (dirt, stickers, damage, other vehicles).
- Diverse plate formats (different countries, states, ages).
- Extreme viewing angles and perspective distortion.
- Weather effects (rain, fog, snow on plates).
- Low-resolution images (distant cameras, cheap sensors).

---

## Section 2: YOLOv8 and Object Detection

**Q4. What does YOLO stand for and what is its core idea?**

YOLO stands for "You Only Look Once." The core idea is to frame object
detection as a single regression problem: a single convolutional neural
network simultaneously predicts bounding boxes and class probabilities from
the entire image in one forward pass. This makes it much faster than two-stage
detectors that first generate candidate regions then classify each one.

---

**Q5. What is the architecture of YOLOv8?**

YOLOv8 consists of three parts:
- **Backbone** (CSPDarknet): Extracts features at multiple scales using
  cross-stage partial connections to improve gradient flow and reduce computation.
- **Neck** (PANet — Path Aggregation Network): Combines features from different
  backbone stages (multi-scale feature pyramid) so the detector can find both
  large and small objects.
- **Head**: An anchor-free detection head that predicts bounding box offsets
  directly from grid cell centres without pre-defined anchor box priors.

---

**Q6. What is "anchor-free" detection and why does YOLOv8 use it?**

Earlier YOLO versions (v1–v5) used "anchor boxes" — pre-defined bounding box
shapes (aspect ratios, sizes) that the model predicted offsets from. YOLOv8's
anchor-free head predicts the four bounding box coordinates (x, y, w, h)
directly. Benefits: no need to manually design anchor shapes for your specific
dataset, better generalisation to unusual object shapes (such as wide licence
plates), simpler training procedure.

---

**Q7. What is the difference between YOLO and Faster R-CNN?**

| Aspect | YOLO | Faster R-CNN |
|--------|------|-------------|
| Stages | 1 (single pass) | 2 (RPN proposals + ROI classification) |
| Speed | Fast (real-time capable) | Slower |
| Accuracy (small objects) | Can miss very small objects | Generally better |
| Complexity | Simpler architecture | More complex |
| Use case | Real-time detection | High-accuracy offline detection |

---

**Q8. What is a confidence score in object detection?**

The confidence score is a float between 0 and 1 output by the detection model
for each bounding box. It represents the model's estimated probability that
(a) an object is present in that box, and (b) the box is well-localised around
that object. In YOLOv8, the final confidence is the product of the objectness
score and the class probability. Boxes with confidence below a threshold are
discarded.

---

**Q9. What is Non-Maximum Suppression (NMS)?**

NMS is a post-processing step that removes duplicate bounding boxes for the
same object. When the detector outputs many overlapping boxes, NMS:
1. Sorts boxes by confidence score.
2. Keeps the highest-confidence box.
3. Removes all other boxes that overlap with it by more than an IoU threshold.
4. Repeats until no overlapping boxes remain.

---

**Q10. What is Intersection over Union (IoU)?**

IoU measures the overlap between two bounding boxes:

`IoU = Area of Intersection / Area of Union`

A value of 1.0 means perfect overlap; 0.0 means no overlap. Used in NMS to
decide if two boxes are detecting the same object, and in evaluation metrics
to decide if a detection is correct.

---

## Section 3: Image Preprocessing

**Q11. Why do we preprocess the plate crop before OCR?**

Raw plate crops from detection may suffer from low contrast, noise, shadows,
or insufficient resolution. Preprocessing improves the signal-to-noise ratio
for the OCR engine. Different conditions call for different enhancements, so
we generate multiple variants and let the OCR confidence score select the best.

---

**Q12. What is CLAHE and how does it help?**

CLAHE (Contrast-Limited Adaptive Histogram Equalisation) is a local contrast
enhancement technique. Unlike global histogram equalisation, CLAHE divides the
image into small tiles and equalises each tile's histogram independently, then
interpolates between tiles. The "contrast-limited" part clips the histogram to
avoid over-amplifying noise. It helps plates that are unevenly lit — the shadowed
half is enhanced without over-brightening the well-lit half.

---

**Q13. What is Otsu's thresholding?**

Otsu's method automatically finds the optimal global threshold to binarise a
grayscale image into black and white. It minimises the intra-class variance
between the two resulting pixel groups (foreground/background). No manual
threshold parameter is needed. Works well when the histogram has a clear
bimodal distribution, as often seen in well-lit plates.

---

**Q14. What is adaptive thresholding and when is it better than Otsu?**

Adaptive thresholding computes a different threshold for each small neighbourhood
of pixels (based on local mean or Gaussian-weighted mean). This is better than
Otsu when the plate has uneven illumination — for example, one end is in shadow.
A global Otsu threshold would either fail on the dark side or over-binarise the
bright side. Adaptive thresholding handles this because each pixel is compared
to its local neighbourhood.

---

**Q15. Why should we avoid over-preprocessing?**

Over-preprocessing (applying every enhancement blindly) can:
- Destroy fine character strokes that are needed for OCR.
- Introduce artifacts from aggressive sharpening.
- Remove useful colour information that EasyOCR can leverage.
- Slow down the pipeline unnecessarily.

We generate targeted variants and use the OCR confidence to select the best
rather than chaining all operations.

---

## Section 4: OCR

**Q16. What is EasyOCR and how does it work?**

EasyOCR is a Python library (JaidedAI, 2020) that uses two neural networks:
1. **CRAFT** (Character Region Awareness For Text): A fully convolutional network
   that detects individual character regions and groups them into text boxes.
2. **CRNN** (Convolutional Recurrent Neural Network): Reads each text box
   left-to-right. A CNN extracts image features; a bidirectional LSTM captures
   sequence context; CTC (Connectionist Temporal Classification) decodes the
   sequence into characters.

---

**Q17. What is CTC and why is it used in OCR?**

CTC (Connectionist Temporal Classification) is a loss function and decoding
algorithm for sequence-to-sequence problems where the alignment between input
and output is unknown. In CRNN-based OCR, the input is a sequence of image
column features and the output is a character sequence. CTC allows training
without pre-segmenting characters and handles variable-length inputs/outputs.

---

**Q18. Why did we choose EasyOCR over Tesseract for this project?**

EasyOCR uses deep learning for both text detection and recognition, which
generally gives better accuracy on natural-scene text (including licence plates)
compared to Tesseract's classical approach. EasyOCR also installs cleanly via
pip on Windows without requiring OS-level binary dependencies (Tesseract requires
separate installation and PATH configuration). EasyOCR also provides a per-word
confidence score natively.

---

**Q19. What does text normalization do in this project?**

The raw EasyOCR output may contain:
- Lowercase characters (plates are uppercase).
- Punctuation or OCR artifacts.
- Extra spaces between characters.

Normalization: uppercases all characters, removes non-alphanumeric characters
(except spaces and hyphens), collapses multiple spaces to one, and strips
leading/trailing whitespace. This produces a clean candidate string.

---

**Q20. Why don't you automatically replace O with 0 and I with 1?**

This is a common question. The substitution seems logical for plates (which
typically use 0 not O, and 1 not I), but:
1. It can corrupt **correct** characters. "MHO2" could be a valid partial
   read where 'O' is truly an 'O' in the state code.
2. Modern neural OCR (EasyOCR with CRNN) is trained on license plates and
   already learns context-appropriate predictions. Pre-applying character
   substitutions before OCR is not necessary.
3. Post-OCR substitution without configuration risks introducing hard-to-debug
   errors. The project design applies corrections only when justified by an
   explicit documented rule.

---

## Section 5: OpenCV

**Q21. What is OpenCV and what does it do in this project?**

OpenCV (Open Source Computer Vision Library) is a widely used C++ library with
a Python binding. In this project it is used for: decoding image bytes
(`cv2.imdecode`), colour space conversion BGR↔RGB (`cv2.cvtColor`), resizing
(`cv2.resize`), CLAHE (`cv2.createCLAHE`), Gaussian blur, Otsu thresholding,
adaptive thresholding, sharpening (`cv2.addWeighted`), and drawing bounding
boxes and text on images.

---

**Q22. Why does OpenCV use BGR instead of RGB?**

OpenCV reads images in BGR (Blue-Green-Red) channel order by historical
convention — it was designed to be compatible with older Windows bitmap formats
that store channels in BGR order. Most other libraries (Pillow, Matplotlib,
EasyOCR) expect RGB. This project explicitly converts between them where needed
using `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

---

## Section 6: Confidence and Evaluation

**Q23. What is the difference between detector confidence and OCR confidence in this project?**

- **Detector confidence**: Reported by YOLOv8 for each detected bounding box.
  Represents the model's certainty that a licence plate exists in that region.
- **OCR confidence**: Reported by EasyOCR as an average of per-word confidence
  scores. Represents how certain the text recognition model is about the
  characters it read.

These are separate scores from different models and must not be combined into a
single "accuracy percentage."

---

**Q24. What is mAP (mean Average Precision) and how would you evaluate this detector?**

mAP is the standard object detection evaluation metric. For each class:
1. Compute precision and recall at different confidence thresholds.
2. Plot the precision-recall curve.
3. Compute Average Precision (AP) as the area under the curve.

mAP is the mean of APs across all classes. Common variants: mAP@50 (IoU ≥ 0.5
for a detection to be "correct") and mAP@50:95 (averaged over IoU 0.5 to 0.95).

To evaluate this project's detector: `yolo detect val model=models/plate_detector.pt data=data.yaml`

---

**Q25. What are false positives and false negatives in the context of this system?**

- **False positive (FP)**: The detector reports a plate where there is none
  (e.g., a bumper sticker is detected as a plate). Reduces precision.
- **False negative (FN)**: A real plate exists but the detector misses it
  (e.g., plate is too small or occluded). Reduces recall.

Both have real-world costs. A false positive in a toll system charges the wrong
vehicle. A false negative lets an unregistered vehicle pass undetected.

---

## Section 7: Streamlit

**Q26. What is Streamlit and why was it chosen for this project?**

Streamlit is an open-source Python framework that turns Python scripts into
interactive web applications without requiring HTML, CSS, or JavaScript
knowledge. It was chosen because: (a) the entire team is proficient in Python;
(b) it allows rapid prototyping of data-science-style interfaces; (c) it
provides built-in widgets (file uploader, sliders, download buttons, tabs)
that match the project's UI requirements; (d) it has an active community and
good documentation.

---

**Q27. What is `@st.cache_resource` and why is it important for this project?**

`@st.cache_resource` is a Streamlit decorator that caches the result of a
function across all user sessions and reruns. When a user interacts with a
Streamlit widget (e.g., moves a slider), the entire script re-executes. Without
caching, the YOLOv8 model and EasyOCR reader would reload from disk on every
interaction — which takes 5–30 seconds. `@st.cache_resource` ensures models are
loaded exactly once per server lifecycle, making the UI responsive.

---

## Section 8: Datasets and Training

**Q28. What dataset was used to train the detection model?**

This project uses a pre-trained model (e.g., `keremberke/license-plate-detection`
from Hugging Face) trained on publicly available licence plate detection datasets.
The training dataset details are on the model's Hugging Face card. For a
custom Indian plate model, datasets can be sourced from Roboflow Universe or
created by annotating collected images using LabelImg.

---

**Q29. What dataset format does YOLOv8 training expect?**

YOLOv8 expects:
- `images/train/*.jpg` (or png) — training images.
- `labels/train/*.txt` — one .txt per image, with one line per bounding box:
  `class_id cx cy w h` (all values normalized 0–1).
- A `data.yaml` file specifying `path`, `train`, `val`, `nc` (number of classes),
  and `names` (class name list).

---

**Q30. What is data augmentation and why is it used in YOLO training?**

Data augmentation synthetically increases training data diversity by applying
random transformations (flips, rotations, brightness changes, mosaic, cutout)
to training images. This improves model generalisation because the model learns
to detect plates in a wider variety of visual conditions than the training set
alone provides. YOLOv8 applies augmentation automatically during training.

---

## Section 9: Real-World Limitations and Improvements

**Q31. What are the main limitations of this ALPR system?**

1. Requires a trained model weight file (not included in code).
2. Static images only — no video stream support.
3. OCR accuracy degrades on heavily blurred or low-resolution plates.
4. Validation is tailored to Indian formats — international plates may show warnings.
5. Only the highest-confidence plate is shown in detail.
6. CPU inference is slow (5–10 s per image on a laptop).

---

**Q32. How would you improve OCR accuracy for Indian number plates specifically?**

1. Fine-tune EasyOCR on a dataset of annotated Indian plate crops.
2. Add a post-OCR correction dictionary of known Indian state codes.
3. Apply controlled O→0 and I→1 substitution ONLY in positions where a digit
   is expected according to the Indian plate format (e.g., positions 3–4).
4. Use a higher plate crop resolution (zoom in more before OCR).
5. Train a plate-super-resolution model to upscale low-resolution crops.

---

**Q33. How could the system be extended to process video?**

Replace the file uploader with `cv2.VideoCapture()` for webcam or video file
input. Process each frame through the pipeline and display annotated frames
using `st.image()` in a loop. For real-time performance, use a lighter YOLOv8
variant (yolov8n) and possibly skip alternate frames. Streamlit's
`st.empty()` placeholder can be updated each iteration without refreshing the
entire page.

---

**Q34. What ethical considerations apply to ALPR systems?**

1. **Privacy**: Recording vehicle movements creates detailed surveillance data.
   Collection must comply with local data protection laws (PDPB in India, GDPR in EU).
2. **Accuracy and bias**: ALPR systems trained on one region may fail on plates
   from another. Incorrect reads can wrongly target innocent people.
3. **Purpose limitation**: Data collected for one purpose (tolls) must not be
   used for another (tracking individuals) without consent.
4. **No ownership assumption**: OCR text is not proof of vehicle ownership.
   This system explicitly does not connect to registration databases.

---

**Q35. What metrics would you report in a final project evaluation?**

- **Plate detection**: mAP@50 and mAP@50:95 on a held-out test set.
- **Plate localisation**: Mean IoU between predicted and ground-truth bounding boxes.
- **OCR accuracy**: Character Error Rate (CER) = Levenshtein distance / true text length.
- **End-to-end accuracy**: Exact match rate (entire plate string correctly read).
- **Speed**: Average inference time per image (seconds) on the target hardware.
- All metrics reported with the dataset size and split used.
