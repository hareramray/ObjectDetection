"""Author the course orientation, dataset workshop and final practical project."""
from notebook_factory import md, code, bootstrap, write_notebook

write_notebook("00_Start_Here_and_Environment.ipynb", [
md('''
# 00 · Start here: your object detection laboratory

**Level:** first lesson · **Class time:** 25–40 minutes · **Hardware:** CPU is enough.

By the end, you will run notebook cells, explain what a detector returns, check your
Python environment, and display your first labeled image. Basic Python variables,
lists and functions are helpful; no deep learning knowledge is assumed.

**Presenter opening:** “Which objects are here, and exactly where are they?”
Classification answers *what is in an image*. Detection returns a class and a box
for **each instance**. Segmentation additionally labels pixels. Tracking associates
detections across time; a track ID is not a person's identity.
'''),
md('''
## Step 1 — Start Jupyter on Windows

From PowerShell in this project folder, follow `README.md`. Select the kernel
**Python (Object Detection)**. If the environment was already prepared, run
`./.venv/Scripts/python.exe -m jupyterlab notebooks`.

- **Shift+Enter:** run the selected cell and move on.
- **Markdown cells:** explanation; **code cells:** Python to execute.
- Run from the top. Restart Kernel → Run All checks that no hidden variables are needed.
- Interrupt Kernel stops a long computation; restarting clears variables, not saved files.
- Change one setting at a time, rerun that cell, then rerun cells that depend on it.

Cells marked as optional require their switch to be enabled. No notebook opens your
webcam automatically. First use of pretrained models needs internet; cache them before class.
'''), bootstrap(),
md('''
## Step 2 — Read the environment, not just the installation command

Jupyter's kernel may differ from the Python in a terminal. `sys.executable` tells you
which interpreter is actually running this code. The course uses a project environment;
the delivered environment reuses your already installed PyTorch/CUDA packages.
'''),
code('''
import platform
from importlib.metadata import version, PackageNotFoundError
print("Python:", platform.python_version())
print("Interpreter:", sys.executable)
for package in ["numpy", "matplotlib", "pillow", "torch", "torchvision", "ultralytics", "jupyterlab"]:
    try:
        print(f"{package:15s} {version(package)}")
    except PackageNotFoundError:
        print(f"{package:15s} MISSING — install requirements.txt in this kernel's environment")
'''),
code('''
import torch
from course_utils import choose_device
print("CUDA available:", torch.cuda.is_available())
print("Recommended training device:", choose_device())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CPU mode works. Start with the small demo settings.")
'''),
md('''
## Step 3 — A detection is structured data

`[x1, y1, x2, y2]` means left, top, right, bottom in image coordinates.
The origin is at the top-left, x increases rightward, and y increases downward.
A confidence score is a model score used to rank/filter predictions; it is not
automatically a calibrated probability that the prediction is correct.
'''),
code('''
detection = {"class_name": "rectangle", "box_xyxy": [20, 30, 100, 90], "score": 0.87}
x1, y1, x2, y2 = detection["box_xyxy"]
print(detection)
print("Width:", x2-x1, "Height:", y2-y1, "Area:", (x2-x1)*(y2-y1))
'''),
md('''
## Step 4 — Make a labeled scene without downloading anything

The generator draws original rectangles, circles and triangles and records their boxes.
These boxes are **ground truth**, not predictions. Later we will ask a model to predict
them. Shape colors vary independently of class, so color alone cannot identify the class.
'''),
code('''
import matplotlib.pyplot as plt
from course_utils import make_scene, CLASS_NAMES
image, boxes, labels = make_scene(seed=7)
print("Image size (width, height):", image.size)
print("Boxes:\\n", boxes)
show_boxes(image, boxes, [CLASS_NAMES[int(c)] for c in labels], title="Ground truth, not model output")
plt.show()
'''),
md('''
## Step 5 — Follow the learning path

| Notebook | Practical outcome |
|---|---|
| 01 | Inspect RGB pixels; draw, crop, resize and convert boxes |
| 02 | Build a classical detector and see where fixed rules fail |
| 03 | Compute IoU and remove duplicate predictions with NMS |
| 04 | Detect everyday objects using pretrained YOLO |
| 05 | Process video, maintain track IDs and count line crossings |
| 06 | Generate, inspect, annotate and audit a custom dataset |
| 07 | Train and save a custom YOLO model |
| 08 | Measure precision, recall, AP and mAP without leakage |
| 09 | Transform boxes correctly and investigate errors |
| 10 | Train a small single-object CNN from scratch |
| 11 | Fine-tune Faster R-CNN with a manual PyTorch training loop |
| 12 | Export to ONNX and measure equivalent inference paths |
| 13 | Deliver a reproducible detection project and model report |

The later lessons practice advanced engineering skills. Completing notebooks alone
does not establish expertise; repeat the capstone on real, independently collected data.
'''),
md('''
## Step 6 — Distinguish demonstration from evidence

1. **Training:** update weights from labeled examples.
2. **Validation:** choose settings and inspect errors.
3. **Test:** measure the frozen choice once on untouched examples.

Our generated shapes make a small, repeatable teaching problem. Performance on them
does not predict performance on real traffic, products, or people. A quick five-epoch
run demonstrates the workflow; a longer run and a representative dataset are needed
before claiming useful accuracy.

The default YOLO model is **YOLO11n**, an established small model chosen to connect
the NMS lesson to actual inference. The course does not claim it is the newest model.
'''),
md('''
## Your first experiment

Change the seed to 21 and request exactly three shapes. Before running, predict which
parts of the returned data will change. Explain why changing a seed is different from
training a model. **Presenter checkpoint:** ask a learner to point out width versus height.
'''),
code('''
# Exercise: edit these values and rerun.
DEMO_SEED = 21
NUMBER_OF_OBJECTS = 3
example, example_boxes, example_labels = make_scene(
    seed=DEMO_SEED, min_objects=NUMBER_OF_OBJECTS, max_objects=NUMBER_OF_OBJECTS)
show_boxes(example, example_boxes, [CLASS_NAMES[int(c)] for c in example_labels])
plt.show()
'''),
md('''
**Solution/checkpoint:** a new seed changes the generated pixels, positions and possibly
classes. The length of both arrays remains three. No model weights were updated.

**If something fails:** wrong import → check the selected kernel; missing module →
install the requirements in that environment; out of GPU memory → use `device="cpu"`
or a smaller batch; stalled download → prepare/cache assets before the presentation.

**Next:** [01 — Image basics](01_Image_Basics_and_Bounding_Boxes.ipynb).
Primary references: [JupyterLab notebooks](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html),
[YOLO11 model documentation](https://docs.ultralytics.com/models/yolo11/).
''')])

write_notebook("06_Custom_Dataset_and_Annotation.ipynb", [
md('''
# 06 · Build and audit a custom detection dataset

**Level:** intermediate · **Class time:** 45–60 minutes · **Prerequisites:** 01 and 03.
**Resources:** offline, CPU, no paid service or annotation account.

You will create 144 labeled images, understand YOLO annotation files, inspect labels,
check class balance and duplicates, and practice a manual annotation. **Presenter
question:** “Can a model learn an object that the annotator forgot to label?”
'''), bootstrap(),
md('''
## Step 1 — Generate train, validation and test partitions

Our three classes are `rectangle=0`, `circle=1`, `triangle=2`. The generator uses
separate seeds for each split: 96 training, 24 validation and 24 test images.
Every twelfth image is intentionally empty. This helps distinguish true negatives
from missing annotations. Rerunning this cell reuses existing files.
'''),
code('''
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from course_utils import (ensure_synthetic_dataset, audit_dataset, read_yolo_labels,
                          xyxy_to_yolo, CLASS_NAMES)
DATA_YAML = ensure_synthetic_dataset()
DATA_ROOT = DATA_YAML.parent
print(DATA_YAML.read_text())
print((DATA_ROOT / "manifest.json").read_text())
'''),
md('''
## Step 2 — Connect the files to the task

```
synthetic_shapes/
  data.yaml
  images/train/train_0001.png
  labels/train/train_0001.txt
  images/val/...
  labels/val/...
  images/test/...
  labels/test/...
```

For every object, a row holds `class_id center_x center_y width height`.
Coordinates are divided by image width or height and lie between zero and one.
Class IDs start at zero. The course writes explicit empty text files for negatives;
Ultralytics also permits missing label files for images with no objects.
'''),
code('''
image_path = DATA_ROOT / "images/train/train_0001.png"
label_path = DATA_ROOT / "labels/train/train_0001.txt"
image = Image.open(image_path).convert("RGB")
print("Raw label file:\\n", label_path.read_text())
boxes, labels = read_yolo_labels(label_path, *image.size)
show_boxes(image, boxes, [CLASS_NAMES[int(c)] for c in labels], title=image_path.name)
plt.show()
'''),
md('''
## Step 3 — Work through one annotation by hand

For a 256×256 image and `[32, 64, 128, 160]`, the center is `(80,112)`;
the width and height are both 96. Normalize x quantities by 256 and y quantities by
256. For a non-square image, those denominators differ.
'''),
code('''
manual_box = np.array([[32., 64., 128., 160.]])
normalized = xyxy_to_yolo(manual_box, width=256, height=256)
print("class cx cy width height:", 0, *normalized[0])
assert np.allclose(normalized, [[0.3125, 0.4375, 0.375, 0.375]])
'''),
md('''
## Step 4 — See a grid before trusting a statistic

Look for boxes that are too loose, shifted coordinates, missing objects, incorrect
class IDs and truncated objects. The first image is a deliberately negative example.
We inspect training and validation images; keep the test split for the final project.
'''),
code('''
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, path in zip(axes.flat, sorted((DATA_ROOT / "images/train").glob("*.png"))[:8]):
    with Image.open(path) as im:
        b, c = read_yolo_labels(DATA_ROOT / "labels/train" / (path.stem + ".txt"), *im.size)
        show_boxes(im, b, [CLASS_NAMES[int(k)] for k in c], ax=ax, title=path.stem)
plt.tight_layout()
plt.show()
'''),
md('''
## Step 5 — Audit the actual files

The audit checks image readability, matching label files, finite normalized values,
positive box sizes, valid class IDs, in-bounds boxes and exact image duplicates across
splits. Hash checks cannot detect different frames of the same scene or near-duplicates;
split by recording session, object, location or person when collecting real data.
'''),
code('''
audit = pd.DataFrame(audit_dataset(DATA_YAML))
display(audit)
audit.set_index("split")[CLASS_NAMES].plot.bar(figsize=(8, 4), title="Object counts, not image counts")
plt.ylabel("Annotated instances")
plt.tight_layout()
plt.show()
'''),
md('''
## Step 6 — Split by group before augmentation

Here twelve imaginary recording sessions each contribute five images. Shuffle the
session IDs, then assign all images of a session to one split. Do this **before**
augmentation. A rotated copy of a training image in validation is data leakage.
For real data, ensure every important class and condition has enough independent groups.
'''),
code('''
records = pd.DataFrame([{"file": f"session{s:02d}_frame{f:03d}.jpg", "group": s}
                        for s in range(12) for f in range(5)])
rng = np.random.default_rng(42)
groups = rng.permutation(records["group"].unique())
assignment = {int(g): "train" if i < 8 else "val" if i < 10 else "test"
              for i, g in enumerate(groups)}
records["split"] = records["group"].map(assignment)
assert records.groupby("group")["split"].nunique().max() == 1
display(records.groupby("split").agg(images=("file", "size"), independent_groups=("group", "nunique")))
'''),
md('''
## Step 7 — Annotate a sample yourself

Choose a training image with one or more shapes. Display it without boxes and read its
pixel coordinates from the axes. Enter one box and a class ID for a visible object.
This exercise saves a separate annotation under `artifacts/annotation_practice/`;
it does not replace the generated training labels. A complete image annotation must
include **all** in-scope visible objects, not just the easiest one.
'''),
code('''
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Read pixel coordinates from the axes")
plt.grid(alpha=0.3)
plt.show()
print("Image dimensions:", image.size)
'''),
code('''
# Start with one known box; move its edges to see how annotation quality changes.
practice_box = boxes[:1].copy()
practice_class = int(labels[0])
practice_dir = ROOT / "artifacts" / "annotation_practice"
practice_dir.mkdir(parents=True, exist_ok=True)
practice_path = practice_dir / "one_object.txt"
row = xyxy_to_yolo(practice_box, *image.size)[0]
practice_path.write_text(f"{practice_class} " + " ".join(f"{v:.8f}" for v in row) + "\\n")
roundtrip_boxes, roundtrip_classes = read_yolo_labels(practice_path, *image.size)
show_boxes(image, roundtrip_boxes, [CLASS_NAMES[int(c)] for c in roundtrip_classes], title="Your annotation")
plt.show()
assert np.allclose(roundtrip_boxes, practice_box, atol=1e-4)
'''),
md('''
## Exercise and solution

**Exercise:** Explain what is wrong with `1 0.4 0.4 -0.2 0.3` and
`7 0.4 0.4 0.2 0.3`. How should an empty image be represented? Why does using a
random split of neighboring video frames inflate your confidence in a model?

**Solution:** the first has negative width; the second refers to an undefined class.
An empty label file represents a negative under this course's explicit convention.
Neighboring frames are correlated; the validation set becomes too similar to training.

**Real-data handoff:** replace the generator with your images and YOLO labels, define
consistent annotation rules (occlusion, tiny objects and class boundaries), group-split
the data, visually inspect labels, and point notebook 07's `DATA_YAML` to your config.
Do not train on confidential or unauthorized images.

**Next:** [07 — Train your detector](07_Train_a_Custom_YOLO_Detector.ipynb).
Reference: [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect/).
''')])

write_notebook("13_Capstone_Reproducible_Detection_Project.ipynb", [
md('''
# 13 · Capstone: deliver a reproducible detection project

**Level:** advanced · **Class time:** 60–90 minutes · **Prerequisites:** 06–09; 12 for export.

**Brief:** detect and count three shape types for a small visual inspection demo.
Deliver annotated examples, per-image results, aggregate precision/recall, a saved
configuration and a short model report. The fully runnable baseline is a classical
shape detector; if notebook 07 produced a checkpoint, you can use the trained model.

**Presenter opening:** show an image, ask for a prediction, then expose the label and
the failure analysis. Do not describe synthetic validation as evidence of real-world readiness.
'''), bootstrap(),
md('''
## Step 1 — Freeze decisions before opening the test set

Use validation to choose confidence and IoU settings. The defaults below are fixed
teaching settings, not optimized thresholds. Set the engine now. Run the test once
for a final report. If you change settings after seeing test failures, those images
have become development data; collect a fresh test set for an unbiased final estimate.
'''),
code('''
import os, json, time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from course_utils import ensure_synthetic_dataset, audit_dataset, read_yolo_labels, CLASS_NAMES
DATA_YAML = ensure_synthetic_dataset()
DATA_ROOT = DATA_YAML.parent
ENGINE = "classical"  # Change to "yolo" after notebook 07 has saved a checkpoint.
CONFIDENCE = 0.25     # If tuned, use validation only and record the selection.
MATCH_IOU = 0.50
OUTPUT = ROOT / "artifacts" / "capstone"
OUTPUT.mkdir(parents=True, exist_ok=True)
display(pd.DataFrame(audit_dataset(DATA_YAML)))
'''),
md('''
## Step 2 — Implement a small inference interface

Each engine returns pixel boxes, zero-based class IDs and scores. The classical
detector thresholds dark foreground, finds external contours and uses polygon
vertices to classify shapes. Its score is a fixed **placeholder ranking value** of 1,
not a learned confidence. This hand-built rule can fail when lighting or outlines change.
'''),
code('''
def classical_predict(image):
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.uint8(gray < 205) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_boxes, found_classes = [], []
    for contour in contours:
        if cv2.contourArea(contour) < 30:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        class_id = 2 if len(polygon) == 3 else 0 if len(polygon) == 4 else 1
        x, y, w, h = cv2.boundingRect(contour)
        found_boxes.append([x, y, x+w, y+h])
        found_classes.append(class_id)
    return (np.asarray(found_boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(found_classes, dtype=np.int64), np.ones(len(found_boxes)))

model = None
checkpoint = None
if ENGINE == "yolo":
    from ultralytics import YOLO
    manifest_path = ROOT / "artifacts" / "custom_model.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run notebook 07 first or select ENGINE='classical'.")
    checkpoint = json.loads(manifest_path.read_text())["best_weights"]
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    model = YOLO(checkpoint)
elif ENGINE != "classical":
    raise ValueError("Choose classical or yolo")

def predict(image):
    if model is None:
        return classical_predict(image)
    r = model.predict(image, conf=CONFIDENCE, imgsz=256, device="cpu", verbose=False)[0]
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy())
'''),
md('''
## Step 3 — Match each prediction to at most one ground-truth box

Match in descending score order, on the same class, at IoU ≥ 0.5. A duplicate prediction
is a false positive; an unmatched ground truth is a false negative. A negative image
can have zero true positives and still be handled correctly. These aggregate counts
are **not mAP**; use notebook 08 / the framework evaluator for that metric.
'''),
code('''
def iou_one_to_many(box, candidates):
    if not len(candidates):
        return np.zeros(0)
    lo = np.maximum(box[:2], candidates[:, :2])
    hi = np.minimum(box[2:], candidates[:, 2:])
    intersection = np.clip(hi-lo, 0, None).prod(axis=1)
    area = np.clip(box[2:]-box[:2], 0, None).prod()
    areas = np.clip(candidates[:, 2:]-candidates[:, :2], 0, None).prod(axis=1)
    return intersection / np.maximum(area+areas-intersection, 1e-9)

def match_image(pred_boxes, pred_classes, scores, gt_boxes, gt_classes, threshold):
    used, tp = set(), 0
    for index in np.argsort(-scores, kind="stable"):
        candidates = np.array([j for j, c in enumerate(gt_classes)
                               if c == pred_classes[index] and j not in used], dtype=int)
        overlap = iou_one_to_many(pred_boxes[index], gt_boxes[candidates])
        if overlap.size and overlap.max() >= threshold:
            used.add(int(candidates[int(overlap.argmax())]))
            tp += 1
    return tp, len(pred_boxes)-tp, len(gt_boxes)-tp
'''),
md('''
## Step 4 — Warm up, evaluate and save traceable results

The measured latency includes the inference function and its preprocessing but excludes
file loading, plotting and matching. This is not camera-to-screen latency. The sample
is small, so do not generalize timing or accuracy to another device or domain.
'''),
code('''
# Warm up using a training image, before recording timings.
with Image.open(DATA_ROOT / "images/train/train_0001.png") as warmup_image:
    predict(warmup_image.convert("RGB"))
rows, examples = [], []
for path in sorted((DATA_ROOT / "images/test").glob("*.png")):
    image = Image.open(path).convert("RGB")
    gt_boxes, gt_classes = read_yolo_labels(DATA_ROOT / "labels/test" / (path.stem + ".txt"), *image.size)
    started = time.perf_counter()
    pb, pc, scores = predict(image)
    milliseconds = (time.perf_counter()-started)*1000
    tp, fp, fn = match_image(pb, pc, scores, gt_boxes, gt_classes, MATCH_IOU)
    rows.append({"image": path.name, "ground_truth_objects": len(gt_boxes), "predictions": len(pb),
                 "tp": tp, "fp": fp, "fn": fn, "latency_ms": milliseconds})
    if len(examples) < 6:
        examples.append((path.name, image.copy(), pb, pc, scores))
results = pd.DataFrame(rows)
results.to_csv(OUTPUT / "per_image_results.csv", index=False)
tp, fp, fn = results[["tp", "fp", "fn"]].sum()
summary = {"engine": ENGINE, "checkpoint": checkpoint, "split": "test", "images": len(results),
           "confidence": CONFIDENCE if ENGINE == "yolo" else None, "matching_iou": MATCH_IOU,
           "tp": int(tp), "fp": int(fp), "fn": int(fn),
           "precision": float(tp/(tp+fp)) if tp+fp else 0.0,
           "recall": float(tp/(tp+fn)) if tp+fn else 0.0,
           "median_latency_ms": float(results.latency_ms.median()),
           "p95_latency_ms": float(results.latency_ms.quantile(0.95)),
           "dataset": str(DATA_YAML), "scope": "Generated shapes only"}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
display(pd.DataFrame([summary]))
display(results.sort_values(["fn", "fp"], ascending=False).head())
'''),
code('''
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, (name, image, boxes, classes, scores) in zip(axes.flat, examples):
    show_boxes(image, boxes, [CLASS_NAMES[int(c)] for c in classes], scores=scores, ax=ax, title=name)
plt.tight_layout()
fig.savefig(OUTPUT / "predictions.png", dpi=140, bbox_inches="tight")
plt.show()
'''),
md('''
## Step 5 — Package the result with its limitations

A useful handoff contains the class mapping, dataset version, checkpoint, preprocessing,
threshold selection, metric definitions, environment and known failure cases. A model
file alone does not communicate those decisions. The template below records observed
metrics, not invented claims of production accuracy.
'''),
code('''
from importlib.metadata import version
environment = {name: version(name) for name in ["numpy", "torch", "torchvision", "ultralytics"]}
(OUTPUT / "environment.json").write_text(json.dumps(environment, indent=2))
report = f"""# Shape detection demonstration

Engine: {ENGINE}
Classes: {', '.join(CLASS_NAMES)}
Dataset: procedural shapes, generator version 1; 96 train / 24 validation / 24 test.
Evaluation: class-aware, one-to-one matches at IoU {MATCH_IOU}; not COCO mAP.
Observed precision: {summary['precision']:.3f}; recall: {summary['recall']:.3f}.
Confidence: {summary['confidence']} (None means classical rules, no learned score).
Median inference-call latency: {summary['median_latency_ms']:.2f} ms.

Intended use: classroom demonstration of a reproducible detection workflow.
Limitations: synthetic domain; small test set; uncalibrated scores; no real-camera validation.
Threshold selection: fixed classroom default; replace with a validation-based record if tuned.
Next experiment: collect independent real scenes and establish a fresh held-out test split.
Artifacts: summary.json, per_image_results.csv, predictions.png, environment.json.
"""
(OUTPUT / "model_report.md").write_text(report, encoding="utf-8")
print(report)
print("Saved to:", OUTPUT)
'''),
md('''
## Demonstrate your project in ten minutes

1. **Problem (1 min):** define the objects and one success criterion.
2. **Data (2 min):** show labels, negatives and group splits.
3. **Model (2 min):** explain your baseline and training choice.
4. **Evidence (3 min):** run inference, show failures and report measured metrics.
5. **Handoff (2 min):** show exported artifacts and explain remaining limitations.

**Extension challenge:** compare classical, YOLO and Faster R-CNN on the same validation
split; select one using accuracy and measured latency; then run a fresh final test.
Repeat on your own annotated photos with grouped splits. Keep each experiment's seed,
configuration and dataset version. Quantization, transformer detectors, distributed
training and production monitoring are follow-on subjects beyond this course's core.

**Answer/checkpoint:** a stronger model is not automatically a better deployment choice.
Pick using the requirements and measured tradeoffs, and communicate uncertainty from
limited data. Archive a failed example and propose a testable next experiment.

Reference: [Ultralytics validation metrics](https://docs.ultralytics.com/modes/val/).
''')])
