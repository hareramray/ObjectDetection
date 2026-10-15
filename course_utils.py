"""Shared plumbing for the notebooks; learning algorithms stay in the lessons."""
from pathlib import Path
import hashlib
import json
import os
import random
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["rectangle", "circle", "triangle"]
COLORS = ["#f16b58", "#26b6aa", "#6486ef"]

def ensure_dirs():
    for name in ["assets", "data", "models", "artifacts", "runs"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Avoid importing torch in the introductory notebooks.
    import sys
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def choose_device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"

def show_boxes(image, boxes, labels=None, scores=None, ax=None, title=None):
    """Display RGB pixels and half-open xyxy pixel boxes; return the matplotlib Axes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(image)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if labels is not None and len(labels) != len(boxes):
        raise ValueError("One label is required per box")
    if scores is not None and len(scores) != len(boxes):
        raise ValueError("One score is required per box")
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = COLORS[i % len(COLORS)]
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, lw=2, edgecolor=color))
        label = str(labels[i]) if labels is not None else str(i)
        if scores is not None:
            label += f" {scores[i]:.2f}"
        ax.text(x1, max(0, y1-3), label, color="white", fontsize=9,
                bbox={"facecolor": color, "alpha": 0.9, "pad": 2})
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax

def make_scene(seed=42, size=256, min_objects=1, max_objects=3, shift=False):
    """Original procedural artwork, with exact shape labels; no network required.

    Boxes use [x1,y1,x2,y2) pixel-edge coordinates. Color is randomized independently
    of shape class to avoid a trivial class/color shortcut. Shift changes brightness
    and noise, not the annotation coordinate system.
    """
    if size < 32 or min_objects < 0 or max_objects < min_objects or max_objects > 8:
        raise ValueError("size>=32 and 0<=min_objects<=max_objects<=8 required")
    rng = np.random.default_rng(seed)
    bg = 175 if shift else 235
    noise = rng.normal(0, 12 if shift else 3, (size, size, 3))
    image = Image.fromarray(np.uint8(np.clip(bg + noise, 0, 255)))
    draw = ImageDraw.Draw(image)
    boxes, labels = [], []
    n_objects = int(rng.integers(min_objects, max_objects + 1))
    for _ in range(n_objects):
        for attempt in range(200):
            w = int(rng.integers(max(8, size//8), max(10, size//3)))
            h = int(rng.integers(max(8, size//8), max(10, size//3)))
            label = int(rng.integers(0, len(CLASS_NAMES)))
            if label == 1:
                h = w  # a circle, not an ellipse
            x = int(rng.integers(2, size-w-1))
            y = int(rng.integers(2, size-h-1))
            candidate = [x, y, x+w, y+h]
            if all(x+w+3 <= b[0] or x >= b[2]+3 or y+h+3 <= b[1] or y >= b[3]+3 for b in boxes):
                break
        else:
            raise RuntimeError("Could not place all shapes; reduce max_objects.")
        color = tuple(int(v) for v in rng.integers(35, 175, 3))
        if shift:
            color = tuple(min(255, v+40) for v in color)
        bounds = [x, y, x+w-1, y+h-1]  # Pillow includes its endpoint.
        if label == 0:
            draw.rectangle(bounds, fill=color)
        elif label == 1:
            draw.ellipse(bounds, fill=color)
        else:
            draw.polygon([(x+w//2, y), (x, y+h-1), (x+w-1, y+h-1)], fill=color)
        boxes.append(candidate)
        labels.append(label)
    return image, np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64)

def xyxy_to_yolo(boxes, width, height):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if width <= 0 or height <= 0:
        raise ValueError("Positive image dimensions required")
    return np.column_stack(((boxes[:, 0]+boxes[:, 2])/(2*width),
                            (boxes[:, 1]+boxes[:, 3])/(2*height),
                            (boxes[:, 2]-boxes[:, 0])/width,
                            (boxes[:, 3]-boxes[:, 1])/height))

def read_yolo_labels(path, width, height):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.int64)
    rows = np.array([[float(v) for v in line.split()] for line in text.splitlines()])
    if rows.ndim != 2 or rows.shape[1] != 5 or not np.isfinite(rows).all():
        raise ValueError(f"Expected finite class cx cy w h rows: {path}")
    c, cx, cy, w, h = rows.T
    if np.any(c != c.astype(int)) or np.any(c < 0):
        raise ValueError(f"Class IDs must be nonnegative integers: {path}")
    if np.any(rows[:, 1:] < 0) or np.any(rows[:, 1:] > 1) or np.any(w <= 0) or np.any(h <= 0):
        raise ValueError(f"Invalid normalized coordinates: {path}")
    boxes = np.column_stack(((cx-w/2)*width, (cy-h/2)*height,
                             (cx+w/2)*width, (cy+h/2)*height))
    if np.any(boxes < -1e-3) or np.any(boxes[:, [0, 2]] > width+1e-3) or np.any(boxes[:, [1, 3]] > height+1e-3):
        raise ValueError(f"Box extends beyond image: {path}")
    return boxes.astype(np.float32), c.astype(np.int64)

def ensure_synthetic_dataset():
    """Create a deterministic, separate train/val/test dataset. Never remove user data."""
    import yaml
    ensure_dirs()
    folder = ROOT / "data" / "synthetic_shapes"
    config = {"generator_version": 1, "size": 256, "splits": {"train": 96, "val": 24, "test": 24},
              "seed_starts": {"train": 1000, "val": 2000, "test": 3000}, "names": CLASS_NAMES}
    marker = folder / "manifest.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError("Existing synthetic dataset uses another configuration. Rename its folder to preserve it.")
    for split, count in config["splits"].items():
        image_dir, label_dir = folder / "images" / split, folder / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            image_path, label_path = image_dir / f"{split}_{i:04d}.png", label_dir / f"{split}_{i:04d}.txt"
            if image_path.exists() and label_path.exists():
                continue
            if image_path.exists() or label_path.exists():
                raise FileExistsError(
                    f"Incomplete image/label pair: {image_path.name}. Restore its matching file "
                    "or move both files aside before regenerating; existing files are preserved.")
            # Every twelfth image is a negative example with an empty label file.
            image, boxes, labels = make_scene(config["seed_starts"][split]+i,
                                            min_objects=0 if i % 12 == 0 else 1,
                                            max_objects=0 if i % 12 == 0 else 3)
            image.save(image_path)
            rows = xyxy_to_yolo(boxes, *image.size)
            label_path.write_text("".join(f"{int(c)} " + " ".join(f"{v:.8f}" for v in row) + "\n"
                                          for c, row in zip(labels, rows)), encoding="utf-8")
    yaml_path = folder / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": folder.resolve().as_posix(),
                                        "train": "images/train", "val": "images/val", "test": "images/test",
                                        "names": dict(enumerate(CLASS_NAMES))}, sort_keys=False), encoding="utf-8")
    marker.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return yaml_path

def audit_dataset(yaml_path):
    """Audit actual files, bounds, class IDs and exact cross-split image duplicates."""
    import yaml
    config = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    root = Path(config["path"])
    names = config["names"]
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("YAML names keys must be consecutive integers starting at zero")
        names = [names[i] for i in range(len(names))]
    if not names or len(set(names)) != len(names):
        raise ValueError("YAML names must contain unique class names")
    seen, report = {}, []
    for split in ("train", "val", "test"):
        image_dir = root / config[split]
        images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        if not images:
            raise ValueError(f"No images in {image_dir}")
        counts = np.zeros(len(names), dtype=int)
        negatives = 0
        for path in images:
            label_path = root / "labels" / split / (path.stem + ".txt")
            if not label_path.exists():
                raise FileNotFoundError(f"Course convention requires explicit empty labels for negatives: {label_path}")
            with Image.open(path) as image:
                image.load()  # Decode pixels as well as reading the image header.
                boxes, labels = read_yolo_labels(label_path, *image.size)
            if np.any(labels >= len(counts)):
                raise ValueError(f"Class outside names mapping: {label_path}")
            counts += np.bincount(labels, minlength=len(counts))
            negatives += len(boxes) == 0
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f"Exact image leakage: {path} also appears in {seen[digest]}")
            seen[digest] = split
        report.append({"split": split, "images": len(images), "negative_images": negatives,
                       **dict(zip(names, counts.tolist()))})
    return report

def get_sample_image():
    ensure_dirs()
    path = ROOT / "assets" / "bus.jpg"
    if not path.exists():
        if os.getenv("COURSE_OFFLINE") == "1":
            raise RuntimeError("bus.jpg is not cached. Run scripts/prepare_demo.py once with internet.")
        url = "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg"
        temporary = path.with_suffix(".download")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ObjectDetectionCourse/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            with Image.open(temporary) as check:
                check.verify()
            temporary.replace(path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Cannot download the sample. Check internet, or save your own image as assets/bus.jpg.") from exc
    return path

def load_yolo(weights="yolo11n.pt"):
    from ultralytics import YOLO
    ensure_dirs()
    weights = Path(weights)
    path = weights if weights.is_absolute() else ROOT / "models" / weights
    if not path.exists() and os.getenv("COURSE_OFFLINE") == "1":
        raise FileNotFoundError(f"Cache the weights before going offline: {path}")
    return YOLO(str(path))

def make_demo_video():
    """Pan a cached still image. This illustrates API mechanics, not tracking quality."""
    import cv2
    ensure_dirs()
    path = ROOT / "assets" / "bus_pan.avi"
    if path.exists():
        return path
    frame = cv2.imread(str(get_sample_image()))
    frame = cv2.resize(frame, (480, 640))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (480, 640))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MJPG video writer.")
    try:
        for i in range(48):
            matrix = np.float32([[1, 0, 50*np.sin(i/47*np.pi*2)], [0, 1, 0]])
            writer.write(cv2.warpAffine(frame, matrix, (480, 640), borderMode=cv2.BORDER_REFLECT))
    finally:
        writer.release()
    return path
