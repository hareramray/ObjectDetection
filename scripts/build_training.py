"""Generate the training, evaluation, augmentation, and Faster R-CNN lessons."""
from notebook_factory import md, code, bootstrap, write_notebook


def training():
    return [
        md("""
        # 07 · Train a custom YOLO detector

        **Level:** Intermediate · **Teaching time:** 35–50 minutes plus training
        (a CPU can need considerably longer; measure your own rehearsal).

        **Prerequisites:** notebooks 01–06; the project Jupyter kernel; internet for the
        first pretrained-weight download. A GPU is optional.

        **You will:** inspect a labeled dataset, fine-tune a pretrained detector,
        read real loss/metric plots, and save a checkpoint for later lessons.

        **Demo story:** “A general detector knows everyday objects. Can we teach it our
        three custom classes: rectangle, circle, and triangle?” This tiny synthetic
        dataset makes the mechanics reproducible. Five epochs demonstrate the workflow;
        they do **not** establish high accuracy or real-world readiness.

        Run cells from top to bottom. `COURSE_OFFLINE=1` leaves the inspection and
        teaching steps available and skips training; `COURSE_SMOKE=1` uses one epoch
        on one quarter of the training split. Neither mode fabricates a trained model.
        """),
        bootstrap(),
        md("""
        ## Choose the dataset

        Leave `CUSTOM_DATA_YAML = None` for the generated shape dataset. To teach your
        own classes, set it to a local YAML file; the class names are read from that
        file, so the saved model manifest describes the classes you actually trained.
        Both `names: [class_a, class_b]` and `names: {0: class_a, 1: class_b}` work.

        **Recommended layout:** `images/train`, `images/val`, optional `images/test`,
        with matching `labels/train`, `labels/val`, and `labels/test`. Each image has
        a same-stem `.txt` label file; create an empty one for a background-only image.
        Use `train: images/train`, `val: images/val`, and optionally `test: images/test`
        in the YAML. The inspection cells below scan PNG/JPG/JPEG files recursively in
        **one directory per split**. Other YOLO layouts such as image-list text files or
        multiple directories need adaptations to `split_images` and/or `label_path_for_image`.

        An omitted `path` means the YAML's folder. This lesson resolves a relative
        `path` against the YAML's folder and creates `artifacts/custom_dataset_resolved.yaml`
        with an absolute root for training. This avoids machine-dependent Ultralytics
        dataset settings. Your original custom YAML stays unchanged.
        """),
        code("""
        import os, json, time
        from datetime import datetime, timezone
        import yaml
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from PIL import Image
        from course_utils import ensure_synthetic_dataset, read_yolo_labels, load_yolo, choose_device

        OFFLINE = os.environ.get("COURSE_OFFLINE", "0") == "1"
        SMOKE = os.environ.get("COURSE_SMOKE", "0") == "1"
        CUSTOM_DATA_YAML = None  # Example: ROOT / "data" / "my_objects" / "data.yaml"
        SOURCE_DATA_YAML = ensure_synthetic_dataset() if CUSTOM_DATA_YAML is None else Path(CUSTOM_DATA_YAML).expanduser()
        if not SOURCE_DATA_YAML.is_absolute():
            SOURCE_DATA_YAML = ROOT / SOURCE_DATA_YAML
        SOURCE_DATA_YAML = SOURCE_DATA_YAML.resolve()
        data_spec = yaml.safe_load(SOURCE_DATA_YAML.read_text(encoding="utf-8"))
        if not isinstance(data_spec, dict):
            raise ValueError("Dataset YAML must contain a mapping with train, val, and names.")
        raw_names = data_spec.get("names")
        if isinstance(raw_names, dict):
            names_by_id = {int(key): str(value) for key, value in raw_names.items()}
            if len(names_by_id) != len(raw_names) or sorted(names_by_id) != list(range(len(raw_names))):
                raise ValueError("Class-name IDs must be consecutive integers starting at 0.")
            CLASS_NAMES = [names_by_id[index] for index in range(len(names_by_id))]
        elif isinstance(raw_names, list):
            CLASS_NAMES = [str(value) for value in raw_names]
        else:
            raise ValueError("Provide names as a list or a zero-based class-ID mapping in the YAML.")
        if not CLASS_NAMES or not all(CLASS_NAMES):
            raise ValueError("Provide at least one nonempty class name.")
        DATA_DIR = Path(data_spec.get("path") or SOURCE_DATA_YAML.parent).expanduser()
        if not DATA_DIR.is_absolute():
            DATA_DIR = SOURCE_DATA_YAML.parent / DATA_DIR
        DATA_DIR = DATA_DIR.resolve()
        DATA_YAML = SOURCE_DATA_YAML
        if CUSTOM_DATA_YAML is not None:
            DATA_YAML = ROOT / "artifacts" / "custom_dataset_resolved.yaml"
            DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
            resolved_spec = {**data_spec, "path": DATA_DIR.as_posix(), "names": dict(enumerate(CLASS_NAMES))}
            DATA_YAML.write_text(yaml.safe_dump(resolved_spec, sort_keys=False), encoding="utf-8")
        print("Source YAML:", SOURCE_DATA_YAML)
        print("Classes:", CLASS_NAMES)
        print(DATA_YAML.read_text(encoding="utf-8"))
        """),
        md("""
        ## 1. Inspect the training contract before spending compute

        Each YOLO label row is `class_id center_x center_y width height`, normalized
        by image width/height. Classes start at **0**. The YAML maps each integer to
        a name and identifies the train, validation, and test folders.

        Train images update weights; validation images help compare settings; hold the
        test split aside until the experiment is finished. New photographs of the same
        scene can still leak information across splits, even when filenames differ.

        **Say, then do:** “Wrong labels produce wrong learning.” Show three label files
        and overlay their boxes before pressing Train.
        """),
        code("""
        def split_images(split):
            entry = data_spec.get(split)
            if entry is None and split == "test":
                return []  # A reserved test split is recommended but optional here.
            if not isinstance(entry, str):
                raise ValueError(f"Set {split} to one image directory, or adapt split_images() for your layout.")
            directory = (DATA_DIR / entry).resolve()
            if not directory.is_dir():
                raise ValueError(f"This inspector needs a directory for {split}, not an image-list file: {directory}")
            return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})

        def label_path_for_image(image_path):
            parts = list(image_path.parts)
            if "images" not in parts:
                raise ValueError("Use matching images/... and labels/... folders, or adapt label_path_for_image().")
            parts[max(i for i, part in enumerate(parts) if part == "images")] = "labels"
            return Path(*parts).with_suffix(".txt")

        split_paths = {split: split_images(split) for split in ("train", "val", "test")}
        if not split_paths["train"] or not split_paths["val"]:
            raise ValueError("Train and validation directories each need at least one PNG/JPG/JPEG image.")
        counts = {split: len(paths) for split, paths in split_paths.items()}
        display(pd.DataFrame({"split": list(counts), "images": list(counts.values())}))
        samples = split_paths["train"][:3]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax in axes:
            ax.axis("off")
        for ax, image_path in zip(axes, samples):
            image = Image.open(image_path).convert("RGB")
            label_path = label_path_for_image(image_path)
            boxes, labels = read_yolo_labels(label_path, *image.size)
            if np.any(labels >= len(CLASS_NAMES)):
                raise ValueError(f"Class ID outside YAML names in {label_path}")
            show_boxes(image, boxes, labels, ax=ax, title=image_path.name)
            print(label_path.name, "→", label_path.read_text().strip().splitlines())
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 2. Make the budget explicit

        An **epoch** is a pass over the selected training images. A **batch** determines
        how many images contribute to one optimizer step. `imgsz` changes input resolution;
        `device` selects the GPU or CPU. Windows notebooks use `workers=0` here so data
        loading stays in the notebook process.

        **Classroom default:** This small dataset uses CPU training for a predictable
        local demonstration. To use an available CUDA GPU, change `DEVICE` below to
        `choose_device()`. Rehearse the selected device before presenting; timings depend
        on your hardware and dataset.

        We use the small YOLO11n checkpoint as a stable teaching baseline. Transfer
        learning starts from features learned on a larger dataset and replaces the
        class-specific prediction setup for the classes in your YAML.

        **Predict:** If the training loss decreases, does that prove generalization?
        No: inspect held-out validation behavior too.
        """),
        code("""
        FULL_RUN = False       # Optional: set True for 25 epochs, then rerun from this cell.
        EPOCHS = 1 if SMOKE else (25 if FULL_RUN else 5)
        TRAIN_FRACTION = 0.25 if SMOKE else 1.0
        DEVICE = "cpu"  # Classroom default; change to choose_device() to use an available CUDA GPU.
        train_config = dict(
            data=str(DATA_YAML), epochs=EPOCHS, imgsz=256, batch=8,
            workers=0, device=DEVICE, seed=42, deterministic=True,
            fraction=TRAIN_FRACTION, amp=False, cache=False,
            project=str(ROOT / "artifacts" / "yolo_runs"),
            name="shapes_yolo11n", exist_ok=False, plots=True,
        )
        print({k: v for k, v in train_config.items() if k != "data"})
        print("Training skipped:", OFFLINE)
        """),
        md("""
        ## 3. Train and keep an explicit record

        This cell performs real optimization and may download pretrained weights on
        its first run. Rehearse it before presenting. The console reports individual
        loss terms and validation metrics. With a tiny dataset, expect noisy results.

        Each run gets its own directory. `best.pt` is chosen using the trainer's
        validation criterion; `last.pt` contains the final saved state. The small
        manifest points later notebooks to the **actual** saved checkpoint.
        """),
        code("""
        trained_model = None
        run_dir = None
        manifest_path = ROOT / "artifacts" / "custom_model.json"
        if OFFLINE:
            print("Offline mode: training skipped. No new checkpoint or metrics were created.")
        else:
            trained_model = load_yolo("yolo11n.pt")
            started = time.perf_counter()
            training_result = trained_model.train(**train_config)
            elapsed = time.perf_counter() - started
            run_dir = Path(trained_model.trainer.save_dir).resolve()
            best_weights = Path(trained_model.trainer.best).resolve()
            if not best_weights.exists():
                raise FileNotFoundError(f"Training did not produce the expected checkpoint: {best_weights}")
            manifest = {
                "best_weights": str(best_weights), "data_yaml": str(DATA_YAML.resolve()),
                "run_dir": str(run_dir), "classes": CLASS_NAMES,
                "epochs_requested": EPOCHS, "train_fraction": TRAIN_FRACTION,
                "seed": 42, "imgsz": 256, "elapsed_seconds": round(elapsed, 2),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "short educational run; not a production quality claim",
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Completed in {elapsed / 60:.1f} min. Manifest: {manifest_path}")
            print("Best weights:", best_weights)
        """, tags=["network", "train"]),
        md("""
        ## 4. Read the learning curves

        A loss is the quantity optimization tries to reduce. Precision, recall, and
        mAP evaluate detections and need their own explanation (next lesson).
        Training and validation losses need not be numerically identical: augmentation,
        sample composition, and evaluation behavior differ.

        **Presenter prompt:** Point to an actual epoch on the plot. Ask what evidence
        would suggest underfitting or overfitting. One or five epochs rarely support a
        confident diagnosis; use a longer, controlled experiment before concluding.
        """),
        code("""
        if run_dir is None:
            print("No training run in this execution. Run the training cell online to plot its log.")
        else:
            history = pd.read_csv(run_dir / "results.csv")
            history.columns = history.columns.str.strip()
            display(history.tail())
            loss_cols = [c for c in history if c.startswith(("train/", "val/")) and "loss" in c]
            metric_cols = [c for c in history if c.startswith("metrics/")]
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            history.plot(x="epoch", y=loss_cols, ax=axes[0], marker="o", title="Recorded losses")
            if metric_cols:
                history.plot(x="epoch", y=metric_cols, ax=axes[1], marker="o", title="Validation metrics")
            plt.tight_layout()
            plt.show()
        """),
        md("""
        ## 5. Inspect held-out validation predictions

        A low threshold deliberately exposes weak predictions. Empty detections are a
        valid result for a short run. Never replace them with ground-truth boxes and
        call those model predictions.
        """),
        code("""
        if run_dir is not None:
            from ultralytics import YOLO
            best_model = YOLO(str(best_weights))
            val_images = split_paths["val"][:3]
            predictions = best_model.predict([str(p) for p in val_images], imgsz=256,
                                             conf=0.10, device=DEVICE, verbose=False)
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for ax in axes:
                ax.axis("off")
            for ax, prediction, source in zip(axes, predictions, val_images):
                # Ultralytics plot() returns a BGR image, so reverse channels for Matplotlib.
                ax.imshow(prediction.plot()[..., ::-1])
                ax.set_title(f"{source.name}: {len(prediction.boxes)} predictions")
                ax.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print("Prediction display needs this notebook's completed training run.")
        """, tags=["model-inference"]),
        md("""
        ## 6. Exercise: run one controlled comparison

        1. Keep the split, seed, image size, and starting weights fixed. Compare 5 and
           25 epochs. Record wall time and validation mAP50–95 from each run.
        2. Inspect at least three failures. Are they missing objects, wrong classes,
           duplicated boxes, or poor localization?
        3. Explain why you should not repeatedly tune settings against the test split.

        <details><summary>Solution / discussion</summary>

        Change only `FULL_RUN`; rebuild `train_config` and rerun the later cells.
        Each training cell creates a fresh model from the same pretrained weights.
        Copy each run's path and actual metrics into your experiment notes before
        starting another run; the manifest intentionally points to the latest run.
        More epochs may improve performance, do nothing, or overfit. Repeated test-set
        tuning turns that test set into a validation set and biases your final estimate.
        </details>

        **Troubleshooting:** CUDA out of memory → reduce `batch` to 4 or 2; keep
        `workers=0` on Windows. Slow CPU → use smoke mode for a rehearsal. Download
        error → check connectivity and rerun, or use offline mode for the other cells.
        Unexpected class names → verify the YAML and zero-based label IDs first.

        **Checkpoint:** Explain the split roles, locate `best.pt`, and identify the
        manifest. Next: notebook 08 explains what those validation metrics mean.

        **Primary references:** [Ultralytics training](https://docs.ultralytics.com/modes/train/),
        [detection dataset format](https://docs.ultralytics.com/datasets/detect/),
        [YOLO11](https://docs.ultralytics.com/models/yolo11/).
        """),
    ]


def evaluation():
    return [
        md("""
        # 08 · Evaluation: precision, recall, and mAP

        **Level:** Intermediate → advanced · **Time:** 40–55 minutes.
        **Prerequisites:** bounding boxes and IoU; notebook 07 only for the optional
        real-model section. The core lesson runs offline with NumPy and Matplotlib.

        **You will:** match detections to ground truth one-to-one, investigate duplicate
        and missed detections, sweep confidence, and compute educational 101-point AP.

        **Demo story:** “Six predicted boxes for four objects: is that good?” Counts
        alone cannot tell us. We need class labels, box overlap, confidence, and a rule
        that prevents two predictions from claiming the same object.
        """),
        bootstrap(),
        code("""
        import os, json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # Two deliberately simple images. These are hand-written examples, not model output.
        ground_truth = [
            {"image": "A", "class": 0, "box": [10, 10, 50, 50]},
            {"image": "A", "class": 1, "box": [65, 10, 100, 50]},
            {"image": "B", "class": 0, "box": [20, 20, 60, 60]},
            {"image": "B", "class": 1, "box": [70, 70, 110, 110]},
        ]
        predictions = [
            {"image": "A", "class": 0, "box": [10, 10, 50, 50], "score": .95, "note": "correct"},
            {"image": "A", "class": 0, "box": [11, 11, 50, 50], "score": .90, "note": "duplicate"},
            {"image": "A", "class": 1, "box": [66, 10, 101, 50], "score": .85, "note": "correct"},
            {"image": "B", "class": 0, "box": [23, 23, 63, 63], "score": .80, "note": "slightly shifted"},
            {"image": "B", "class": 1, "box": [0, 80, 20, 100], "score": .70, "note": "background"},
            {"image": "A", "class": 1, "box": [10, 10, 50, 50], "score": .60, "note": "wrong class"},
        ]
        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        for row, image_id in enumerate(["A", "B"]):
            canvas = np.full((120, 120, 3), 245, dtype=np.uint8)
            gt = [g for g in ground_truth if g["image"] == image_id]
            pred = [p for p in predictions if p["image"] == image_id]
            show_boxes(canvas, [g["box"] for g in gt], [g["class"] for g in gt],
                       ax=axes[row, 0], title=f"{image_id}: ground truth")
            show_boxes(canvas, [p["box"] for p in pred], [p["class"] for p in pred],
                       [p["score"] for p in pred], ax=axes[row, 1], title=f"{image_id}: toy predictions")
        plt.tight_layout()
        plt.show()
        display(pd.DataFrame(predictions))
        """),
        md("""
        ## 1. Match boxes one-to-one

        Sort predictions by confidence, highest first. For each prediction, consider
        **unmatched ground-truth boxes in the same image and class**. The highest-IoU
        eligible box is a match if IoU reaches the chosen threshold. A match is a true
        positive (TP); an unmatched prediction is a false positive (FP). Unmatched
        ground truth becomes a false negative (FN).

        **Presenter prompt:** Before running, count how many predictions should be true
        positives at IoU 0.50. Explain why the duplicate must count as a false positive.

        Coordinates here use continuous `xyxy`: area is `(x2-x1)*(y2-y1)` without `+1`.
        """),
        code("""
        def iou_one_to_many(box, boxes):
            boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
            if len(boxes) == 0:
                return np.zeros(0)
            box = np.asarray(box, dtype=float)
            top_left = np.maximum(box[:2], boxes[:, :2])
            bottom_right = np.minimum(box[2:], boxes[:, 2:])
            intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
            area = np.prod(np.maximum(box[2:] - box[:2], 0))
            other_area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
            return intersection / np.maximum(area + other_area - intersection, 1e-12)

        def match_detections(preds, truth, iou_threshold=.50, confidence=0.0):
            remaining = set(range(len(truth)))
            rows = []
            for p in sorted((p for p in preds if p["score"] >= confidence),
                            key=lambda p: p["score"], reverse=True):
                candidates = [j for j in sorted(remaining)
                              if truth[j]["image"] == p["image"] and truth[j]["class"] == p["class"]]
                overlaps = iou_one_to_many(p["box"], [truth[j]["box"] for j in candidates])
                best = int(np.argmax(overlaps)) if len(overlaps) else None
                is_tp = best is not None and overlaps[best] >= iou_threshold
                if is_tp:
                    remaining.remove(candidates[best])
                rows.append({**p, "TP": int(is_tp), "FP": int(not is_tp),
                             "matched_iou": float(overlaps[best]) if is_tp else 0.0})
            return rows, [truth[j] for j in sorted(remaining)]

        matched, missed = match_detections(predictions, ground_truth)
        display(pd.DataFrame(matched)[["image", "class", "score", "note", "TP", "FP", "matched_iou"]])
        print("Missed ground-truth objects:", missed)
        tp, fp, fn = sum(r["TP"] for r in matched), sum(r["FP"] for r in matched), len(missed)
        assert (tp, fp, fn) == (3, 3, 1)
        assert next(r for r in matched if r["note"] == "duplicate")["FP"] == 1
        print(f"TP={tp}; FP={fp}; FN={fn}")
        """),
        md("""
        ## 2. Confidence is an operating decision

        **Precision = TP / (TP + FP)** asks how many reported objects are correct.
        **Recall = TP / (TP + FN)** asks how many real objects were found.
        **F1** is their harmonic mean. A confidence threshold discards predictions;
        an IoU threshold decides whether a retained box is localized well enough.
        These are different controls.

        Here we report precision as zero when no predictions remain. Other reporting
        conventions may mark it undefined; always state the convention.
        """),
        code("""
        sweep_rows = []
        for threshold in [0.0, .65, .75, .82, .88, .93, .99]:
            rows, remaining = match_detections(predictions, ground_truth, confidence=threshold)
            tp = sum(r["TP"] for r in rows)
            fp = sum(r["FP"] for r in rows)
            fn = len(remaining)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            sweep_rows.append(dict(confidence=threshold, TP=tp, FP=fp, FN=fn,
                                   precision=precision, recall=recall, F1=f1))
        sweep = pd.DataFrame(sweep_rows)
        display(sweep.round(3))
        sweep.plot(x="confidence", y=["precision", "recall", "F1"], marker="o", ylim=(0, 1.05),
                   figsize=(8, 4), title="Toy detections: threshold trade-off at IoU 0.50")
        plt.show()
        """),
        md("""
        ## 3. Average precision summarizes a ranked list

        For each class, accumulate TP/FP along the confidence-sorted predictions to
        obtain precision and recall. At each recall level from 0.00 to 1.00 in steps
        of 0.01, take the best precision achieved at that recall or higher; use zero
        for an unreachable recall level. The mean of those 101 values is **AP101**.

        Average AP across classes to get mAP for one IoU threshold. Average again over
        IoU 0.50, 0.55, …, 0.95 for the educational mAP50–95 below. Missing classes with
        ground truth get AP=0; a class with no ground truth is undefined and excluded.

        This is a small transparent evaluator, **not the complete COCO evaluator**:
        it omits crowd/ignore annotations, area ranges, maximum-detection limits and
        other benchmark details. Use the official/established evaluator for a benchmark.
        """),
        code("""
        def class_ap101(preds, truth, class_id, iou_threshold=.50):
            class_gt = [g for g in truth if g["class"] == class_id]
            if not class_gt:
                return np.nan, np.array([]), np.array([])
            class_preds = [p for p in preds if p["class"] == class_id]
            rows, _ = match_detections(class_preds, class_gt, iou_threshold=iou_threshold)
            cumulative_tp = np.cumsum([r["TP"] for r in rows], dtype=float)
            cumulative_fp = np.cumsum([r["FP"] for r in rows], dtype=float)
            recall = cumulative_tp / len(class_gt)
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
            recall_grid = np.linspace(0, 1, 101)
            interpolated = [precision[recall >= r].max() if np.any(recall >= r) else 0.0
                            for r in recall_grid]
            return float(np.mean(interpolated)), precision, recall

        classes = sorted({g["class"] for g in ground_truth})
        iou_thresholds = np.linspace(.50, .95, 10)
        ap_rows = []
        for iou_threshold in iou_thresholds:
            for class_id in classes:
                ap, _, _ = class_ap101(predictions, ground_truth, class_id, iou_threshold)
                ap_rows.append({"IoU": round(float(iou_threshold), 2), "class": class_id, "AP101": ap})
        ap_table = pd.DataFrame(ap_rows)
        display(ap_table.pivot(index="IoU", columns="class", values="AP101").round(3))
        print("Educational mAP50:", round(ap_table.loc[ap_table["IoU"] == .5, "AP101"].mean(), 4))
        print("Educational mAP50–95:", round(ap_table["AP101"].mean(), 4))
        # A perfect set scores 1; no predictions score 0 when ground truth exists.
        perfect = [{**g, "score": 1.0} for g in ground_truth]
        assert np.isclose(class_ap101(perfect, ground_truth, 0)[0], 1.0)
        assert class_ap101([], ground_truth, 0)[0] == 0.0
        """),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for class_id in classes:
            ap, precision, recall = class_ap101(predictions, ground_truth, class_id)
            axes[0].step(recall, precision, where="post", marker="o", label=f"class {class_id}: AP={ap:.3f}")
        axes[0].set(xlim=(0, 1.05), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision",
                    title="Raw ranked precision/recall points")
        axes[0].legend()
        ap_table.groupby("IoU")["AP101"].mean().plot(ax=axes[1], marker="o")
        axes[1].set(ylim=(0, 1.05), ylabel="Mean AP101", title="Stricter localization requirements")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 4. Optional: evaluate the actual custom checkpoint

        The next cell uses the local checkpoint manifest from notebook 07. It evaluates
        the validation split with Ultralytics' implementation, allowing very low
        confidence predictions into the ranked evaluation (`conf=0.001`). That is
        different from selecting a threshold for a live demo.

        Compare the model's own metrics with visual failure cases. Do not compare these
        scores directly to the hand-written examples above: their data and evaluator
        details differ. Keep `split="val"` while tuning; evaluate the reserved test split
        once after choosing your configuration.
        """),
        code("""
        model_metrics = None
        manifest_path = ROOT / "artifacts" / "custom_model.json"
        OFFLINE = os.environ.get("COURSE_OFFLINE", "0") == "1"
        if OFFLINE:
            print("Offline lesson mode: optional Ultralytics validation skipped.")
        elif not manifest_path.exists():
            print("No checkpoint manifest. Complete notebook 07 to run real-model validation.")
        else:
            from ultralytics import YOLO
            from course_utils import choose_device
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            weights = Path(manifest["best_weights"])
            if not weights.exists():
                print("Manifest points to a moved/deleted checkpoint. Rerun notebook 07:", weights)
            else:
                model = YOLO(str(weights))
                model_metrics = model.val(data=manifest["data_yaml"], split="val", imgsz=256,
                                          batch=8, workers=0, device=choose_device(), conf=.001,
                                          plots=True, project=str(ROOT / "artifacts" / "evaluation"),
                                          name="custom_validation", verbose=False)
                display(pd.DataFrame([model_metrics.results_dict]))
                print("Validation plots:", model_metrics.save_dir)
        """, tags=["model-inference"]),
        md("""
        ## 5. Exercises and demonstration checkpoints

        1. Remove the duplicate prediction. Does recall change? Does AP change?
        2. Add a correct class-1 box for image B at confidence 0.55. What happens to
           achievable recall, even though a higher deployment threshold may hide it?
        3. Change the IoU threshold to 0.75. Which previous TP becomes an FP, and why
           does a corresponding FN appear?
        4. Choose a threshold for a task where missed objects are more costly than
           extra alerts. Explain why mAP alone does not choose that threshold.

        <details><summary>Solutions</summary>

        1. Recall stays the same; removing a high-ranked FP increases class-0 AP.
        2. The class can reach recall 1.0. AP improves, but a threshold above 0.55 still
           hides that box. Confidence is not a calibrated probability by default.
        3. The shifted class-0 box in B has IoU about 0.748 and fails at 0.75. Its
           prediction is an FP and the unmatched ground-truth object is an FN.
        4. Usually start with a lower threshold, then measure the alert burden on
           representative validation data. The acceptable trade-off is application-specific.
        </details>

        **Checkpoint:** Explain why a detection can be correctly classified but fail
        localization, and why two boxes on one object cannot both be TPs.

        **Primary references:** [COCO detection evaluation](https://cocodataset.org/#detection-eval),
        [COCO evaluator implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py),
        [Ultralytics validation](https://docs.ultralytics.com/modes/val/).
        """),
    ]


def augmentation():
    return [
        md("""
        # 09 · Augmentation and error analysis

        **Level:** Advanced · **Time:** 40–55 minutes.
        **Prerequisites:** notebooks 02, 06, and 08; notebook 07 is optional.

        **You will:** transform images and boxes together, clip/filter boxes after a
        crop, measure a simple detector under a synthetic domain shift, and turn a
        failure into a testable improvement hypothesis.

        **Demo story:** “The training demo works on a plain background. Does it still
        work when the appearance changes?” We will measure actual outputs rather than
        assume that an augmentation always improves a model.
        """),
        bootstrap(),
        code("""
        import os, json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import cv2
        from PIL import Image, ImageEnhance
        from course_utils import make_scene

        image, boxes, labels = make_scene(seed=19, size=256, min_objects=3, max_objects=3)
        show_boxes(image, boxes, labels, title="Original training-style scene")
        plt.show()
        """),
        md("""
        ## 1. A geometric transform must also transform the labels

        For a horizontal flip of an image of width W:
        `new_x1 = W - old_x2` and `new_x2 = W - old_x1`. The y coordinates stay fixed.
        We use continuous box boundaries, so this formula uses W rather than W−1.

        Photometric changes (brightness/color) normally leave box coordinates unchanged.
        Geometric changes (flip, crop, resize, rotation) need corresponding box updates.
        A rotated object's axis-aligned box may need to grow; this lesson starts with
        flips and crops whose coordinate rules are easy to inspect.

        **Presenter prompt:** Ask the audience to predict the flipped position of the
        leftmost object before drawing the transformed boxes.
        """),
        code("""
        def flip_horizontal(image, boxes, labels):
            transformed = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
            transformed[:, [0, 2]] = image.width - transformed[:, [2, 0]]
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), transformed, np.asarray(labels).copy()

        flipped, flipped_boxes, flipped_labels = flip_horizontal(image, boxes, labels)
        restored_image, restored_boxes, _ = flip_horizontal(flipped, flipped_boxes, flipped_labels)
        assert np.allclose(restored_boxes, boxes)
        assert np.array_equal(np.asarray(restored_image), np.asarray(image))
        brightened = ImageEnhance.Brightness(image).enhance(0.60)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        show_boxes(image, boxes, labels, ax=axes[0], title="Original")
        show_boxes(flipped, flipped_boxes, flipped_labels, ax=axes[1], title="Flip image + boxes")
        show_boxes(brightened, boxes, labels, ax=axes[2], title="Brightness: boxes unchanged")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 2. Crop, translate, clip, then filter

        Cropping changes the coordinate origin. Subtract the crop's left/top values,
        clip coordinates to its width/height, and discard boxes with no useful visible
        region. We use **retained bounding-box area / original box area** as a simple
        visibility proxy; it is not the exact visible area of the object.

        Filter the corresponding class labels with the same mask. A crop can correctly
        produce no objects: preserve boxes with shape `(0, 4)` and labels with shape `(0,)`.
        """),
        code("""
        def crop_with_boxes(image, boxes, labels, crop, min_visibility=0.30, min_side=3):
            left, top, right, bottom = map(int, crop)
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError("Crop must be a nonempty rectangle inside the image.")
            original = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            shifted = original - np.array([left, top, left, top], dtype=np.float32)
            shifted[:, [0, 2]] = np.clip(shifted[:, [0, 2]], 0, right - left)
            shifted[:, [1, 3]] = np.clip(shifted[:, [1, 3]], 0, bottom - top)
            size = np.maximum(shifted[:, 2:] - shifted[:, :2], 0)
            original_area = np.prod(np.maximum(original[:, 2:] - original[:, :2], 0), axis=1)
            visibility = np.prod(size, axis=1) / np.maximum(original_area, 1e-12)
            keep = (size[:, 0] >= min_side) & (size[:, 1] >= min_side) & (visibility >= min_visibility)
            return image.crop((left, top, right, bottom)), shifted[keep], np.asarray(labels)[keep], visibility

        crop = (60, 30, 220, 210)
        cropped, cropped_boxes, cropped_labels, visibility = crop_with_boxes(image, boxes, labels, crop)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        show_boxes(image, boxes, labels, ax=axes[0], title="Original and requested crop")
        from matplotlib.patches import Rectangle
        axes[0].add_patch(Rectangle((crop[0], crop[1]), crop[2]-crop[0], crop[3]-crop[1],
                                   fill=False, edgecolor="magenta", linewidth=3, linestyle="--"))
        show_boxes(cropped, cropped_boxes, cropped_labels, ax=axes[1], title=f"Kept {len(cropped_boxes)}/{len(boxes)} boxes")
        plt.tight_layout()
        plt.show()
        print("Retained box-area fractions:", np.round(visibility, 3))
        assert cropped_boxes.shape[1:] == (4,)
        if len(cropped_boxes):
            assert np.all(cropped_boxes[:, 2:] > cropped_boxes[:, :2])
            assert np.all(cropped_boxes >= 0)
            assert np.all(cropped_boxes[:, [0, 2]] <= cropped.width)
            assert np.all(cropped_boxes[:, [1, 3]] <= cropped.height)
        # A crop containing no object is a valid training example.
        _, empty_boxes, empty_labels, _ = crop_with_boxes(
            Image.new("RGB", (20, 20)), np.array([[1, 1, 5, 5]]), np.array([0]), (10, 10, 20, 20))
        assert empty_boxes.shape == (0, 4) and empty_labels.shape == (0,)
        """),
        md("""
        ## 3. Build an intentionally simple appearance baseline

        This baseline identifies pixels far from a **source-domain background color**
        and surrounds connected components with boxes. It classifies each component
        using its occupied area divided by its box area: approximately 1 for a rectangle,
        π/4 for a circle, and 1/2 for a triangle. The cutoffs 0.90 and 0.64 are simple
        fixed heuristics based on those ideal shapes, not learned model parameters.

        This is an actual runnable detector with a deliberately narrow assumption.
        It makes shortcuts visible: a good score on matching synthetic data can conceal
        dependence on the background. Shape colors in this dataset are randomized
        independently of class, so class-color prototypes would be an inappropriate
        classifier. We estimate source background from seed 100 and evaluate distinct
        seeds 4000–4019, outside the custom training/validation/test seed ranges.
        No evaluation labels are used to set the thresholds.
        """),
        code("""
        # Top-left corner is background in the generated scenes (inspect the generator).
        source_image, _, _ = make_scene(seed=100)
        source_background = np.asarray(source_image, dtype=np.float32)[0, 0]
        print("Fixed source-domain background RGB:", source_background)

        def color_component_detector(image, minimum_area=60):
            pixels = np.asarray(image, dtype=np.float32)
            foreground = np.linalg.norm(pixels - source_background, axis=2) > 50
            result_boxes, result_labels = [], []
            count, _, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
            for x, y, w, h, area in stats[1:]:
                if area >= minimum_area:
                    occupancy = area / (w * h)
                    class_id = 0 if occupancy > .90 else (1 if occupancy > .64 else 2)
                    result_boxes.append([x, y, x+w, y+h])
                    result_labels.append(class_id)
            return np.asarray(result_boxes, dtype=np.float32).reshape(-1, 4), np.asarray(result_labels, dtype=np.int64)

        baseline_boxes, baseline_labels = color_component_detector(image)
        show_boxes(image, baseline_boxes, baseline_labels, title="Measured foreground/component baseline predictions")
        plt.show()
        """),
        md("""
        ## 4. Quantify a domain shift

        `shift=True` asks the generator for changed visual conditions. Inspect the
        examples and its implementation in `course_utils.py` to understand what changed.
        Both groups use the same seed list, preserving geometry and class labels.
        Background brightness, object brightness, and image noise change together.
        This is a reproducible stress test, not a causal proof that one of those
        properties alone explains a score difference.

        The component baseline has no meaningful confidence scores. We report one
        fixed operating point: class-aware, one-to-one precision/recall at IoU 0.50.
        Matching is greedy in component order. Do not label this result mAP.
        """),
        code("""
        def count_matches(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=.5):
            unmatched = set(range(len(gt_boxes)))
            tp = 0
            for box, class_id in zip(pred_boxes, pred_labels):
                candidates = [j for j in sorted(unmatched) if gt_labels[j] == class_id]
                if not candidates:
                    continue
                candidates_boxes = np.asarray(gt_boxes)[candidates]
                intersection = np.prod(np.maximum(np.minimum(box[2:], candidates_boxes[:, 2:]) -
                                                    np.maximum(box[:2], candidates_boxes[:, :2]), 0), axis=1)
                area = np.prod(np.maximum(box[2:] - box[:2], 0))
                gt_area = np.prod(np.maximum(candidates_boxes[:, 2:] - candidates_boxes[:, :2], 0), axis=1)
                ious = intersection / np.maximum(area + gt_area - intersection, 1e-12)
                best = int(np.argmax(ious))
                if ious[best] >= iou_threshold:
                    unmatched.remove(candidates[best])
                    tp += 1
            return tp, len(pred_boxes) - tp, len(unmatched)

        shift_rows = []
        for shifted in [False, True]:
            totals = np.zeros(3, dtype=int)
            for seed in range(4000, 4020):
                sample, gt_boxes, gt_labels = make_scene(seed=seed, shift=shifted)
                pred_boxes, pred_labels = color_component_detector(sample)
                totals += count_matches(pred_boxes, pred_labels, gt_boxes, gt_labels)
            tp, fp, fn = map(int, totals)
            shift_rows.append({"domain": "shifted" if shifted else "source", "TP": tp, "FP": fp, "FN": fn,
                               "precision": tp / max(tp+fp, 1), "recall": tp / max(tp+fn, 1)})
        shift_results = pd.DataFrame(shift_rows)
        display(shift_results.round(3))
        shift_results.plot.bar(x="domain", y=["precision", "recall"], ylim=(0, 1.05), rot=0,
                               title="Measured component baseline · IoU 0.50", figsize=(7, 4))
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, shifted in zip(axes, [False, True]):
            sample, _, _ = make_scene(seed=4001, shift=shifted)
            pred_boxes, pred_labels = color_component_detector(sample)
            show_boxes(sample, pred_boxes, pred_labels, ax=ax,
                       title="Shifted predictions" if shifted else "Source predictions")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 5. Optional: inspect your trained YOLO model on both domains

        These are real predictions from the saved custom checkpoint. The plot is a
        small qualitative check, not an aggregate robustness score. A short training
        run may perform poorly even before the shift. Keep that baseline in view.
        """),
        code("""
        manifest_path = ROOT / "artifacts" / "custom_model.json"
        if os.environ.get("COURSE_OFFLINE", "0") == "1" or not manifest_path.exists():
            print("Optional checkpoint inference skipped. Train notebook 07 to enable it online.")
        else:
            from ultralytics import YOLO
            from course_utils import choose_device
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            weights = Path(manifest["best_weights"])
            if weights.exists():
                custom_model = YOLO(str(weights))
                scenes = [make_scene(seed=4001, shift=shifted)[0] for shifted in [False, True]]
                results = custom_model.predict(scenes, imgsz=256, conf=.10, device=choose_device(), verbose=False)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for ax, result, name in zip(axes, results, ["Source", "Shifted"]):
                    ax.imshow(result.plot()[..., ::-1]); ax.axis("off")
                    ax.set_title(f"{name}: {len(result.boxes)} model predictions")
                plt.tight_layout(); plt.show()
            else:
                print("Saved weights moved or missing:", weights)
        """, tags=["model-inference"]),
        md("""
        ## 6. Turn observations into an error-analysis plan

        | Error | Evidence to inspect | One controlled experiment |
        |---|---|---|
        | Missed object | Ground truth with no matching prediction | Collect more examples of that size/appearance |
        | Wrong class | Good overlap, wrong label | Check annotation consistency and class confusion |
        | Poor localization | Correct class, low IoU | Inspect box tightness and image resolution |
        | Duplicate | Multiple predictions for one object | Inspect NMS settings and crowded scenes |
        | Background false alarm | Prediction with no object | Add representative negative images |
        | Domain shift | Held-out subgroup performance changes | Add realistic variation to training, then remeasure |

        **Live exercise:** Change `min_visibility` from 0.30 to 0.80 and rerun the crop.
        Explain every removed label. Then choose one realistic change in your intended
        application and propose an augmentation. Do not augment validation/test images
        randomly during normal evaluation; keep evaluation conditions reproducible.

        **Advanced exercise:** Replace fixed-source-background subtraction with a
        per-image background estimate. Keep the evaluation seeds fixed and measure both
        groups again. Separate fitting data from evaluation data. Which shapes still
        fail because noise, occlusion, or low contrast changes their component geometry?

        <details><summary>Solutions / discussion</summary>

        Raising minimum visibility can only keep the same number or fewer boxes. An
        object may remain partly visible after its label is removed; aggressive cropping
        can therefore teach unintended negatives. Color jitter is reasonable when color
        varies in the real task, but harmful if color defines the label. A per-image
        background estimate can help the segmentation step without repairing dependence
        on the foreground mask. Measure the actual outcome; neither fix guarantees improvement.
        </details>

        **Checkpoint:** Demonstrate a correctly transformed box and state a measured
        failure with one proposed experiment. Next: tracking/video and model internals.

        **Primary references:** [TorchVision transforms and bounding boxes](https://docs.pytorch.org/vision/stable/transforms.html),
        [Ultralytics augmentation guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/).
        """),
    ]


def faster_rcnn():
    return [
        md("""
        # 11 · Faster R-CNN and an explicit transfer-learning loop

        **Level:** Advanced · **Time:** 45–60 minutes plus model download/compute.
        **Prerequisites:** notebooks 06–09; basic Python functions, tensors, and gradients;
        compatible PyTorch/TorchVision packages from the project requirements.

        **You will:** build a TorchVision detection Dataset, replace a classifier head,
        inspect loss components, perform two real optimizer steps, and run inference.

        **Demo story:** YOLO's training call packages the loop for us. Here we expose the
        data contract and gradient update using a two-stage Faster R-CNN model.
        Two batches demonstrate the mechanics; they are far too few to claim convergence.

        The normal path downloads pretrained weights once. With `COURSE_OFFLINE=1`,
        construction uses random weights, downloads nothing, and skips the expensive
        training/inference stages. Set `COURSE_FRCNN_OFFLINE_TRAIN=1` explicitly to test
        two updates on the random model offline; those are architecture checks, not
        pretrained transfer learning.
        """),
        bootstrap(),
        code("""
        import os, time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from PIL import Image
        import torch
        import torchvision
        from torch.utils.data import Dataset, DataLoader
        from torchvision.transforms.functional import pil_to_tensor
        from course_utils import ensure_synthetic_dataset, read_yolo_labels

        # Bootstrap precedes the torch import; seed torch now as well as NumPy/Python.
        seed_everything(42)

        OFFLINE = os.environ.get("COURSE_OFFLINE", "0") == "1"
        OFFLINE_TRAIN = os.environ.get("COURSE_FRCNN_OFFLINE_TRAIN", "0") == "1"
        USE_PRETRAINED = not OFFLINE
        RUN_OPTIMIZER_DEMO = not OFFLINE or OFFLINE_TRAIN
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DATA_DIR = ensure_synthetic_dataset().parent
        print("PyTorch:", torch.__version__, "TorchVision:", torchvision.__version__, "device:", DEVICE)
        """),
        md("""
        ## 1. Understand the two-stage pipeline

        `RGB image → backbone/FPN features → region proposals → ROI features → class + box`

        The **region proposal network (RPN)** proposes candidate regions. The **ROI head**
        classifies those regions and refines their coordinates. Training returns a
        dictionary of classification, regression, objectness, and proposal-regression
        losses. Evaluation returns boxes, labels, and scores.

        Unlike our zero-based YOLO labels, TorchVision reserves **0 for background**.
        Our rectangle, circle, and triangle labels become **1, 2, and 3**. The new head
        therefore needs **4 classes**, including background.
        """),
        code("""
        class ShapesDetectionDataset(Dataset):
            def __init__(self, split="train"):
                self.split = split
                self.image_paths = sorted((DATA_DIR / "images" / split).glob("*.png"))

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, index):
                path = self.image_paths[index]
                image = Image.open(path).convert("RGB")
                label_path = DATA_DIR / "labels" / self.split / (path.stem + ".txt")
                xyxy, class_ids = read_yolo_labels(label_path, *image.size)
                boxes = torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
                labels = torch.as_tensor(class_ids, dtype=torch.int64).reshape(-1) + 1
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                target = {
                    "boxes": boxes, "labels": labels,
                    "image_id": torch.tensor(index, dtype=torch.int64),
                    "area": area, "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
                }
                # Detector expects float CHW images in [0, 1]; it normalizes internally.
                tensor = pil_to_tensor(image).to(dtype=torch.float32) / 255.0
                return tensor, target

        def detection_collate(batch):
            # Variable image sizes / object counts cannot be stacked as fixed targets.
            images, targets = zip(*batch)
            return list(images), list(targets)

        train_dataset = ShapesDetectionDataset("train")
        val_dataset = ShapesDetectionDataset("val")
        loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0,
                            collate_fn=detection_collate,
                            generator=torch.Generator().manual_seed(42))
        sample_image, sample_target = train_dataset[1]  # Index 0 is a deliberate negative example.
        print("Image:", sample_image.shape, sample_image.dtype,
              "range:", (sample_image.min().item(), sample_image.max().item()))
        print({k: (tuple(v.shape), str(v.dtype)) for k, v in sample_target.items()})
        assert sample_target["boxes"].dtype == torch.float32
        assert sample_target["labels"].dtype == torch.int64
        assert torch.all((sample_target["labels"] >= 1) & (sample_target["labels"] <= 3))
        show_boxes(sample_image.permute(1, 2, 0).numpy(), sample_target["boxes"].numpy(),
                   (sample_target["labels"] - 1).numpy(), title="Dataset contract: displayed labels converted back to 0–2")
        plt.show()
        """),
        md("""
        ## 2. Replace the head and freeze the feature extractor

        Use the MobileNetV3/FPN variant to keep the exercise relatively small. We reduce
        the transform resolution to 256 and bound proposal counts for a classroom CPU
        demo; this changes the compute/accuracy trade-off from the pretrained defaults.

        The new `FastRCNNPredictor` learns our 4 outputs. We freeze the backbone to save
        compute, then train the RPN/ROI parameters. In training mode we also put the
        backbone in evaluation mode so its normalization behavior stays fixed. Batch
        size 2 and this explicit freeze avoid the single-image BatchNorm pitfall.
        """),
        code("""
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if USE_PRETRAINED else None
        detector = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights,
            weights_backbone=None,  # Important: no implicit ImageNet download in random/offline mode.
            min_size=256, max_size=256,
            rpn_pre_nms_top_n_train=300, rpn_post_nms_top_n_train=150,
            rpn_pre_nms_top_n_test=150, rpn_post_nms_top_n_test=100,
            box_detections_per_img=30,
        )
        in_features = detector.roi_heads.box_predictor.cls_score.in_features
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)
        for parameter in detector.backbone.parameters():
            parameter.requires_grad = False
        detector = detector.to(DEVICE)
        trainable = [p for p in detector.parameters() if p.requires_grad]
        print("Initialization:", "COCO-pretrained + new class head" if USE_PRETRAINED else "random/offline architecture")
        print("Trainable parameters:", f"{sum(p.numel() for p in trainable):,}")
        print("All parameters:", f"{sum(p.numel() for p in detector.parameters()):,}")
        print(detector.roi_heads.box_predictor)
        """, tags=["network", "model-build"]),
        md("""
        ## 3. Make two real optimizer updates

        The sequence is `zero_grad → forward → sum losses → backward → optimizer.step`.
        The forward pass needs **images and targets** during training. A non-finite
        loss is an error worth stopping for, not a number to hide.

        **Presenter prompt:** Predict which head parameters should change after one
        update. The code measures the classifier weight change to verify that an update
        happened. A finite loss or a changed weight does not prove detection quality.
        """),
        code("""
        loss_rows = []
        if not RUN_OPTIMIZER_DEMO:
            print("Offline mode: model architecture built; optimizer/inference demo skipped.")
        else:
            optimizer = torch.optim.SGD(trainable, lr=0.005, momentum=0.9, weight_decay=0.0005)
            detector.train()
            detector.backbone.eval()  # Freeze normalization statistics as well as gradients.
            initial_head = detector.roi_heads.box_predictor.cls_score.weight.detach().clone()
            started = time.perf_counter()
            for step, (images, targets) in enumerate(loader):
                if step >= 2:
                    break
                images = [im.to(DEVICE) for im in images]
                targets = [{k: v.to(DEVICE) for k, v in target.items()} for target in targets]
                optimizer.zero_grad(set_to_none=True)
                losses = detector(images, targets)
                total_loss = sum(losses.values())
                if not torch.isfinite(total_loss):
                    raise RuntimeError(f"Non-finite loss: {losses}")
                total_loss.backward()
                optimizer.step()
                row = {"step": step + 1, "total_loss": total_loss.detach().item(),
                       **{k: v.detach().item() for k, v in losses.items()}}
                loss_rows.append(row)
                print(row)
            weight_change = (detector.roi_heads.box_predictor.cls_score.weight.detach() - initial_head).abs().max().item()
            assert len(loss_rows) == 2 and weight_change > 0
            print(f"Two updates took {time.perf_counter() - started:.1f}s; max head weight change={weight_change:.6g}")
        """, tags=["train"]),
        code("""
        if loss_rows:
            loss_history = pd.DataFrame(loss_rows)
            display(loss_history.round(4))
            loss_history.set_index("step").drop(columns="total_loss").plot.bar(
                figsize=(9, 4), title="Measured loss terms for two different batches", rot=0)
            plt.ylabel("Loss"); plt.tight_layout(); plt.show()
        else:
            print("Loss chart appears after an optimizer demo. No synthetic losses are shown.")
        """),
        md("""
        ## 4. Switch to evaluation and inspect real predictions

        Call `eval()` and use `torch.inference_mode()`. During evaluation the model needs
        only images. Targets stay aside for scoring. Results use pixel `xyxy` boxes,
        integer labels, and confidence scores. Filter all three arrays using the same
        mask, then subtract 1 only when converting back to the course's class IDs.

        Expect noisy detections after two updates, and especially with random offline
        initialization. This cell demonstrates the output contract, not a finished model.
        """),
        code("""
        if RUN_OPTIMIZER_DEMO:
            detector.eval()
            val_image, val_target = val_dataset[1]
            with torch.inference_mode():
                prediction = detector([val_image.to(DEVICE)])[0]
            prediction = {key: value.detach().cpu() for key, value in prediction.items()}
            CONFIDENCE = 0.30
            keep = prediction["scores"] >= CONFIDENCE
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            rgb = val_image.permute(1, 2, 0).numpy()
            show_boxes(rgb, val_target["boxes"].numpy(), (val_target["labels"] - 1).numpy(),
                       ax=axes[0], title="Validation ground truth")
            show_boxes(rgb, prediction["boxes"][keep].numpy(), (prediction["labels"][keep]-1).numpy(),
                       prediction["scores"][keep].numpy(), ax=axes[1],
                       title=f"Actual model output: {int(keep.sum())} boxes ≥ {CONFIDENCE}")
            plt.tight_layout(); plt.show()
            print("Raw output shapes:", {k: tuple(v.shape) for k, v in prediction.items()})
        else:
            print("Run the optimizer demo to inspect this model's predictions.")
        """, tags=["model-inference"]),
        md("""
        ## 5. Exercises: move from a loop demonstration to an experiment

        1. Explain why `num_classes=3` would be wrong for our three foreground classes.
        2. Extend the loop to several epochs, evaluate the validation split after each
           epoch, and save the best state. Keep the test split untouched while tuning.
        3. Compare YOLO and Faster R-CNN using the **same data split, resolution policy,
           device, warm-up, timing scope, and evaluation protocol**. Do not compare
           these two update steps with a fully trained YOLO checkpoint as a fair contest.
        4. Verify that an empty target can use `boxes=torch.empty((0,4), dtype=float32)`
           and `labels=torch.empty((0,), dtype=int64)`. Why is `torch.tensor([])` insufficient
           for boxes?

        <details><summary>Solutions / discussion</summary>

        1. Background occupies ID 0; three foreground classes need 4 outputs.
        2. Repeat the DataLoader loop, select a checkpoint using validation only, and
           evaluate all images with an established detection evaluator. Accumulating
           losses is useful for training diagnostics but is not mAP.
        3. Report accuracy and latency separately. Warm up the model, synchronize CUDA
           when timing GPU work, and state whether preprocessing/postprocessing is timed.
        4. `torch.tensor([])` has shape `(0,)`; the box contract requires `(N,4)` even
           when N=0. Labels also need the correct integer dtype.
        </details>

        **Troubleshooting:** An operator such as `torchvision::nms` missing usually means
        incompatible PyTorch/TorchVision installations; reinstall a matched pair using
        the project setup guidance. “Target labels out of bounds” → check 1–3 labels
        and the four-class head. Out of memory → keep batch size 2, reduce image/proposal
        sizes, or use CPU. Do not blindly lower to batch 1 with a trainable BatchNorm backbone.

        **Checkpoint:** Show the target dictionary, identify background, explain each
        optimizer step, and show the measured classifier weight change.

        **Primary references:** [TorchVision detection fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html),
        [MobileNetV3 Faster R-CNN API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html),
        [Faster R-CNN paper](https://arxiv.org/abs/1506.01497).
        """),
    ]


if __name__ == "__main__":
    write_notebook("07_Train_a_Custom_YOLO_Detector.ipynb", training())
    write_notebook("08_Evaluation_Precision_Recall_and_mAP.ipynb", evaluation())
    write_notebook("09_Augmentation_and_Error_Analysis.ipynb", augmentation())
    write_notebook("11_Faster_RCNN_and_Transfer_Learning.ipynb", faster_rcnn())
