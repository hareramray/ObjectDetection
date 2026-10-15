"""Build the pretrained inference, tracking, and ONNX demonstrations."""
from notebook_factory import md, code, bootstrap, write_notebook


def build_detection():
    cells = [
        md("""
        # 04 · Detect everyday objects with pretrained YOLO

        **Level:** beginner → intermediate · **Presentation:** 25–40 minutes · **Hardware:** CPU is enough.

        **By the end:** run a real pretrained detector, inspect its boxes, explain confidence versus
        NMS IoU, and save a result that another program can consume. Prerequisites: notebooks 01–03.
        We deliberately use the small, established **YOLO11n** detector, not a claim about the latest
        YOLO release. Its COCO vocabulary does not include every object you can imagine.

        Run cells from top to bottom. The first normal run downloads official example weights and a
        bus photograph; later runs reuse local files. `COURSE_OFFLINE=1` skips model/download sections
        explicitly. An offline skip is **not** evidence that real inference passed.

        **Presenter opening:** “A detector predicts a class, a location, and a score for each candidate
        object. Let us inspect all three before trusting the picture.”
        """),
        bootstrap(),
        code("""
        import os
        import json
        import time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from PIL import Image
        from IPython.display import display
        from course_utils import get_sample_image, load_yolo

        OFFLINE = os.getenv("COURSE_OFFLINE", "0") == "1"
        SMOKE = os.getenv("COURSE_SMOKE", "0") == "1"
        MODEL = "yolo11n.pt"
        DEVICE = "cpu"       # Set to 0 only after confirming a CUDA-enabled PyTorch installation.
        IMAGE_SIZE = 320     # Try 640 later; small objects may benefit at a latency cost.
        CONFIDENCE = 0.25
        NMS_IOU = 0.70
        OUT = ROOT / "outputs" / "04_pretrained"
        OUT.mkdir(parents=True, exist_ok=True)
        model = image = result = None
        print({"offline": OFFLINE, "model": MODEL, "device": DEVICE, "imgsz": IMAGE_SIZE})
        """),
        md("""
        ## 1. Inspect the input before predicting

        PIL uses RGB; OpenCV generally uses BGR. Here we pass a PIL image to the detector, so the
        channel convention is unambiguous. The model will resize and pad the image internally;
        reported `xyxy` coordinates are mapped back to the original image dimensions.

        **Predict aloud:** name objects you can see, including partially hidden ones. This is a
        qualitative inspection, not a labeled evaluation set.
        """),
        code("""
        if OFFLINE:
            print("Skipped: sample download and pretrained model inference (COURSE_OFFLINE=1).")
        else:
            image_path = get_sample_image()
            image = Image.open(image_path).convert("RGB")
            print("Cached image:", image_path, "| RGB size (width, height):", image.size)
            display(image.copy().resize((405, round(image.height * 405 / image.width))))
        """, tags=["network"]),
        md("""
        ## 2. Run the detector

        `conf` filters weak predictions. `iou` controls overlap suppression between competing boxes
        in this NMS-based model. They are different thresholds. A confidence of 0.8 is not a guarantee
        of 80% correctness on your camera or dataset. Keep thresholds fixed when comparing models.
        """),
        code("""
        if not OFFLINE:
            model = load_yolo(MODEL)
            started = time.perf_counter()
            result = model.predict(image, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                                   iou=NMS_IOU, device=DEVICE, verbose=False)[0]
            print(f"First call: {1000 * (time.perf_counter() - started):.1f} ms (includes initialization)")
            print("Classes available:", len(model.names))
            print("Example class names:", list(model.names.items())[:10])
            print("Detections:", len(result.boxes))
        """, tags=["network"]),
        md("""
        ## 3. Turn tensors into a readable table

        Each row is a detection. `x1, y1` is the upper-left corner; `x2, y2` is the lower-right.
        `.cpu().numpy()` moves results to ordinary arrays so plotting and data analysis work even
        if you later choose a GPU. Empty results are valid and must not crash an application.
        """),
        code("""
        def detection_table(prediction):
            columns = ["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
            rows = []
            for box, score, class_id in zip(prediction.boxes.xyxy.cpu().numpy(),
                                             prediction.boxes.conf.cpu().numpy(),
                                             prediction.boxes.cls.cpu().numpy().astype(int)):
                rows.append([int(class_id), prediction.names[int(class_id)], float(score),
                             *[float(v) for v in box]])
            return pd.DataFrame(rows, columns=columns)

        detections = pd.DataFrame()
        if result is not None:
            detections = detection_table(result)
            display(detections.round(3))
            fig, ax = plt.subplots(figsize=(8, 8))
            show_boxes(np.asarray(image), detections[["x1", "y1", "x2", "y2"]].to_numpy(),
                       labels=detections["class_name"].tolist(), scores=detections["confidence"].tolist(),
                       ax=ax, title="YOLO11n: inspect boxes, labels, and scores")
            plt.show()
        """),
        md("""
        ## 4. Change one knob: confidence threshold

        First predict what happens at 0.10, 0.25, 0.50, and 0.80. Lower thresholds usually keep more
        detections, including uncertain ones. A larger count does not mean better accuracy.
        This experiment uses actual model calls; it does not alter the saved baseline result.
        """),
        code("""
        sweep = pd.DataFrame()
        if model is not None:
            rows = []
            for threshold in ([0.25, 0.80] if SMOKE else [0.10, 0.25, 0.50, 0.80]):
                prediction = model.predict(image, imgsz=IMAGE_SIZE, conf=threshold,
                                           iou=NMS_IOU, device=DEVICE, verbose=False)[0]
                table = detection_table(prediction)
                rows.append({"confidence_threshold": threshold, "detections": len(table),
                             "class_names": ", ".join(sorted(set(table["class_name"])))})
            sweep = pd.DataFrame(rows)
            display(sweep)
            ax = sweep.plot(x="confidence_threshold", y="detections", marker="o", legend=False,
                            title="Threshold changes retained predictions, not ground truth")
            ax.set_ylabel("Retained detections")
            ax.set_ylim(bottom=0)
            plt.show()
        """, tags=["network"]),
        md("""
        ## 5. Filter a class, crop detections, and save an artifact

        We look up the class ID by its **name**, avoiding a hidden dependency on a hard-coded index.
        Crops are clamped to image bounds. Saved coordinates remain in original-image pixels.
        The JSON is suitable for a later app, but predictions still need evaluation and error handling.
        """),
        code("""
        if result is not None:
            person_ids = [class_id for class_id, name in result.names.items() if name == "person"]
            people = detections[detections["class_id"].isin(person_ids)].copy()
            print("Predicted people in this single image:", len(people))
            for number, (_, row) in enumerate(people.head(4).iterrows()):
                x1, y1, x2, y2 = row[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
                crop_box = (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
                            min(image.width, int(np.ceil(x2))), min(image.height, int(np.ceil(y2))))
                if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                    crop = image.crop(crop_box)
                    crop.thumbnail((160, 240))
                    display(crop)
            payload = {"model": MODEL, "image": image_path.name, "width": image.width,
                       "height": image.height, "coordinate_format": "xyxy_pixels",
                       "confidence_threshold": CONFIDENCE, "nms_iou": NMS_IOU,
                       "predictions": detections.to_dict(orient="records")}
            (OUT / "detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            detections.to_csv(OUT / "detections.csv", index=False)
            # Ultralytics result.plot() returns BGR; convert before saving through PIL.
            Image.fromarray(result.plot()[..., ::-1]).save(OUT / "annotated.png")
            print("Saved JSON, CSV, and annotated.png to", OUT)
        else:
            print("No model artifacts written because inference was skipped.")
        """),
        md("""
        ## 6. Demonstration challenges

        1. Set `IMAGE_SIZE=640`, rerun inference, and inspect small or occluded objects. Do not infer
           accuracy from a single image. Why is the first timing unsuitable as a steady-state benchmark?
        2. Compare NMS IoU values 0.30 and 0.90 while keeping confidence fixed. A larger NMS threshold
           permits more overlap; it does **not** require predictions to match ground truth better.
        3. Replace the sample with your own image: set `image = Image.open(ROOT / "assets" / "my_photo.jpg").convert("RGB")`
           and `image_path = ROOT / "assets" / "my_photo.jpg"`, then rerun steps 2–5.
        4. Explain a miss, a false detection, and a poor box. Which needs a threshold change and which
           might require more representative training data?

        **Reveal after discussion — solution to challenge 2:**
        """),
        code("""
        if model is not None:
            comparison = []
            for overlap in [0.30, 0.90]:
                prediction = model.predict(image, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                                           iou=overlap, device=DEVICE, verbose=False)[0]
                comparison.append({"nms_iou": overlap, "detections": len(prediction.boxes)})
            display(pd.DataFrame(comparison))
            print("Equal counts are possible when boxes do not compete at these IoU values.")
        """, tags=["network"]),
        md("""
        ## Presenter checkpoint and troubleshooting

        You should be able to point to the original input, a prediction table, an annotated image,
        a threshold sweep, and three saved files. The exact boxes can vary with package versions.

        | Symptom | Practical action |
        |---|---|
        | First download fails | Check internet access; rerun after connectivity returns. Do not substitute invented detections. |
        | `No module named ultralytics` | Select the course kernel and install the root `requirements.txt`. |
        | No detections on your photo | Inspect image orientation, class vocabulary, resolution, then confidence. |
        | Wrong colors | Pass a PIL RGB image or respect OpenCV's BGR convention. |
        | High first-call latency | Warm up before timing; notebook 12 demonstrates this. |

        **Next:** temporal IDs and actual crossing events in notebook 05.

        Primary references: [YOLO11](https://docs.ultralytics.com/models/yolo11/),
        [prediction arguments and Results](https://docs.ultralytics.com/modes/predict/).
        """),
    ]
    write_notebook("04_Pretrained_YOLO_Detection.ipynb", cells)


def build_tracking():
    cells = [
        md("""
        # 05 · Video tracking and counting crossings

        **Level:** intermediate · **Presentation:** 35–50 minutes · **Prerequisite:** notebook 04.

        Detecting a person in 30 frames does not mean 30 people visited. Here we separate detections,
        tracker IDs, and **line-crossing events**. You will build a deterministic counter first, then
        connect it to YOLO11n + ByteTrack on video. CPU, bounded loops, and inline Jupyter output are defaults.

        The generated demo clip pans a single bus photograph. It is an **artificial repeated scene**
        for exercising the pipeline, not real object motion, physical visitor counts, or a tracking
        accuracy benchmark. For meaningful counting, use a fixed camera, an appropriate line, and
        manually labeled real video. An ID is a temporary tracker label, not a verified identity.

        **Presenter question to pose aloud:** “What exactly should we count: frames, track IDs, or events?”
        """),
        bootstrap(),
        code("""
        import os
        import importlib.util
        from dataclasses import dataclass, field
        from collections import Counter
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from PIL import Image
        from IPython.display import display
        from course_utils import make_demo_video, load_yolo

        OFFLINE = os.getenv("COURSE_OFFLINE", "0") == "1"
        SMOKE = os.getenv("COURSE_SMOKE", "0") == "1"
        MODEL = "yolo11n.pt"
        DEVICE = "cpu"
        MAX_FRAMES = 8 if SMOKE else 60
        SHOW_EVERY = 10
        LINE_FRACTION = 0.40  # Count line at 40% of image width; choose for your own scene.
        OUT = ROOT / "outputs" / "05_tracking"
        OUT.mkdir(parents=True, exist_ok=True)
        """),
        md("""
        ## 1. Define the event before writing the loop

        Our vertical line has a small deadband. A track must be seen clearly on one side and later
        clearly on the other. Centers inside the deadband preserve the last stable side. This prevents
        tiny jitter near the line from generating an immediate event. We count each **ID + direction**
        at most once per run. A first observation already on the right is not a crossing.

        Long observation gaps reset the previous side, because a crossing during a long absence is
        uncertain. Short gaps may still infer a crossing. This is an explicit policy, not perfect truth.
        """),
        code("""
        @dataclass
        class LineCounter:
            line_x: float
            margin: float = 8.0
            max_gap: int = 8
            state: dict = field(default_factory=dict)
            counted: set = field(default_factory=set)
            events: list = field(default_factory=list)

            def update(self, track_id, x_center, frame_index):
                track_id = int(track_id)
                old_side, last_frame = self.state.get(track_id, (None, frame_index))
                if frame_index - last_frame > self.max_gap:
                    old_side = None
                side = (-1 if x_center < self.line_x - self.margin else
                        1 if x_center > self.line_x + self.margin else 0)
                if side == 0:
                    self.state[track_id] = (old_side, frame_index)
                    return None
                event = None
                if old_side is not None and side != old_side:
                    direction = "left_to_right" if side == 1 else "right_to_left"
                    key = (track_id, direction)
                    if key not in self.counted:
                        self.counted.add(key)
                        event = {"frame": int(frame_index), "track_id": track_id,
                                 "direction": direction, "x_center": float(x_center)}
                        self.events.append(event)
                self.state[track_id] = (side, frame_index)
                return event

            @property
            def totals(self):
                tally = Counter(event["direction"] for event in self.events)
                return {"left_to_right": tally["left_to_right"],
                        "right_to_left": tally["right_to_left"]}
        """),
        md("""
        ## 2. Test the counting rule without a model or internet

        These are **known synthetic center trajectories**, not detector outputs. ID 1 moves right,
        ID 2 moves left, and ID 3 jitters in the deadband. Expect exactly one crossing each way.
        Keeping the event logic separate makes it possible to debug counting independently of detection.
        """),
        code("""
        trajectories = {
            1: [20, 30, 40, 47, 50, 54, 62, 72, 80],
            2: [82, 75, 68, 60, 53, 48, 39, 30, 22],
            3: [50, 49, 51, 50, 52, 48, 49, 51, 50],
        }
        demo_counter = LineCounter(line_x=50, margin=5)
        for frame_index in range(9):
            for track_id, positions in trajectories.items():
                demo_counter.update(track_id, positions[frame_index], frame_index)
        display(pd.DataFrame(demo_counter.events))
        print(demo_counter.totals)
        assert demo_counter.totals == {"left_to_right": 1, "right_to_left": 1}

        fig, ax = plt.subplots(figsize=(9, 4))
        for track_id, positions in trajectories.items():
            ax.plot(positions, marker="o", label=f"known ID {track_id}")
        ax.axhspan(45, 55, alpha=0.15, color="gray", label="deadband")
        ax.axhline(50, color="black", linestyle="--")
        ax.set(xlabel="Frame index", ylabel="Horizontal center (pixels)",
               title="Event logic can be verified before running a detector")
        ax.legend()
        plt.show()
        """),
        md("""
        ## 3. Prepare a short video and check tracking dependencies

        `make_demo_video()` uses the cached/downloaded official sample image and creates a short
        artificial pan locally. Set `VIDEO_PATH` to your own local file for real footage.
        The line at 40% of image width intersects a person's image trajectory in the artificial pan;
        moving the line to the center may produce no events. Choose placement to fit the real scene.
        Only one source runs at a time. A fresh model is created per source so old track IDs do not
        leak from one clip into another. ByteTrack's assignment solver needs the `lap` package.
        """),
        code("""
        VIDEO_PATH = None  # Example: ROOT / "assets" / "my_fixed_camera_clip.mp4"
        tracking_ready = not OFFLINE
        if OFFLINE:
            print("Offline mode: real video/model section skipped; deterministic counter above still runs.")
        elif importlib.util.find_spec("lap") is None:
            tracking_ready = False
            print("Tracking needs lap. Install the root requirements.txt in this kernel, restart, and rerun.")
        else:
            import cv2
            VIDEO_PATH = Path(VIDEO_PATH) if VIDEO_PATH is not None else make_demo_video()
            if not VIDEO_PATH.exists():
                raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
            print("Video:", VIDEO_PATH)
        """, tags=["network"]),
        md("""
        ## 4. Track consecutive frames and update the counter

        `persist=True` preserves state across consecutive frames. `classes=[person_id]` limits this
        demonstration to people. We process every frame up to the bound; the display cadence only
        changes rendering, not tracking. The `finally` block releases the camera/file even after an error.

        Inline previews show the latest processed frame. They are not a real-time playback guarantee.
        We retain a small table of track centers, a separate event table, and one final annotated frame.
        """),
        code("""
        def run_tracking(source, max_frames=60, show_every=10, line_fraction=0.5):
            if max_frames < 1 or show_every < 1 or not 0 < line_fraction < 1:
                raise ValueError("Use positive frame/display bounds and line_fraction between 0 and 1.")
            import cv2
            detector = load_yolo(MODEL)  # New instance = new tracker state.
            person_ids = [k for k, name in detector.names.items() if name == "person"]
            if not person_ids:
                raise ValueError("This model has no 'person' class; choose your target class explicitly.")
            capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
            rows, counter, handle, last_rgb = [], None, None, None
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"Cannot open {source}. Check file/codec or Windows camera permission.")
                for frame_index in range(max_frames):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    height, width = frame.shape[:2]
                    if counter is None:
                        counter = LineCounter(line_x=width * line_fraction,
                                              margin=max(5, width * 0.015), max_gap=8)
                    prediction = detector.track(frame, persist=True, tracker="bytetrack.yaml",
                                                classes=person_ids, imgsz=320, conf=0.10,
                                                iou=0.70, device=DEVICE, verbose=False)[0]
                    boxes = prediction.boxes
                    if boxes is not None and boxes.id is not None:
                        for box, track_id in zip(boxes.xyxy.cpu().numpy(),
                                                  boxes.id.int().cpu().tolist()):
                            center_x = float((box[0] + box[2]) / 2)
                            center_y = float((box[1] + box[3]) / 2)
                            event = counter.update(track_id, center_x, frame_index)
                            rows.append({"frame": frame_index, "track_id": track_id,
                                         "center_x": center_x, "center_y": center_y,
                                         "crossing": event["direction"] if event else ""})
                    annotated = prediction.plot()
                    x = int(counter.line_x)
                    cv2.line(annotated, (x, 0), (x, height - 1), (0, 255, 255), 2)
                    caption = f"Frame {frame_index} | L->R {counter.totals['left_to_right']} | R->L {counter.totals['right_to_left']}"
                    cv2.putText(annotated, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 2)
                    last_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    if frame_index % show_every == 0 or frame_index == max_frames - 1:
                        preview = Image.fromarray(last_rgb)
                        preview.thumbnail((700, 500))
                        if handle is None:
                            handle = display(preview, display_id=True)
                        else:
                            handle.update(preview)
            finally:
                capture.release()
            track_table = pd.DataFrame(rows, columns=["frame", "track_id", "center_x", "center_y", "crossing"])
            events = pd.DataFrame(counter.events if counter else [],
                                  columns=["frame", "track_id", "direction", "x_center"])
            if last_rgb is None:
                raise RuntimeError("The source opened but returned no readable frames. Try a different local video/codec.")
            return track_table, events, last_rgb
        """),
        code("""
        tracks = pd.DataFrame()
        events = pd.DataFrame()
        if tracking_ready:
            tracks, events, last_frame = run_tracking(VIDEO_PATH, MAX_FRAMES, SHOW_EVERY,
                                                     line_fraction=LINE_FRACTION)
            tracks.to_csv(OUT / "track_centers.csv", index=False)
            events.to_csv(OUT / "crossing_events.csv", index=False)
            Image.fromarray(last_frame).save(OUT / "last_frame.png")
            print("Track observations (NOT visitors):", len(tracks))
            print("Distinct tracker IDs (NOT verified people):", tracks["track_id"].nunique())
            print("Line-crossing events:", len(events))
            display(events)
            print("Saved artifacts to", OUT)
        """, tags=["network"]),
        md("""
        ## 5. Inspect the temporal behavior

        A broken line or a new color can reveal track fragmentation. A plausible drawing alone
        does not validate ID accuracy. **Zero crossings is a valid result**, especially in a short
        artificial clip or smoke run; use the deterministic example to demonstrate an expected event.
        """),
        code("""
        if not tracks.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            for track_id, group in tracks.groupby("track_id"):
                ax.plot(group["frame"], group["center_x"], label=f"ID {track_id}")
            ax.axhline(last_frame.shape[1] * LINE_FRACTION, color="black", linestyle="--", label="count line")
            ax.set(xlabel="Frame", ylabel="Center x (pixels)", title="Observed tracker trajectories")
            ax.legend(ncol=3)
            plt.show()
        else:
            print("No tracked observations available for plotting.")
        """),
        md("""
        ## 6. Optional live camera — explicit opt-in

        Set `RUN_WEBCAM=True` only when you intend to open your camera. The loop is bounded and releases
        the device. The camera remains local; the preview is shown in the notebook. For a presentation,
        use a local recorded clip first so lighting, permissions, and hardware do not disrupt the lesson.
        The cell below does not save camera frames to disk.
        """),
        code("""
        RUN_WEBCAM = False
        WEBCAM_INDEX = 0
        WEBCAM_MAX_FRAMES = 100
        if RUN_WEBCAM and tracking_ready:
            webcam_tracks, webcam_events, webcam_last_frame = run_tracking(
                WEBCAM_INDEX, max_frames=WEBCAM_MAX_FRAMES, show_every=5)
            display(webcam_events)
        else:
            print("Camera not opened. Enable RUN_WEBCAM and the real-model section to try it.")
        """, tags=["network", "optional-webcam"]),
        md("""
        ## 7. Exercises and solutions

        1. Feed an ID that appears at x=80 first. Should it count as left-to-right? **No:** previous-side
           evidence is missing.
        2. Let the same ID cross right, left, then right again. Our declared policy records two events
           total, one per direction. To count every trip, remove the `counted` de-duplication and add
           a suitable cooldown; report the policy with the result.
        3. Increase the deadband and describe the tradeoff: fewer jitter events, but a short crossing
           may not reach the opposite stable side.
        4. Use a fixed-camera clip and manually label crossings. Compare event direction/time and
           missed events. Counting errors may arise from detector misses, ID switches, line placement,
           or the event policy.

        Run these edge cases to make the counting policy concrete:
        """),
        code("""
        policy_test = LineCounter(50, margin=5)
        for frame, x in enumerate([20, 80, 20, 80, 80]):
            policy_test.update(7, x, frame)
        assert policy_test.totals == {"left_to_right": 1, "right_to_left": 1}
        first_seen_right = LineCounter(50, margin=5)
        first_seen_right.update(9, 80, 0)
        assert len(first_seen_right.events) == 0
        gap_test = LineCounter(50, margin=5, max_gap=3)
        gap_test.update(2, 20, 0)
        gap_test.update(2, 80, 10)
        assert len(gap_test.events) == 0
        print("Verified: de-duplication, first observation, and long-gap policies.")
        """),
        md("""
        **Presenter checkpoint:** explain why `len(tracks)`, distinct IDs, and `len(events)` are different
        quantities. A tracker can assign new IDs after occlusion, so even de-duplicated ID counts can
        overcount physical objects. This short demo keeps state in memory; a long-running service needs
        state expiry, logs, and a tested policy for camera interruptions.

        Primary references: [Ultralytics tracking and persistent frames](https://docs.ultralytics.com/modes/track/),
        [ByteTrack implementation and paper](https://github.com/ifzhang/ByteTrack).
        **Next:** custom-data preparation and training before the advanced deployment lessons.
        """),
    ]
    write_notebook("05_Video_Tracking_and_Counting.ipynb", cells)


def build_export():
    cells = [
        md("""
        # 12 · Export to ONNX, check agreement, and benchmark fairly

        **Level:** advanced · **Presentation:** 35–50 minutes · **Prerequisites:** notebook 04 and
        the evaluation lesson. This notebook is independent of a custom-training checkpoint.

        A deployment artifact is useful only if it loads, produces sensible predictions, and meets
        a measured latency target. You will export established YOLO11n to ONNX, inspect its graph,
        compare **postprocessed detections**, and measure warm-call CPU latency.

        The export uses static batch 1, 320×320 input, FP32, and no graph simplification. Both paths use
        the Ultralytics wrapper for the same preprocessing and NMS. This is an application-pipeline
        comparison, **not a raw-runtime benchmark or a claim that ONNX is always faster**.

        Install optional dependencies with the course environment activated:

        ```powershell
        python -m pip install -r requirements-onnx.txt
        ```

        Then restart the Jupyter kernel. The preflight reports missing packages and skips the optional
        export coherently; it does not fabricate timing results. `COURSE_OFFLINE=1` also skips model work.
        """),
        bootstrap(),
        code("""
        import os
        import sys
        import json
        import shutil
        import platform
        import importlib.util
        import importlib.metadata
        import time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from PIL import Image
        from IPython.display import display
        from course_utils import load_yolo, get_sample_image

        OFFLINE = os.getenv("COURSE_OFFLINE", "0") == "1"
        SMOKE = os.getenv("COURSE_SMOKE", "0") == "1"
        IMG_SIZE = 320
        DEVICE = "cpu"  # Both backends use CPU for this comparison.
        WARMUP = 1 if SMOKE else 3
        REPEATS = 2 if SMOKE else 15
        EXPORT_DIR = ROOT / "models" / "exports"
        OUT = ROOT / "outputs" / "12_onnx"
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        required = ["torch", "ultralytics", "onnx", "onnxruntime", "onnxscript"]
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        READY = not OFFLINE and not missing
        pt_model = onnx_model = image = pt_result = onnx_result = None
        if OFFLINE:
            print("Offline mode: export and backend benchmarks skipped explicitly.")
        elif missing:
            print("Missing:", ", ".join(missing))
            print(f'Install into this kernel: "{sys.executable}" -m pip install -r "{ROOT / "requirements-onnx.txt"}"')
            print("Then restart the kernel and Run All.")
        else:
            print("Optional export dependencies found. Executing the real CPU export.")
        """),
        md("""
        ## 1. Export a fixed-shape graph

        We copy the checkpoint into `models/exports` before exporting because Ultralytics writes the
        `.onnx` beside its checkpoint. No custom training is required. `simplify=False` avoids adding
        a graph-simplifier dependency. `nms=False` keeps this YOLO11 graph's raw prediction head;
        the Python inference wrapper performs the NMS afterward.

        This notebook targets YOLO11's output contract. Other detector families, end-to-end heads,
        and exports with embedded NMS can have different shapes and postprocessing requirements.
        """),
        code("""
        if READY:
            import torch
            import onnx
            import onnxruntime as ort
            from ultralytics import YOLO

            loaded = load_yolo("yolo11n.pt")
            original_checkpoint = Path(loaded.ckpt_path)
            local_checkpoint = EXPORT_DIR / "yolo11n.pt"
            if original_checkpoint.resolve() != local_checkpoint.resolve():
                shutil.copy2(original_checkpoint, local_checkpoint)
            export_model = YOLO(str(local_checkpoint))
            onnx_path = Path(export_model.export(format="onnx", imgsz=IMG_SIZE, batch=1,
                                                 dynamic=False, simplify=False, nms=False,
                                                 device=DEVICE, opset=17))
            if not onnx_path.is_file():
                raise RuntimeError(f"Exporter did not produce the expected file: {onnx_path}")
            graph = onnx.load(str(onnx_path))
            onnx.checker.check_model(graph)
            print("Validated ONNX graph:", onnx_path)
            print(f"File size: {onnx_path.stat().st_size / 1024**2:.2f} MiB")
        """, tags=["network", "optional-onnx"]),
        md("""
        ## 2. Inspect the model contract

        The runtime input should be batch 1, three channels, height 320, width 320. Inspect the output
        names and shapes instead of assuming an arbitrary detector uses this same layout. With YOLO11
        detection and no embedded NMS, the head describes many candidate boxes and class scores;
        it is not a ready-to-draw table of final detections.
        """),
        code("""
        if READY:
            inspection_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            contract = []
            for kind, tensors in [("input", inspection_session.get_inputs()),
                                  ("output", inspection_session.get_outputs())]:
                for tensor in tensors:
                    contract.append({"kind": kind, "name": tensor.name,
                                     "shape": str(tensor.shape), "type": tensor.type})
            display(pd.DataFrame(contract))
            assert inspection_session.get_inputs()[0].shape == [1, 3, IMG_SIZE, IMG_SIZE]
            print("Providers:", inspection_session.get_providers())
            del inspection_session  # Release its thread pool before the wrapper benchmark.
        """),
        md("""
        ## 3. Make preprocessing and postprocessing match

        `rect=False` forces the PyTorch path to use square padding, matching the static ONNX graph.
        Without it, a rectangular source image can take a different letterbox path and produce
        misleading differences. Both paths receive the same PIL RGB image and identical confidence,
        IoU, and image-size settings. Ultralytics maps coordinates back to the original image.
        """),
        code("""
        prediction_settings = dict(imgsz=IMG_SIZE, conf=0.25, iou=0.70, rect=False,
                                   device=DEVICE, verbose=False)
        if READY:
            image = Image.open(get_sample_image()).convert("RGB")
            pt_model = YOLO(str(local_checkpoint))  # Fresh instance after export.
            onnx_model = YOLO(str(onnx_path), task="detect")
            pt_result = pt_model.predict(image, **prediction_settings)[0]
            onnx_result = onnx_model.predict(image, **prediction_settings)[0]
            fig, axes = plt.subplots(1, 2, figsize=(12, 7))
            for ax, title, prediction in zip(axes, ["PyTorch", "ONNX Runtime"], [pt_result, onnx_result]):
                boxes = prediction.boxes
                show_boxes(np.asarray(image), boxes.xyxy.cpu().numpy(),
                           labels=[prediction.names[int(c)] for c in boxes.cls.cpu().tolist()],
                           scores=boxes.conf.cpu().tolist(), ax=ax, title=title)
            plt.show()
        """, tags=["network", "optional-onnx"]),
        md("""
        ## 4. Match detections by class and IoU

        Sorting rows is insufficient because backend floating-point differences can change score order.
        Below, greedy one-to-one matching finds same-class boxes above IoU 0.5. We report unmatched
        predictions and numerical deltas. This single-image check is a **deployment sanity check**,
        not validation mAP; evaluate both backends on held-out labeled data before deployment.
        """),
        code("""
        def box_iou_one(a, b):
            left_top = np.maximum(a[:2], b[:2])
            right_bottom = np.minimum(a[2:], b[2:])
            intersection = np.maximum(right_bottom - left_top, 0).prod()
            area_a = np.maximum(a[2:] - a[:2], 0).prod()
            area_b = np.maximum(b[2:] - b[:2], 0).prod()
            return float(intersection / max(area_a + area_b - intersection, 1e-9))

        def compare_detections(reference, candidate, minimum_iou=0.5):
            a = reference.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
            b = candidate.boxes.data.cpu().numpy()
            pairs = sorted([(box_iou_one(x[:4], y[:4]), i, j)
                            for i, x in enumerate(a) for j, y in enumerate(b)
                            if int(x[5]) == int(y[5])], reverse=True)
            used_a, used_b, rows = set(), set(), []
            for overlap, i, j in pairs:
                if overlap < minimum_iou or i in used_a or j in used_b:
                    continue
                used_a.add(i)
                used_b.add(j)
                rows.append({"class_name": reference.names[int(a[i, 5])], "iou": overlap,
                             "max_coordinate_delta_px": float(np.abs(a[i, :4] - b[j, :4]).max()),
                             "confidence_delta": float(abs(a[i, 4] - b[j, 4]))})
            summary = {"pytorch_boxes": len(a), "onnx_boxes": len(b), "matched": len(rows),
                       "unmatched_pytorch": len(a) - len(used_a),
                       "unmatched_onnx": len(b) - len(used_b)}
            return pd.DataFrame(rows, columns=["class_name", "iou", "max_coordinate_delta_px", "confidence_delta"]), summary

        if READY:
            agreement, agreement_summary = compare_detections(pt_result, onnx_result)
            display(agreement.round(6))
            print(agreement_summary)
            agreement.to_csv(OUT / "prediction_agreement.csv", index=False)
            if agreement_summary["unmatched_pytorch"] or agreement_summary["unmatched_onnx"]:
                print("Investigate unmatched boxes, especially predictions near score/NMS thresholds.")
            elif agreement.empty:
                print("Both paths found no detections; this is inconclusive, so add positive examples.")
            else:
                print("All boxes matched on this image. Inspect coordinate and score deltas before accepting export.")
        """),
        md("""
        ## 5. Measure warm CPU latency and its variability

        We exclude model loading, export, image disk I/O, warmup, plotting, and JSON writing. Each timed
        call includes wrapper preprocessing, inference, and postprocessing. Alternate backend order
        across repetitions to reduce simple ordering bias. The CPU call is synchronous; a GPU benchmark
        would require explicit synchronization and a separate protocol.

        Median describes a typical call; p95 highlights slower calls. With 15 observations p95 is only
        a rough estimate; the two-repeat smoke mode verifies execution, **not performance**. Derived FPS
        assumes serial processing and excludes video decoding and display. Thread settings are reported,
        not forced equal: the result compares these default backend configurations on this machine.
        """),
        code("""
        if READY:
            backends = {"PyTorch": pt_model, "ONNX Runtime": onnx_model}
            for backend in backends.values():
                for _ in range(WARMUP):
                    backend.predict(image, **prediction_settings)
            timing_rows = []
            names = list(backends)
            for repeat in range(REPEATS):
                for name in (names if repeat % 2 == 0 else names[::-1]):
                    start = time.perf_counter()
                    backends[name].predict(image, **prediction_settings)
                    timing_rows.append({"backend": name, "repeat": repeat,
                                        "milliseconds": 1000 * (time.perf_counter() - start)})
            timings = pd.DataFrame(timing_rows)
            summary_rows = []
            for name, group in timings.groupby("backend", sort=False):
                values = group["milliseconds"].to_numpy()
                summary_rows.append({"backend": name, "calls": len(values),
                                     "median_ms": float(np.median(values)),
                                     "p95_ms": float(np.percentile(values, 95)),
                                     "serial_fps_from_median": float(1000 / np.median(values))})
            timing_summary = pd.DataFrame(summary_rows)
            display(timing_summary.round(2))
            fig, ax = plt.subplots(figsize=(8, 4))
            for name, group in timings.groupby("backend", sort=False):
                ax.plot(group["repeat"], group["milliseconds"], marker="o", label=name)
            ax.set(xlabel="Repetition", ylabel="Pipeline latency (ms)",
                   title="Warm CPU calls — loading and plotting excluded")
            ax.legend()
            plt.show()
            timings.to_csv(OUT / "timing_samples.csv", index=False)
            timing_summary.to_csv(OUT / "timing_summary.csv", index=False)
            environment = {"python": sys.version, "platform": platform.platform(),
                           "processor": platform.processor(), "cpu_count": os.cpu_count(),
                           "device": DEVICE, "torch_num_threads": torch.get_num_threads(),
                           "onnx_providers_available": ort.get_available_providers(),
                           "versions": {name: importlib.metadata.version(name) for name in required},
                           "settings": prediction_settings, "warmup": WARMUP, "repeats": REPEATS,
                           "image_size_original": list(image.size), "agreement": agreement_summary}
            (OUT / "benchmark_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
            print("Saved benchmark evidence to", OUT)
            if SMOKE:
                print("SMOKE RUN: insufficient repetitions for a performance conclusion.")
        else:
            print("No real benchmark produced; complete the preflight and rerun for measured results.")
        """, tags=["optional-onnx"]),
        md("""
        ## 6. Exercises, interpretation, and a deployment checklist

        1. Export again at 640. You must change `IMG_SIZE`, rerun export, and recreate **both** wrappers;
           a static 320 graph cannot accept arbitrary tensor sizes. Compare latency and held-out accuracy.
        2. Increase `REPEATS` to 50, close other heavy apps, and rerun the timing section. Explain changes
           in median versus p95. Keep the same power mode and document the CPU.
        3. Try a blank image and an image with many small objects. Why can NMS and detection counts
           change pipeline latency even when the network input tensor has fixed size?
        4. Replace the checkpoint with your custom `best.pt`. Confirm names, coordinate mapping,
           preprocessing, and held-out metrics before measuring speed.

        **Solution sketch:** 640 processes four times as many input pixels as 320, but whole-pipeline
        latency need not scale by exactly four. Equivalent output requires matched transforms and
        thresholds; fast output alone is not a successful export. Near-threshold boxes can differ across
        floating-point implementations even when most predictions agree closely.

        | Before shipping | Evidence to collect |
        |---|---|
        | Graph loads on the target machine | ONNX checker plus runtime inference |
        | Preprocessing/postprocessing agree | Class names, coordinate mapping, matched boxes |
        | Accuracy remains acceptable | Held-out labeled data, error review, and mAP |
        | Latency meets the target | Representative inputs, warm median/p95, full app timing |
        | Artifact is reproducible | Weights, versions, settings, and benchmark environment |

        **Troubleshooting:** missing ONNX packages → install `requirements-onnx.txt` in the selected
        kernel; Windows DLL errors → use matching 64-bit Python/package wheels and the required runtime;
        static-shape errors → match the exported input size; large disagreement → verify `rect=False`,
        RGB/BGR handling, class mapping, and NMS settings before relaxing tolerances.

        Primary references: [Ultralytics export](https://docs.ultralytics.com/modes/export/),
        [prediction](https://docs.ultralytics.com/modes/predict/),
        [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).
        """),
    ]
    write_notebook("12_ONNX_Export_and_Benchmarking.ipynb", cells)


if __name__ == "__main__":
    build_detection()
    build_tracking()
    build_export()
