"""Build the offline foundations and tiny-detector teaching notebooks."""
from notebook_factory import md, code, bootstrap, write_notebook


def image_basics():
    write_notebook("01_Image_Basics_and_Bounding_Boxes.ipynb", [
        md("""
        # 01 · Images and bounding boxes
        **Level:** beginner · **Time:** 35–45 minutes · **Runs offline on CPU**

        **You will:** inspect pixels, draw annotations, convert box formats, and keep boxes aligned when resizing or flipping an image.

        **Prerequisites:** complete notebook 00; understand variables, lists, and indexing. Run cells from top to bottom with **Shift+Enter**. Restart the kernel and run all cells to repeat the demonstration.

        **Teaching question:** a classifier answers “what is in this picture?” A detector also answers “where is each object?” Each object needs its own class label and bounding box.
        """), bootstrap(),
        md("""
        ## Step 1 — Create an image you can explain
        Images are arrays with height, width, and color channels. Our example uses RGB: red, green, blue. NumPy indexes a pixel as `image[y, x]`, even though boxes use `(x, y)` coordinates.

        We use **half-open boxes** `[x1, y1, x2, y2]`: the first corner is included and the last corner is excluded. Width is `x2 - x1`; there is no `+1`. The coordinate origin is at the top left; y grows downward.
        """),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (360, 240), (235, 240, 245))  # width, height
        draw = ImageDraw.Draw(image)
        draw.rectangle((35, 45, 134, 164), fill=(225, 70, 55))
        draw.ellipse((205, 65, 304, 164), fill=(40, 110, 225))
        boxes = np.array([[35, 45, 135, 165], [205, 65, 305, 165]], dtype=np.float32)
        names = ["rectangle", "circle"]
        pixels = np.asarray(image)
        print("Array shape (height, width, channels):", pixels.shape)
        print("Storage:", pixels.dtype, "| red object pixel:", pixels[80, 60])
        show_boxes(image, boxes, labels=names, title="Two objects, two annotations")
        plt.show()
        """),
        md("""
        **Observe:** the circle's bounding box includes background corners. A bounding box describes a rectangle around an object; a segmentation mask would describe the object's individual pixels.

        **Presenter prompt:** point at the red object and ask which of `pixels[80, 60]` and `pixels[60, 80]` means x=60, y=80.
        """),
        md("""
        ## Step 2 — Inspect channels and crop an object
        Pixel values are integers from 0 to 255. Neural networks commonly receive floating-point values after preprocessing; the precise transform belongs to the model. Cropping follows `[y1:y2, x1:x2]`.
        """),
        code("""
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for channel, name in enumerate(["Red", "Green", "Blue"]):
            axes[channel].imshow(pixels[:, :, channel], cmap="gray", vmin=0, vmax=255)
            axes[channel].set_title(name + " intensity")
        x1, y1, x2, y2 = boxes[0].astype(int)
        axes[3].imshow(pixels[y1:y2, x1:x2])
        axes[3].set_title("Rectangle crop")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        float_image = pixels.astype(np.float32) / 255.0
        print("Float range:", float_image.min(), "to", float_image.max())
        """),
        md("""
        ## Step 3 — Convert corner boxes to normalized center boxes
        A common YOLO annotation row is `class_id center_x center_y width height`, with all four geometry values divided by the image width or height. IDs start at zero. This is a **label format**, not a prediction confidence score.

        A normalized x value of 0.5 is halfway across **this image**. Keep image dimensions with your annotations.
        """),
        code("""
        def xyxy_to_normalized_xywh(boxes, width, height):
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            sizes = boxes[:, 2:] - boxes[:, :2]
            return np.concatenate([centers, sizes], axis=1) / [width, height, width, height]

        def normalized_xywh_to_xyxy(rows, width, height):
            rows = np.asarray(rows, dtype=np.float32).reshape(-1, 4)
            scaled = rows * [width, height, width, height]
            centers, sizes = scaled[:, :2], scaled[:, 2:]
            return np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)

        width, height = image.size
        normalized = xyxy_to_normalized_xywh(boxes, width, height)
        for class_id, row in enumerate(normalized):
            print(class_id, " ".join(f"{value:.6f}" for value in row))
        restored = normalized_xywh_to_xyxy(normalized, width, height)
        assert np.allclose(boxes, restored, atol=1e-4)
        print("Round-trip conversion preserved the boxes.")
        """),
        md("""
        ## Step 4 — Resize the picture and its annotations together
        Here we deliberately stretch the image to a square. Multiply x coordinates by `new_width / old_width` and y coordinates by `new_height / old_height`. Many real detectors preserve aspect ratio with padding (“letterboxing”); then you must account for both scale and padding.
        """),
        code("""
        new_width, new_height = 256, 256
        resized = image.resize((new_width, new_height))
        resized_boxes = boxes * [new_width / width, new_height / height,
                                new_width / width, new_height / height]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        show_boxes(resized, boxes, names, ax=axes[0], title="Mistake: old pixel coordinates")
        show_boxes(resized, resized_boxes, names, ax=axes[1], title="Correct: scaled coordinates")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## Step 5 — Flip horizontally without losing the objects
        A horizontal flip maps a half-open box to `[W-x2, y1, W-x1, y2]`. Read from the original box while constructing the new one; in-place edits can accidentally reuse a changed coordinate.
        """),
        code("""
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped_boxes = boxes.copy()
        flipped_boxes[:, 0] = width - boxes[:, 2]
        flipped_boxes[:, 2] = width - boxes[:, 0]
        show_boxes(flipped, flipped_boxes, names, title="Horizontal flip with correct labels")
        plt.show()
        assert np.allclose(flipped_boxes[:, 2] - flipped_boxes[:, 0], boxes[:, 2] - boxes[:, 0])
        """),
        md("""
        ## Try it yourself
        1. Add a third object with a valid box; display all three labels.
        2. Print the rectangle's area and the fraction of the image it occupies.
        3. Derive the box transform for a **vertical** flip. Plot it before opening the solution.

        **Checkpoint:** explain why changing an image without changing its annotation silently damages training data.
        """),
        md("""
        ## Solutions — run after attempting the exercises
        The third object below is a triangle. We draw its extreme pixels inside its half-open bounding box.
        """),
        code("""
        extra_image = image.copy()
        ImageDraw.Draw(extra_image).polygon([(155, 185), (185, 185), (170, 214)], fill=(35, 160, 90))
        extra_boxes = np.vstack([boxes, [155, 185, 186, 215]])
        show_boxes(extra_image, extra_boxes, names + ["triangle"])
        plt.show()
        area = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])
        print("Rectangle box area:", area, "pixels squared; fraction:", area / (width * height))
        vertical_boxes = boxes.copy()
        vertical_boxes[:, 1] = height - boxes[:, 3]
        vertical_boxes[:, 3] = height - boxes[:, 1]
        show_boxes(image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), vertical_boxes, names)
        plt.show()
        """, tags=["solution"]),
        md("""
        ## Recap and next step
        You can now interpret RGB arrays, crop objects, convert annotations, and transform boxes with an image. Inspect annotations visually every time you change preprocessing.

        **Next:** [02 · Classical object detection](02_Classical_Object_Detection.ipynb).

        **Reference:** [Pillow image operations](https://pillow.readthedocs.io/en/stable/reference/Image.html).
        """),
    ])


def classical():
    write_notebook("02_Classical_Object_Detection.ipynb", [
        md("""
        # 02 · Build a detector without machine learning
        **Level:** beginner · **Time:** 35–45 minutes · **Runs offline on CPU**

        **You will:** turn pixels into a foreground mask, find connected outlines, filter noise, produce labeled boxes, and deliberately break a detector's assumptions.

        **Prerequisites:** notebook 01; NumPy, Pillow, Matplotlib, and OpenCV from the project requirements. This is a rule-based detector for controlled bright shapes on a dark background. It does not recognize everyday objects.
        """), bootstrap(),
        md("""
        ## Step 1 — Make a controlled scene
        Start with simple conditions so each step can be explained. The three shapes are separated and brighter than the background. The tiny bright dots simulate nuisance detections.
        """),
        code("""
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw

        scene = Image.new("RGB", (420, 220), (25, 30, 35))
        painter = ImageDraw.Draw(scene)
        painter.rectangle((25, 55, 114, 159), fill=(235, 200, 80))
        painter.ellipse((155, 60, 254, 159), fill=(90, 220, 220))
        painter.polygon([(340, 40), (285, 165), (395, 165)], fill=(240, 150, 170))
        for x, y in [(20, 20), (130, 180), (275, 25), (400, 190)]:
            painter.ellipse((x, y, x + 3, y + 3), fill="white")
        plt.figure(figsize=(10, 4))
        plt.imshow(scene)
        plt.title("Objects plus small bright distractors")
        plt.axis("off")
        plt.show()
        """),
        md("""
        ## Step 2 — Separate foreground from background
        Convert RGB to grayscale, then mark pixels above a threshold as foreground. Our images come from Pillow and are RGB; OpenCV `imread` would normally return BGR. Choosing the wrong conversion mixes up channel meaning.

        **Presenter prompt:** before running, predict whether increasing the threshold will retain more pixels or fewer.
        """),
        code("""
        rgb = np.asarray(scene)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        THRESHOLD = 100
        _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Grayscale intensity")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"Foreground: intensity > {THRESHOLD}")
        for ax in axes:
            ax.axis("off")
        plt.show()
        print("Foreground pixels:", int((mask > 0).sum()))
        """),
        md("""
        ## Step 3 — Find outlines and reject tiny regions
        `findContours` follows boundaries of foreground regions. `RETR_EXTERNAL` returns only the outer outlines, which is appropriate for these solid shapes. A minimum area removes our small dots. Area filtering can also remove real small objects: a useful rule has a tradeoff.
        """),
        code("""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_AREA = 200
        kept_contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
        print("Before area filtering:", len(contours), "| after:", len(kept_contours))
        print("Retained contour areas:", [round(cv2.contourArea(c), 1) for c in kept_contours])
        """),
        md("""
        ## Step 4 — Turn geometric rules into predictions
        Approximate each outline with fewer vertices. Three corners suggest a triangle; four suggest a rectangle; other rounded contours are called circles **in this restricted dataset**. A four-corner rule also accepts trapezoids, and an ellipse can look like a circle to this rule.

        `boundingRect` returns `(x, y, width, height)`; convert to half-open corner coordinates `(x, y, x+width, y+height)`. There is no learned confidence score here.
        """),
        code("""
        def classify_contour(contour):
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            vertices = len(polygon)
            return "triangle" if vertices == 3 else "rectangle" if vertices == 4 else "circle-like"

        def detect_bright_shapes(pil_image, threshold=100, min_area=200):
            gray_image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            outlines, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes, labels = [], []
            for contour in outlines:
                if cv2.contourArea(contour) < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x + w, y + h])
                labels.append(classify_contour(contour))
            return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), labels, binary

        predicted_boxes, predicted_names, _ = detect_bright_shapes(scene)
        show_boxes(scene, predicted_boxes, predicted_names, title="A complete rule-based detection pipeline")
        plt.show()
        print(list(zip(predicted_names, predicted_boxes.astype(int).tolist())))
        """),
        md("""
        ## Step 5 — Break the assumptions, one change at a time
        Dim objects fall below the intensity threshold. Touching bright objects become one connected region. These failures illustrate why background contrast and separation are assumptions, not guaranteed properties of images.
        """),
        code("""
        dim_scene = Image.fromarray((np.asarray(scene).astype(float) * 0.40).astype(np.uint8))
        touching_scene = Image.new("RGB", scene.size, (25, 30, 35))
        touch_draw = ImageDraw.Draw(touching_scene)
        touch_draw.rectangle((70, 50, 180, 165), fill=(220, 220, 100))
        touch_draw.ellipse((155, 60, 265, 170), fill=(100, 220, 220))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, picture, title in zip(axes, [scene, dim_scene, touching_scene],
                                      ["Original", "Dim lighting", "Two touching objects"]):
            found, labels, _ = detect_bright_shapes(picture)
            show_boxes(picture, found, labels, ax=ax, title=f"{title}: {len(found)} detections")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        **Observe:** changing a threshold can recover dim foreground, but it cannot split two objects that share one connected mask. A successful demonstration should include a failure case and explain its cause.

        ## Try it yourself
        1. Sweep thresholds 30, 60, 100, 180, 240; record the number of detected shapes in the original scene.
        2. Repair the dim scene by changing one parameter. Does the same setting work on a bright background?
        3. Set `min_area=0`. Which extra boxes appear, and why?
        """),
        md("""
        ## Solutions — reveal after trying
        Counts depend on grayscale intensities and region areas. A count of three alone does not prove that the three boxes are correct: always inspect them.
        """),
        code("""
        for value in [30, 60, 100, 180, 240]:
            found, labels, _ = detect_bright_shapes(scene, threshold=value)
            print(f"threshold={value:3d}: {len(found)} detections", labels)
        repaired, labels, _ = detect_bright_shapes(dim_scene, threshold=50)
        show_boxes(dim_scene, repaired, labels, title="Lower threshold recovers dim shapes")
        plt.show()
        noisy_boxes, noisy_labels, _ = detect_bright_shapes(scene, min_area=0)
        show_boxes(scene, noisy_boxes, noisy_labels, title="Removing the area filter keeps small dots")
        plt.show()
        """, tags=["solution"]),
        md("""
        ## Recap and next step
        You built an image → mask → contours → rules → labeled boxes pipeline. Its assumptions are easy to explain and easy to violate. Learned detectors replace many manual rules with features learned from varied labeled examples.

        **Next:** [03 · IoU and non-maximum suppression](03_IoU_and_Non_Maximum_Suppression.ipynb).

        **References:** [OpenCV contours tutorial](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html), [contour features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html).
        """),
    ])


def iou_nms():
    write_notebook("03_IoU_and_Non_Maximum_Suppression.ipynb", [
        md("""
        # 03 · Measure overlap and remove duplicate predictions
        **Level:** beginner → intermediate · **Time:** 40–50 minutes · **Runs offline on CPU**

        **You will:** calculate intersection over union (IoU), implement greedy non-maximum suppression (NMS), compare class-aware and class-agnostic NMS, and check edge cases.

        **Prerequisites:** notebook 01, NumPy arrays and simple loops. Predictions in this notebook are deliberately hand-crafted so you can inspect every decision.
        """), bootstrap(),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw
        """),
        md("""
        ## Step 1 — Compute intersection over union
        **IoU = intersection area / union area.** It measures overlap between two boxes, from 0 (no overlap) to 1 (identical non-empty boxes). It does not know the object class or prediction confidence.

        We use half-open/continuous `xyxy` coordinates, so area is `(x2-x1)*(y2-y1)` without `+1`. The implementation below returns zero for zero-area boxes and rejects inverted or non-finite coordinates.
        """),
        code("""
        def valid_boxes(values):
            values = np.asarray(values, dtype=np.float64)
            if values.size == 0:
                return np.empty((0, 4), dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 4:
                raise ValueError("Expected boxes with shape (N, 4).")
            if not np.isfinite(values).all() or np.any(values[:, 2:] < values[:, :2]):
                raise ValueError("Boxes must be finite, with x2 >= x1 and y2 >= y1.")
            return values

        def pairwise_iou(boxes_a, boxes_b):
            a, b = valid_boxes(boxes_a), valid_boxes(boxes_b)
            top_left = np.maximum(a[:, None, :2], b[None, :, :2])
            bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
            intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
            area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
            area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
            union = area_a[:, None] + area_b[None, :] - intersection
            return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

        box_a = [[30, 30, 130, 130]]
        box_b = [[80, 60, 180, 160]]
        print("Intersection = 50 × 70 = 3500")
        print("Union = 10000 + 10000 - 3500 = 16500")
        print("IoU:", pairwise_iou(box_a, box_b)[0, 0])
        canvas = Image.new("RGB", (230, 200), "white")
        show_boxes(canvas, box_a + box_b, ["A", "B"], title="Overlapping boxes")
        plt.show()
        """),
        md("""
        ## Step 2 — Build trust with edge cases
        Adjacent boxes that only share a boundary have zero intersection area. Empty input should produce an empty matrix, not a crash. Degenerate annotations have no useful area; training datasets should normally reject them even though this geometry function handles them.
        """),
        code("""
        assert np.isclose(pairwise_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0], 1)
        assert pairwise_iou([[0, 0, 10, 10]], [[10, 0, 20, 10]])[0, 0] == 0
        assert pairwise_iou([[0, 0, 0, 10]], [[0, 0, 0, 10]])[0, 0] == 0
        assert pairwise_iou([], [[0, 0, 10, 10]]).shape == (0, 1)
        try:
            pairwise_iou([[10, 0, 5, 10]], box_a)
        except ValueError as error:
            print("Inverted coordinates caught:", error)
        print("Geometry checks passed.")
        """),
        md("""
        ## Step 3 — Inspect duplicate predictions
        Imagine a detector has proposed several boxes around the same object. Each has a class and score. Filtering low scores removes weak candidates; NMS then removes overlapping duplicates. These are separate thresholds with separate purposes.

        Here the rectangle and circle genuinely overlap. That makes the difference between class-aware and class-agnostic suppression visible.
        """),
        code("""
        scene = Image.new("RGB", (340, 240), (240, 240, 240))
        painter = ImageDraw.Draw(scene)
        painter.rectangle((40, 45, 149, 174), fill=(230, 120, 100))
        painter.ellipse((65, 65, 164, 164), fill=(90, 155, 235))
        painter.rectangle((235, 75, 304, 164), fill=(230, 120, 100))
        candidate_boxes = np.array([
            [40, 45, 150, 175], [44, 48, 151, 176], [33, 40, 145, 170],
            [65, 65, 165, 165], [69, 68, 168, 168], [235, 75, 305, 165],
            [170, 15, 215, 50]], dtype=float)
        candidate_scores = np.array([0.95, 0.86, 0.65, 0.92, 0.80, 0.88, 0.18])
        candidate_classes = np.array([0, 0, 0, 1, 1, 0, 1])
        class_names = ["rectangle", "circle"]
        candidate_labels = [f"{i}: {class_names[c]}" for i, c in enumerate(candidate_classes)]
        show_boxes(scene, candidate_boxes, candidate_labels, candidate_scores, title="Seven candidate predictions")
        plt.show()
        """),
        md("""
        ## Step 4 — Implement greedy NMS
        1. Discard scores below the confidence threshold and boxes with zero area.
        2. Select the remaining box with the highest score.
        3. Suppress remaining boxes whose IoU with it exceeds the NMS threshold.
        4. Repeat until no candidates remain.

        **Class-aware NMS** only suppresses boxes of the same class. **Class-agnostic NMS** compares all classes. Neither can guarantee that overlapping objects of the same class survive. This implementation breaks score ties by original index, making the classroom result repeatable.
        """),
        code("""
        def nms(boxes, scores, classes, iou_threshold=0.5, confidence_threshold=0.25,
                class_aware=True):
            boxes = valid_boxes(boxes)
            scores = np.asarray(scores, dtype=float)
            classes = np.asarray(classes)
            if scores.shape != (len(boxes),) or classes.shape != (len(boxes),):
                raise ValueError("Provide one score and one class per box.")
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Scores must be finite values in [0, 1].")
            if not 0 <= iou_threshold <= 1 or not 0 <= confidence_threshold <= 1:
                raise ValueError("Thresholds must be in [0, 1].")
            positive_area = np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
            candidates = np.flatnonzero((scores >= confidence_threshold) & positive_area)
            order = candidates[np.argsort(-scores[candidates], kind="stable")]
            keep = []
            while len(order):
                best = order[0]
                keep.append(best)
                remaining = order[1:]
                overlaps = pairwise_iou(boxes[best:best + 1], boxes[remaining])[0]
                comparable = classes[remaining] == classes[best] if class_aware else np.ones(len(remaining), dtype=bool)
                order = remaining[~((overlaps > iou_threshold) & comparable)]
            return np.asarray(keep, dtype=int)

        aware = nms(candidate_boxes, candidate_scores, candidate_classes)
        agnostic = nms(candidate_boxes, candidate_scores, candidate_classes, class_aware=False)
        print("Class-aware kept indices:", aware)
        print("Class-agnostic kept indices:", agnostic)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, kept, title in zip(axes, [aware, agnostic], ["Class-aware NMS", "Class-agnostic NMS"]):
            show_boxes(scene, candidate_boxes[kept], [candidate_labels[i] for i in kept],
                       candidate_scores[kept], ax=ax, title=title)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        **Observe:** class-aware NMS can keep both overlapping classes. A lower **NMS IoU threshold** suppresses more aggressively. A higher **confidence threshold** removes more low-score predictions. Neither threshold is the IoU threshold used to declare a prediction correct during evaluation.

        ## Step 5 — Check NMS edge cases
        The same class and identical geometry should collapse to one prediction. Different classes survive in class-aware mode. Our stable ordering means the first box wins an exact score tie.
        """),
        code("""
        identical = [[0, 0, 10, 10], [0, 0, 10, 10]]
        assert nms([], [], []).size == 0
        assert nms(identical, [0.9, 0.9], [0, 0]).tolist() == [0]
        assert nms(identical, [0.9, 0.8], [0, 1]).tolist() == [0, 1]
        assert nms(identical, [0.9, 0.8], [0, 1], class_aware=False).tolist() == [0]
        assert nms([[0, 0, 0, 10]], [0.9], [0]).size == 0
        print("NMS checks passed.")
        """),
        md("""
        ## Try it yourself
        1. Sweep NMS IoU thresholds 0.1, 0.5, 0.9, 1.0. Which candidate indices survive?
        2. Keep the NMS threshold fixed at 0.5 and sweep confidence thresholds 0.1, 0.5, 0.9.
        3. Give the overlapping circle the same class as the rectangle. Explain why a real object may disappear.
        """),
        md("""
        ## Solutions — compare your predictions with the code
        At an NMS threshold of 1.0 no candidate is suppressed by overlap, because our rule is strictly `IoU > threshold`. Confidence and area filtering still apply.
        """),
        code("""
        for threshold in [0.1, 0.5, 0.9, 1.0]:
            print("NMS IoU", threshold, "->", nms(candidate_boxes, candidate_scores, candidate_classes,
                                                 iou_threshold=threshold).tolist())
        for threshold in [0.1, 0.5, 0.9]:
            print("Confidence", threshold, "->", nms(candidate_boxes, candidate_scores, candidate_classes,
                                                    confidence_threshold=threshold).tolist())
        all_same_class = np.zeros_like(candidate_classes)
        print("All one class ->", nms(candidate_boxes, candidate_scores, all_same_class).tolist())
        """, tags=["solution"]),
        md("""
        ## Recap and next step
        IoU is geometry; confidence is a model score; NMS is postprocessing. Keeping these concepts separate helps explain detector outputs and tune a demonstration without confusing training and evaluation.

        **Next:** [04 · Pretrained YOLO detection](04_Pretrained_YOLO_Detection.ipynb).

        **Reference:** [Torchvision NMS API](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html) describes the standard greedy overlap rule. Our small NumPy implementation adds explicit validation and class-aware filtering for teaching.
        """),
    ])


def tiny_detector():
    write_notebook("10_Build_a_Tiny_Detector_in_PyTorch.ipynb", [
        md("""
        # 10 · Build and train a tiny detector from scratch
        **Level:** advanced foundation · **Time:** 60–90 minutes including discussion · **Runs offline on CPU**

        **You will:** generate independent datasets, build a convolutional network with two prediction heads, combine classification and localization losses, select a model using validation data, and evaluate held-out predictions.

        **Prerequisites:** notebooks 01 and 03, familiarity with Python functions/classes, basic PyTorch tensors, and the project PyTorch installation.

        **Scope:** every image contains **exactly one** rectangle, circle, or triangle. The network predicts one class and one box. It has no background/objectness output, cannot represent multiple objects, and is not a replacement for YOLO or Faster R-CNN. This controlled problem exposes the learning mechanics of a detector.

        Default training uses small 48×48 images and a short CPU run; duration depends on your laptop. `COURSE_SMOKE=1` uses a very small run for code verification, not model-quality claims.
        """), bootstrap(),
        code("""
        import os
        import copy
        import time
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset, DataLoader

        seed_everything(42)
        torch.manual_seed(42)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        DEVICE = torch.device("cpu")  # predictable, no GPU required
        SMOKE = os.environ.get("COURSE_SMOKE", "0") == "1"
        SIZE = 48
        CLASS_NAMES = ["rectangle", "circle", "triangle"]
        TRAIN_N, VAL_N, TEST_N = (96, 30, 30) if SMOKE else (1200, 240, 240)
        EPOCHS = 2 if SMOKE else 18
        print("PyTorch:", torch.__version__, "| device:", DEVICE, "| epochs:", EPOCHS)
        """),
        md("""
        ## Step 1 — Generate one labeled shape per image
        Each box is normalized to `[0, 1]`. Shape color is sampled independently of class, so the model cannot solve classification by a fixed class-to-color lookup. Position, dimensions, color, and background vary.

        Train, validation, and test use different random seeds. Validation is used to select a checkpoint. Test stays out of training and checkpoint selection. Synthetic held-out results describe this generator's distribution, not photographs or real-world performance.
        """),
        code("""
        def build_dataset(count, seed, shift=False):
            rng = np.random.default_rng(seed)
            images, classes, boxes = [], [], []
            for index in range(count):
                class_id = index % 3  # balanced classes; batches are shuffled later
                w, h = rng.integers(14, 29, size=2)
                if class_id == 1:
                    h = w  # equal axes keep the circle class geometrically a circle
                x1 = int(rng.integers(2, SIZE - w - 1))
                y1 = int(rng.integers(2, SIZE - h - 1))
                x2, y2 = x1 + int(w), y1 + int(h)
                background = int(rng.integers(5, 40) if not shift else rng.integers(170, 210))
                picture = Image.new("RGB", (SIZE, SIZE), (background,) * 3)
                painter = ImageDraw.Draw(picture)
                color = tuple(int(c) for c in rng.integers(100, 245, size=3))
                if class_id == 0:
                    painter.rectangle((x1, y1, x2 - 1, y2 - 1), fill=color)
                elif class_id == 1:
                    painter.ellipse((x1, y1, x2 - 1, y2 - 1), fill=color)
                else:
                    painter.polygon([(x1, y2 - 1), (x2 - 1, y2 - 1),
                                     ((x1 + x2 - 1) // 2, y1)], fill=color)
                array = np.asarray(picture).astype(np.float32)
                array = np.clip(array + rng.normal(0, 2, array.shape), 0, 255) / 255.0
                images.append(array.transpose(2, 0, 1))  # HWC -> CHW
                classes.append(class_id)
                boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32) / SIZE)
            return TensorDataset(torch.tensor(np.stack(images), dtype=torch.float32),
                                 torch.tensor(classes, dtype=torch.long),
                                 torch.tensor(np.stack(boxes), dtype=torch.float32))

        train_data = build_dataset(TRAIN_N, seed=100)
        val_data = build_dataset(VAL_N, seed=200)
        test_data = build_dataset(TEST_N, seed=300)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)
        print("Independent samples:", len(train_data), "train /", len(val_data), "validation /", len(test_data), "test")
        """),
        code("""
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        for index, ax in enumerate(axes.flat):
            image_tensor, class_id, box = train_data[index]
            show_boxes(image_tensor.permute(1, 2, 0).numpy(), [box.numpy() * SIZE],
                       [CLASS_NAMES[class_id.item()]], ax=ax, title=f"Training sample {index}")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## Step 2 — Build a shared backbone and two heads
        Convolutions learn local patterns. The flattened spatial feature map retains position information needed for localization. A shared feature vector feeds a three-class head and a four-number box head.

        The class head returns **logits**, not probabilities; `CrossEntropyLoss` expects logits. The box head transforms its values into valid normalized corners: width/height are in `(0,1)` and the top-left corner is constrained so the whole box remains inside the image.
        """),
        code("""
        class TinySingleObjectDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 12, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(12, 24, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Flatten(), nn.Linear(24 * 12 * 12, 128), nn.ReLU())
                self.class_head = nn.Linear(128, 3)
                self.box_head = nn.Linear(128, 4)

            def forward(self, images):
                features = self.features(images)
                logits = self.class_head(features)
                raw = self.box_head(features).sigmoid()
                sizes = raw[:, 2:]
                top_left = raw[:, :2] * (1 - sizes)
                boxes = torch.cat([top_left, top_left + sizes], dim=1)
                return logits, boxes

        model = TinySingleObjectDetector().to(DEVICE)
        demo_logits, demo_boxes = model(train_data.tensors[0][:4].to(DEVICE))
        print("Class logits:", tuple(demo_logits.shape), "| boxes:", tuple(demo_boxes.shape))
        print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
        assert torch.all(demo_boxes[:, 2:] > demo_boxes[:, :2])
        """),
        md("""
        ## Step 3 — Define losses and useful metrics
        We minimize `classification loss + 10 × box loss`. The weight balances different numerical scales; it is a tunable choice, not a universal constant. Smooth L1 penalizes coordinate errors. IoU assesses overlap.

        We report class accuracy, mean IoU, and the fraction of images with **both correct class and IoU ≥ 0.5**. That last measure is a simple one-object success rate, **not mAP**. Multibox detector evaluation requires confidence ranking and matching predictions to multiple ground-truth objects.
        """),
        code("""
        class_loss_fn = nn.CrossEntropyLoss()
        box_loss_fn = nn.SmoothL1Loss(beta=0.1)
        BOX_WEIGHT = 10.0

        def aligned_iou(predicted, target):
            top_left = torch.maximum(predicted[:, :2], target[:, :2])
            bottom_right = torch.minimum(predicted[:, 2:], target[:, 2:])
            intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
            pred_area = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(dim=1)
            true_area = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(dim=1)
            return intersection / (pred_area + true_area - intersection).clamp(min=1e-8)

        @torch.no_grad()
        def evaluate(current_model, loader):
            current_model.eval()
            total, loss_sum, correct_sum, iou_sum, success_sum = 0, 0.0, 0, 0.0, 0
            for images, labels, target_boxes in loader:
                images, labels, target_boxes = images.to(DEVICE), labels.to(DEVICE), target_boxes.to(DEVICE)
                logits, predicted_boxes = current_model(images)
                loss = class_loss_fn(logits, labels) + BOX_WEIGHT * box_loss_fn(predicted_boxes, target_boxes)
                ious = aligned_iou(predicted_boxes, target_boxes)
                correct = logits.argmax(dim=1) == labels
                total += len(images)
                loss_sum += loss.item() * len(images)
                correct_sum += correct.sum().item()
                iou_sum += ious.sum().item()
                success_sum += (correct & (ious >= 0.5)).sum().item()
            return {"loss": loss_sum / total, "class_accuracy": correct_sum / total,
                    "mean_iou": iou_sum / total, "joint_success": success_sum / total}

        print("Untrained validation baseline:", evaluate(model, val_loader))
        """),
        md("""
        ## Step 4 — Train, validate, and retain the best validation checkpoint
        Each batch follows: zero gradients → forward pass → loss → backpropagation → optimizer update. Validation uses evaluation mode and disables gradient tracking. Selecting by validation loss keeps the test set out of model selection.

        **Presenter prompt:** pause after the first epoch. Ask whether lower training loss alone proves that the model generalizes. Inspect validation results before answering.
        """),
        code("""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = []
        best_loss, best_epoch, best_state = float("inf"), 0, None
        start = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            training_sum = 0.0
            for images, labels, target_boxes in train_loader:
                images, labels, target_boxes = images.to(DEVICE), labels.to(DEVICE), target_boxes.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits, predicted_boxes = model(images)
                loss = class_loss_fn(logits, labels) + BOX_WEIGHT * box_loss_fn(predicted_boxes, target_boxes)
                loss.backward()
                optimizer.step()
                training_sum += loss.item() * len(images)
            validation = evaluate(model, val_loader)
            history.append({"epoch": epoch, "train_loss": training_sum / len(train_data), **validation})
            if validation["loss"] < best_loss:
                best_loss, best_epoch = validation["loss"], epoch
                best_state = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch:02d}: train={history[-1]['train_loss']:.3f} "
                  f"val={validation['loss']:.3f} class_acc={validation['class_accuracy']:.3f} "
                  f"IoU={validation['mean_iou']:.3f} joint={validation['joint_success']:.3f}")
        model.load_state_dict(best_state)
        print(f"Restored epoch {best_epoch}; training took {time.perf_counter() - start:.1f} seconds.")
        """, tags=["training"]),
        code("""
        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
        axes[0].plot(epochs, [row["loss"] for row in history], label="Validation")
        axes[0].set(xlabel="Epoch", ylabel="Combined loss", title="Learning curves")
        axes[0].legend()
        for metric in ["class_accuracy", "mean_iou", "joint_success"]:
            axes[1].plot(epochs, [row[metric] for row in history], label=metric)
        axes[1].set(xlabel="Epoch", ylabel="Validation metric", ylim=(0, 1.05))
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """),
        md("""
        **Expected observation:** a complete default run should usually improve over the untrained validation baseline. The precise curve depends on numerical libraries and training randomness. Poor results are evidence to investigate: inspect labels, images, losses, and validation curves. A two-epoch smoke run can remain inaccurate and still verify that the code executes.

        ## Step 5 — Evaluate once on the held-out test set
        This is the performance estimate for the chosen checkpoint. If you repeatedly tune after seeing these results, this set becomes another validation set; reserve a fresh final test set before making a final claim.
        """),
        code("""
        test_metrics = evaluate(model, test_loader)
        print("Held-out test results:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")
        print("These are synthetic one-object metrics, not real-world mAP.")
        """),
        md("""
        ## Step 6 — Inspect examples, including localization failures
        Draw green ground-truth boxes and red predicted boxes. A softmax score expresses relative preference among our three classes; this model always chooses one even when shown an empty or unrelated image. It is not an object-presence probability.
        """),
        code("""
        from matplotlib.patches import Rectangle
        model.eval()
        with torch.no_grad():
            test_logits, test_boxes = model(test_data.tensors[0].to(DEVICE))
            probabilities = test_logits.softmax(dim=1).cpu()
            test_boxes = test_boxes.cpu()
            ious = aligned_iou(test_boxes, test_data.tensors[2])
        worst = torch.argsort(ious)[:3].tolist()
        examples = [0, 1, 2] + worst
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, index in zip(axes.flat, examples):
            picture, true_class, true_box = test_data[index]
            ax.imshow(picture.permute(1, 2, 0).numpy())
            for box, color in [(true_box, "lime"), (test_boxes[index], "red")]:
                x1, y1, x2, y2 = (box.numpy() * SIZE).tolist()
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
            score, predicted_class = probabilities[index].max(dim=0)
            ax.set_title(f"True: {CLASS_NAMES[true_class.item()]} | Pred: {CLASS_NAMES[predicted_class.item()]}\\n"
                         f"score={score.item():.2f}, IoU={ious[index].item():.2f}")
            ax.axis("off")
        plt.suptitle("Top: first examples · bottom: lowest IoU examples · green=true, red=prediction")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## Step 7 — Test a distribution change and save the trained weights
        A bright background was absent from training. Keep the shapes and task the same, then change only this background distribution. Compare the metrics; do not assume performance transfers.

        The saved file contains weights and small configuration values. It is written to `outputs/tiny_detector.pt`; rerunning this cell replaces the course checkpoint.
        """),
        code("""
        shifted_data = build_dataset(TEST_N, seed=400, shift=True)
        shifted_loader = DataLoader(shifted_data, batch_size=128, shuffle=False, num_workers=0)
        shifted_metrics = evaluate(model, shifted_loader)
        print("Original test:", test_metrics)
        print("Bright-background shift:", shifted_metrics)
        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        checkpoint_path = output_dir / "tiny_detector.pt"
        torch.save({"model_state": model.state_dict(), "class_names": CLASS_NAMES,
                    "image_size": SIZE, "selected_epoch": best_epoch}, checkpoint_path)
        print("Saved:", checkpoint_path)
        """),
        md("""
        ## Try it yourself
        1. Explain why changing `BOX_WEIGHT` changes the tradeoff between class and box learning. Compare validation results after restarting and retraining with 1.0 and 20.0; keep the test set untouched during this comparison.
        2. Add bright backgrounds to the **training generator**, then train a fresh model. Use a separate shifted validation set to select settings.
        3. Reload the checkpoint into a new model and verify identical predictions on the same inputs.
        4. Explain what must change to detect zero, one, or many objects per image.
        """),
        md("""
        ## Solutions and discussion
        1. A larger box weight increases localization's contribution to the shared feature gradients. It does not guarantee better IoU; measure validation behavior.
        2. Randomly choose a dark or bright background for each training sample. Preserve independent validation/test seeds. Simply evaluating on shifted data does not adapt the model.
        3. The reload demonstration is below. Load only checkpoints you created or trust.
        4. A general detector needs a variable number of candidate predictions, an object/background mechanism, target assignment, and appropriate loss/evaluation logic. A single four-coordinate head cannot encode several objects.
        """),
        code("""
        saved = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        restored_model = TinySingleObjectDetector().to(DEVICE)
        restored_model.load_state_dict(saved["model_state"])
        restored_model.eval()
        model.eval()
        with torch.no_grad():
            sample_batch = test_data.tensors[0][:5].to(DEVICE)
            original_logits, original_boxes = model(sample_batch)
            restored_logits, restored_boxes = restored_model(sample_batch)
        assert torch.allclose(original_logits, restored_logits)
        assert torch.allclose(original_boxes, restored_boxes)
        print("Reloaded model reproduces the predictions.")
        """, tags=["solution"]),
        md("""
        ## Recap and next step
        You trained an actual network with shared visual features, classification and localization heads, validation checkpoint selection, held-out metrics, and a distribution-shift check. These are reusable habits; real detectors add more complex outputs and matching rules.

        **Next:** [11 · Faster R-CNN and transfer learning](11_Faster_RCNN_and_Transfer_Learning.ipynb), or revisit [03 · IoU and NMS](03_IoU_and_Non_Maximum_Suppression.ipynb).

        **References:** [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), [saving model weights](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).
        """),
    ])


if __name__ == "__main__":
    image_basics()
    classical()
    iou_nms()
    tiny_detector()
