# Demonstrating object detection by doing

## Prepare one day before class

1. Launch Jupyter with the project interpreter and select the project kernel.
2. Run `scripts/prepare_demo.py --faster-rcnn` while internet is available.
3. Rehearse the chosen notebooks from **Restart Kernel → Run All**.
4. If showing custom predictions, train notebook 07 beforehand. Keep the checkpoint
   and loss plots available while you demonstrate a shorter run live.
5. Open the relevant executed notebooks as a backup for slow downloads or training.
6. Use a local video for repeatability. Enable the webcam only when you want that demo.

## A 60-minute introduction

| Minutes | Action | Ask the audience |
|---|---|---|
| 0–5 | 00: show an annotated scene | What does detection add to classification? |
| 5–15 | 01: inspect pixels, boxes and a crop | Which coordinate is x? What changes when we resize? |
| 15–25 | 02: run a threshold/contour detector | What happens if the background changes? |
| 25–35 | 03: move a box and change NMS overlap | Why do two correct-looking boxes compete? |
| 35–50 | 04: run YOLO and change confidence | Which predictions disappeared? Did the image change? |
| 50–60 | Repeat on another permitted image; discuss errors | What evidence would you need before trusting it? |

## A half-day practical workshop

Use 00–04 for 75 minutes, 05 for 30 minutes, 06 for 45 minutes, and selected 07–08
cells for 60 minutes, with breaks. Prepare trained weights in advance. Participants
should inspect their labels and compute one precision/recall example themselves.

## A complete multi-session course

| Session | Lessons | Practical deliverable |
|---|---|---|
| 1: Foundations | 00–03 | Correct box conversions and NMS implementation |
| 2: Real inference | 04–05 | Annotated images, track log and crossing events |
| 3: Custom data | 06–07 | Audited labels and a saved fine-tuned detector |
| 4: Evidence | 08–09 | Metric calculations and a documented failure experiment |
| 5: Model internals | 10–11 | Trained small CNN and a manual Faster R-CNN update |
| 6: Handoff | 12–13 | Export comparison, measured timing and project report |

Allow roughly 10–14 teaching hours across the full course, plus independent practice
and longer training. Adapt the pace to the learners' Python knowledge.

## Use the same teaching rhythm

**Explain → predict → run → inspect → change → explain again.** Run small cells
instead of executing a whole lesson without discussion. Ask learners to predict an
effect before touching confidence, IoU, augmentation, batch size or image resolution.
Change one variable at a time. Keep the exercise solutions hidden until the group
has attempted the question.

Useful deliberate failures:

- 01: resize pixels but leave the boxes unchanged.
- 02: make the foreground/background threshold inappropriate.
- 03: use class-agnostic NMS on overlapping objects of different classes.
- 06: explain a negative-width label and a near-duplicate split.
- 08: count two predictions of one object as if both were correct, then fix it.
- 09: flip an image without transforming its boxes; compare shifted backgrounds.
- 10: distinguish lower training loss from held-out success.
- 12: compare first-call timing with warmed-up repeated inference.

## Final learner demonstration

Ask each learner to show one correct prediction and one failure, explain the split,
report metrics with their definitions, and save a reproducible configuration. Assess
their explanation and measurement discipline, not just whether a box looks plausible.

The capstone defaults to a synthetic inspection task. For a real-world extension,
collect your own permitted photos, define annotation rules, separate acquisition
groups across splits, and repeat the same evaluation workflow with untouched test data.
