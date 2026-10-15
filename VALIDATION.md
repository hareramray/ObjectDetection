# Course validation

Validated locally on **15 October 2026**, using Windows, Python 3.13.15,
PyTorch 2.11.0+cu128 / TorchVision 0.26.0+cu128, and an NVIDIA GeForce RTX 5050
Laptop GPU. Exact observed package versions are in `environment-tested.txt`.
`pip check` reported no broken requirements.

All **14 notebooks** passed notebook-format validation, Python syntax checks and
execution in a fresh **Python (Object Detection)** Jupyter kernel. There are
224 cells, including 109 executable cells. Internal notebook links were checked.
Source notebooks are unexecuted for classroom use; the actual executed copies are
under `artifacts/executed/`, with inline figures and no error outputs.

| Lessons | Work actually verified |
|---|---|
| 00–03 | Environment checks, image/box operations, classical detector failures, IoU and class-aware NMS including edge cases |
| 04 | Actual YOLO11n inference on the cached example image, threshold sweep, saved predictions |
| 05 | All 48 artificial pan frames, persistent track IDs, both crossing directions, bounded processing and deterministic crossing assertions |
| 06 | Generated 96/24/24 image splits, annotation round-trip, class counts and cross-split duplicate checks |
| 07 | GPU smoke/full runs and CPU five-epoch training on all 96 training images; saved checkpoint and manifest; custom YAML setup checks |
| 08 | Hand-computed matching/AP assertions and actual validation of the saved custom model |
| 09 | Bounding-box flip/crop checks, measured background-shift failure and custom-model examples |
| 10 | Full 18-epoch small-CNN training, separate train/validation/test samples, checkpoint selection/reload and distribution-shift experiment |
| 11 | Pretrained Faster R-CNN initialization, two real optimizer updates with finite losses and changed head weights, followed by inference |
| 12 | Real ONNX export, model check, runtime inference, box agreement and warmed-up timing |
| 13 | Complete classical baseline evaluation on the generated test split; saved CSV, JSON, plot and model report |

The ONNX smoke check matched all five detections to PyTorch, with a maximum box
coordinate difference of approximately 0.000122 pixels. This checks the example's
exported inference path, not accuracy on a representative dataset. Timing numbers
are local measurements and are not a general speed claim.

The small CNN achieved approximately 0.996 held-out class accuracy and 0.856 mean
IoU on its simple single-object synthetic distribution. Its shifted-background
example failed badly, illustrating that these results do not imply real-world
generalization. The five-epoch YOLO run is also a teaching demonstration.

One repeated GPU training verification stopped progressing and its kernel was
terminated after approximately five minutes. Its arguments and dataset matched
the earlier successful run; the runtime cause was not established. The final
custom-training lesson defaults to CPU, where five epochs completed normally;
CUDA remains an explicit option. Use `--cpu` with the verifier to force this path.

Offline branches were checked separately for the inference/export and
training/evaluation lessons. An offline branch that skips a model is recorded as
such; it is not used as evidence of successful model execution. The optional
webcam, long 25-epoch training option, arbitrary user datasets, other machines,
and production deployment were **not** validated.

Representative figures were visually reviewed: coordinate mistakes, classical
failures, NMS behavior, real detections, tracking trajectories, label grids,
augmentation crops, loss curves, tiny-CNN predictions and capstone output.

Additional checks verified custom YAML class names in the data audit and rejection
of truncated image files. Dataset regeneration preserves incomplete existing pairs
and reports the problem instead of overwriting an existing image.

Machine-readable execution reports are under `artifacts/validation_*.json`.
Some early reports use smoke settings; newer reports and notebook execution
metadata identify the effective configuration. To repeat the course with full
default settings, run:

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebooks.py --execute
```

This performs real training and writes new model runs. It does not test the webcam,
which remains off unless explicitly enabled in notebook 05.
