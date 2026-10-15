# Object Detection: a practical Jupyter course

Fourteen notebooks take you from pixels and bounding boxes to custom training,
evaluation, tracking, a manual PyTorch training loop, ONNX export and a final project.
They are designed for **live demonstrations on local Windows Jupyter**: explain a
concept, run a short cell, inspect a result, change a setting, and discuss what changed.

Each lesson contains objectives, prerequisites, numbered steps, runnable examples,
presenter prompts, experiments, and exercises with solutions. The small synthetic
dataset is generated automatically; general-object inference uses a pretrained model.

## Start on this computer

The project environment `.venv` has been prepared and the
**Python (Object Detection)** Jupyter kernel registered. It reuses the existing
Python 3.13 / PyTorch CUDA installation and installs the remaining tools locally.
Run from PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

Open [00_Start_Here_and_Environment.ipynb](notebooks/00_Start_Here_and_Environment.ipynb),
select **Python (Object Detection)**, and use **Shift+Enter** to run each cell.
No environment activation or PowerShell execution-policy change is required.

## Learning sequence

Teaching times include explanation and exercises; training/download time varies.

| # | Notebook | What you demonstrate | Teaching time |
|---|---|---|---|
| 00 | [Start here](notebooks/00_Start_Here_and_Environment.ipynb) | Jupyter, Python environment, detection vocabulary | 25–40 min |
| 01 | [Images and boxes](notebooks/01_Image_Basics_and_Bounding_Boxes.ipynb) | RGB arrays, crops, normalization, resizing | 35–45 min |
| 02 | [Classical detection](notebooks/02_Classical_Object_Detection.ipynb) | Thresholds, contours, fixed-rule failures | 35–50 min |
| 03 | [IoU and NMS](notebooks/03_IoU_and_Non_Maximum_Suppression.ipynb) | Overlap, duplicate suppression, class-aware NMS | 35–50 min |
| 04 | [Pretrained YOLO](notebooks/04_Pretrained_YOLO_Detection.ipynb) | Detect everyday objects and inspect scores | 25–40 min |
| 05 | [Video and tracking](notebooks/05_Video_Tracking_and_Counting.ipynb) | Track IDs, bounded video processing, crossing counts | 40–60 min |
| 06 | [Custom dataset](notebooks/06_Custom_Dataset_and_Annotation.ipynb) | YOLO labels, annotation, leakage and data audits | 45–60 min |
| 07 | [Custom YOLO training](notebooks/07_Train_a_Custom_YOLO_Detector.ipynb) | Fine-tune, inspect losses, save a checkpoint | 35–50 min + training |
| 08 | [Precision, recall and mAP](notebooks/08_Evaluation_Precision_Recall_and_mAP.ipynb) | Match predictions, sweep thresholds, compute AP | 45–60 min |
| 09 | [Augmentation and errors](notebooks/09_Augmentation_and_Error_Analysis.ipynb) | Correct box transforms and domain-shift experiments | 40–60 min |
| 10 | [Tiny PyTorch detector](notebooks/10_Build_a_Tiny_Detector_in_PyTorch.ipynb) | Train a single-object CNN with two prediction heads | 60–90 min |
| 11 | [Faster R-CNN](notebooks/11_Faster_RCNN_and_Transfer_Learning.ipynb) | Dataset targets, transfer learning, optimizer steps | 50–75 min |
| 12 | [ONNX export](notebooks/12_ONNX_Export_and_Benchmarking.ipynb) | Export, verify numerical agreement, benchmark | 45–60 min |
| 13 | [Capstone](notebooks/13_Capstone_Reproducible_Detection_Project.ipynb) | Build a reproducible detection project and report | 60–90 min |

Start in order. Later notebooks explicitly recreate their data or load a saved
checkpoint; they do not rely on variables left in another notebook's kernel.
See [DEMONSTRATION_GUIDE.md](DEMONSTRATION_GUIDE.md) for shorter presentation routes.

## Re-create the environment on another Windows machine

Python 3.13 is the tested version. Use a supported 64-bit Python and install PyTorch
and TorchVision as a compatible pair. For a particular NVIDIA CUDA build, use the
[official PyTorch installer](https://pytorch.org/get-started/locally/) before the
remaining requirements. The course also runs on CPU, with longer training times.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name object-detection --display-name "Python (Object Detection)"
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

For ONNX lesson 12:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-onnx.txt
```

`setup_windows.ps1` automates these steps. Its optional `-ReuseSystemPackages` switch
can reuse an existing compatible PyTorch installation, as on this computer;
`-WithOnnx` adds export dependencies. `environment-tested.txt` records observed
versions, including the CUDA build, and is a reference snapshot rather than a
portable lockfile. Ultralytics is pinned in the requirements to the validated version.

## Prepare before presenting

```powershell
.\.venv\Scripts\python.exe scripts/prepare_demo.py --faster-rcnn
```

This generates and audits 144 original shape images (96 train / 24 validation /
24 test), caches the example photograph and YOLO weights, builds a short artificial
pan video, and optionally caches Faster R-CNN weights. Initial downloads require
internet; model weights are tens to hundreds of megabytes. No account or paid API
is required. [Asset provenance](assets/README.md) identifies external samples.

For the offline introductory route, use `scripts/prepare_demo.py --offline`.
`COURSE_OFFLINE=1` deliberately **skips** online/model stages; it does not mean
“use every model from cache.” For normal cached inference, leave that variable unset.

The webcam example is off by default, has a frame limit, and releases the camera
in a `finally` block. A local video file is the recommended repeatable classroom demo.

## Training and outputs

- Notebook 07 runs five epochs by default; `FULL_RUN=True` selects a longer run.
  It defaults to CPU for a repeatable classroom demo; set `DEVICE = choose_device()`
  in its settings cell to use the available CUDA GPU.
  It saves the actual checkpoint location in `artifacts/custom_model.json`.
  Its custom-data option reads your YAML class names; notebook 08 validates the
  dataset recorded in that manifest. Shape-specific examples in 09 and 13 need
  their data and class mapping adapted together when using another domain.
- Notebook 10 trains a small **single-object** detector. It is an educational
  class/box model, not a general replacement for YOLO or Faster R-CNN.
- Notebook 11 performs bounded real optimizer updates to demonstrate fine-tuning.
  It needs substantially more training for useful predictions.
- Notebook 13 starts with a fully runnable classical baseline. Change `ENGINE`
  to `"yolo"` after notebook 07 to use your custom checkpoint.

Generated data is under `data/`, cached YOLO files under `models/`, inference/export
outputs under `outputs/`, and training/evaluation/capstone files under `artifacts/`.
TorchVision uses its normal Torch cache for pretrained weights.

Source notebooks in `notebooks/` are clean for presenting. Executed rehearsal copies,
including plots and console output, are in `artifacts/executed/`; these may use the
short smoke settings described in `VALIDATION.md`. Open them as examples of observed
results, not as promises of accuracy on new data.

## Verify or modify the course

```powershell
# Check notebook format and Python syntax.
.\.venv\Scripts\python.exe scripts/validate_notebooks.py

# Execute actual models with reduced training/frame counts.
.\.venv\Scripts\python.exe scripts/validate_notebooks.py --execute --smoke

# Execute only selected lessons, skipping online stages.
.\.venv\Scripts\python.exe scripts/validate_notebooks.py --execute --offline --only 01 02 03 06

# Force CPU during verification, even on a CUDA-capable machine.
.\.venv\Scripts\python.exe scripts/validate_notebooks.py --execute --cpu --only 07
```

Each notebook runs in a fresh Jupyter kernel. Reports are saved under `artifacts/`.
`COURSE_SMOKE=1` reduces expensive work for a functional check; it is not an accuracy
benchmark. Running training validation again creates a new run/checkpoint manifest.
`scripts/build_*.py` are the authoring sources. Running one regenerates its notebooks
and replaces manual edits to those generated files; edit either the notebook for a
one-off class, or its generator if you want to rebuild consistently.

## Common fixes

| Symptom | Action |
|---|---|
| `ModuleNotFoundError` | Select the project kernel; check `sys.executable` in notebook 00. |
| Kernel is unavailable | Repeat the `ipykernel install --prefix .venv` command and launch Jupyter from `.venv`. |
| Dataset or helper cannot be found | Launch from the project folder; keep `course_utils.py` beside `notebooks/`. |
| CUDA out of memory | Reduce batch/image size, close other GPU kernels, or set the lesson's device to `"cpu"`. |
| `torchvision::nms` import error | Reinstall a matching Torch/TorchVision pair from the official installer. |
| Download fails | Check connectivity or prepare the caches before class; offline early lessons still work. |
| No custom checkpoint yet | Run 07; optional model sections in later lessons explain when they are skipped. |
| Empty predictions after quick training | This is possible; inspect scores, labels and losses, then train longer using validation. |
| Slow CPU or long first cell | Initial imports/model loading take time; rehearse and use smoke settings for a short check. |

The curriculum develops beginner through advanced practical skills. Expert-level
judgment requires repeating experiments on representative real data. Synthetic shapes,
an artificial video and short training runs are teaching aids, not production evidence.

Primary references appear inside the relevant lessons: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
[training](https://docs.ultralytics.com/modes/train/), [validation](https://docs.ultralytics.com/modes/val/),
[tracking](https://docs.ultralytics.com/modes/track/), [export](https://docs.ultralytics.com/modes/export/),
and [TorchVision detection fine-tuning](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
