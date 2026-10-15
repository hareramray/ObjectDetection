"""Small standard-library notebook authoring helpers."""
import json
import textwrap
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": textwrap.dedent(text).strip() + "\n"}

def code(text, tags=None):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8],
            "metadata": {"tags": tags or []}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip() + "\n"}

def bootstrap():
    return code('''
    from pathlib import Path
    import sys
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "course_utils.py").exists()), None)
    if ROOT is None:
        raise FileNotFoundError("Open Jupyter from the ObjectDetection project folder (see README.md).")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from course_utils import seed_everything, show_boxes
    seed_everything(42)
    print("Project:", ROOT)
    ''')

def write_notebook(filename, cells):
    path = ROOT / "notebooks" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python (Object Detection)", "language": "python", "name": "object-detection"},
        "language_info": {"name": "python", "version": "3.13"},
    }, "cells": cells}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(path.name)
