"""Validate notebook structure and optionally execute with isolated Jupyter kernels."""
import argparse
import ast
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Skip pretrained/network stages")
    parser.add_argument("--smoke", action="store_true", help="Reduce expensive training/repetition counts")
    parser.add_argument("--cpu", action="store_true", help="Hide CUDA devices from verification kernels")
    parser.add_argument("--only", nargs="*", help="Notebook prefixes, e.g. 00 01 06")
    args = parser.parse_args()
    import nbformat
    if args.offline:
        os.environ["COURSE_OFFLINE"] = "1"
    if args.smoke:
        os.environ["COURSE_SMOKE"] = "1"
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    effective_offline = os.getenv("COURSE_OFFLINE", "0") == "1"
    effective_smoke = os.getenv("COURSE_SMOKE", "0") == "1"
    # Capture actual figure outputs in executed notebooks, not a silent Agg backend.
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if args.only:
        notebooks = [p for p in notebooks if p.name[:2] in args.only]
    if not notebooks:
        raise SystemExit("No notebooks matched")
    report = []
    output_dir = ROOT / "artifacts" / "executed"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in notebooks:
        started = time.perf_counter()
        record = {"notebook": path.name, "mode": "execute" if args.execute else "structure",
                  "smoke": effective_smoke, "offline": effective_offline, "forced_cpu": args.cpu}
        print(f"Checking {path.name}", flush=True)
        try:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            for index, cell in enumerate(nb.cells):
                if cell.cell_type == "code":
                    ast.parse(cell.source, filename=f"{path.name}:cell{index}")
            record["cells"] = len(nb.cells)
            if args.execute:
                from nbclient import NotebookClient
                nb.metadata["course_execution"] = {
                    "smoke": effective_smoke, "offline": effective_offline,
                    "kernel": "object-detection", "forced_cpu": args.cpu}
                client = NotebookClient(nb, timeout=1800, kernel_name="object-detection",
                                        resources={"metadata": {"path": str(ROOT / "notebooks")}})
                try:
                    client.execute()
                finally:
                    nbformat.write(nb, output_dir / path.name)
            record["status"] = "passed"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            print(str(exc), flush=True)
        record["seconds"] = round(time.perf_counter()-started, 2)
        report.append(record)
        print(f"  {record['status']} ({record['seconds']} s)", flush=True)
    suffix = "_".join(args.only) if args.only else "all"
    report_path = ROOT / "artifacts" / f"validation_{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", report_path)
    if any(row["status"] == "failed" for row in report):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
