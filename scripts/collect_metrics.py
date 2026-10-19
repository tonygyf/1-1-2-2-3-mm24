"""Extract actual LRANet return values; no default/fabricated research scores."""
import argparse
import ast
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extremeocr.data import sha256


def verify_inputs(meta):
    if not meta.get("dataset"):
        raise ValueError("Run lacks execution-time dataset provenance")
    for key in ("checkpoint_asset", "basis_asset", "config_asset"):
        record = meta[key]
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Run input changed: " + record["path"])
    record = meta["dataset"]["annotation"]
    if sha256(record["path"]) != record["sha256"]:
        raise ValueError("Annotation changed since run start")
    reference = meta["dataset"].get("official_reference")
    if reference and sha256(reference["path"]) != reference["sha256"]:
        raise ValueError("Official reference annotation changed since run start")
    for image in meta["dataset"]["images"]:
        if sha256(Path(meta["images"]) / image["file_name"]) != image["sha256"]:
            raise ValueError("Image changed since run start: " + image["file_name"])


def parse_metrics(text):
    for line in reversed(text.splitlines()):
        try:
            result = ast.literal_eval(line.strip())
        except (ValueError, SyntaxError):
            continue
        if isinstance(result, dict) and {"e2e-hmean", "det-hmean"} <= result.keys():
            values = {key: float(result[key]) for key in ("e2e-hmean", "det-hmean")}
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
                raise ValueError("Expected finite H-mean fractions in [0,1]")
            return values
    raise ValueError("No successful evaluator metrics found in log")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True, help="CSV result ledger")
    a = p.parse_args()
    run = Path(a.run_dir)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if meta["mode"] != "eval" or meta["status"] != "completed":
        raise SystemExit("Collect only completed evaluation runs")
    verify_inputs(meta)
    log = (run / "console.log").read_text(encoding="utf-8", errors="replace")
    metrics = parse_metrics(log)
    row = {"run": meta["run"], "module": meta["module"], "split": meta["split"],
           "checkpoint": meta["checkpoint"], "annotations": meta["annotations"],
           "annotation_sha256": meta["dataset"]["annotation"]["sha256"],
           "checkpoint_sha256": meta["checkpoint_asset"]["sha256"],
           "threshold": meta["score_threshold"], "protocol": meta["evaluation_protocol"],
           "det_hmean": metrics["det-hmean"], "e2e_hmean": metrics["e2e-hmean"]}
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    exists = dest.exists() and dest.stat().st_size > 0
    if exists:
        with dest.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(row):
                raise ValueError("Ledger schema changed; choose a new CSV")
            if any(previous["run"] == row["run"] for previous in reader):
                raise ValueError("Run already collected")
    with dest.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
