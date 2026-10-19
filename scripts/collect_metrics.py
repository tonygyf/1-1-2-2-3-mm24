"""Extract actual LRANet return values; no default/fabricated research scores."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path


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
    log = (run / "console.log").read_text(encoding="utf-8", errors="replace")
    metrics = parse_metrics(log)
    annotation_bytes = Path(meta["annotations"]).read_bytes()
    row = {"run": meta["run"], "adapter": meta["adapter"],
           "checkpoint": meta["checkpoint"], "annotations": meta["annotations"],
           "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
           "threshold": meta["score_threshold"], "protocol": meta["evaluation_protocol"],
           "det_hmean": metrics["det-hmean"], "e2e_hmean": metrics["e2e-hmean"]}
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    exists = dest.exists() and dest.stat().st_size > 0
    with dest.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
