"""Run only in baseline environment, after inference on project dev images."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremeocr.dev_evaluation import detection_zip, parsed_metrics, prepare_dev
from extremeocr.provenance import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if meta["split"] != "dev" or meta["protocol"] != "fixed":
        raise ValueError("Dev evaluation requires dev/fixed protocol")
    import mmcv
    doc = json.loads(Path(meta["annotations"]).read_text(encoding="utf-8"))
    predictions = mmcv.load(str(run / "predictions.pkl"))
    folder = run / "dev_evaluator"
    gt = prepare_dev(doc, predictions, folder)
    vendor = Path(meta["vendor"])
    sys.path.insert(0, str(vendor / "mmocr/core/evaluation/evaluation_e2e"))
    from text_eval_script import text_eval_main
    records = json.loads((folder / "my_text_results.json").read_text(encoding="utf-8"))
    detection_zip(records, folder / "det_dev.zip", meta["score_threshold"])
    result = text_eval_main(str(folder / "det_dev.zip"), str(gt), True)
    metrics = parsed_metrics(result)
    write_json(run / "dev_metrics.json", {"protocol": "project-dev-upstream-word-spotting-matcher",
                                          "metrics": metrics, "raw": result})
    print(metrics)


if __name__ == "__main__":
    main()
