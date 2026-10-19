"""Summarize confidence filtering from explicit score-probe inference."""
import argparse
import json
from pathlib import Path


def summarize(predictions, threshold):
    count, pre_kept, post_kept, lost = 0, 0, 0, 0
    def kept(score):
        return score > 0.1 and round(score, 3) >= threshold
    for result in predictions:
        before = result["diagnostic_detection_scores"]
        after = [boundary[-1] for boundary in result["boundary_result"]]
        if len(before) != len(after):
            raise ValueError("Score candidate count mismatch")
        for pre, post in zip(before, after):
            if not (0 <= post <= pre + 1e-6 <= 1 + 1e-6):
                raise ValueError("Scores violate expected confidence multiplication")
            count += 1
            pre_kept += kept(pre)
            post_kept += kept(post)
            lost += kept(pre) and not kept(post)
    return {"candidates": count, "pre_rec_kept": pre_kept, "post_rec_kept": post_kept,
            "lost_after_recognition_confidence": lost, "threshold": threshold,
            "protocol": "candidate filtering diagnostic, not detection accuracy or official Hmean"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    a = p.parse_args()
    run = Path(a.run_dir)
    meta = json.loads((run / "run.json").read_text())
    if not meta.get("score_probe"):
        raise ValueError("Run was not prepared with --score-probe")
    import mmcv
    report = summarize(mmcv.load(str(run / "predictions.pkl")), meta["score_threshold"])
    (run / "score_probe_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
