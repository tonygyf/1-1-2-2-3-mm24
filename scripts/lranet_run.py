"""Run inside the LRANet environment; create explicit per-run config and logs.

Usage: python scripts/lranet_run.py --help
The command is printed by default. --execute actually starts the baseline.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["eval", "train"])
    p.add_argument("--images", required=True, help="Prefix joined with annotation file_name")
    p.add_argument("--annotations", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run", required=True, help="New directory under work/runs, e.g. E001-clean")
    p.add_argument("--adapter", action="store_true")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    vendor = ROOT / "work/upstream/LRANet-PP"
    run_root = (ROOT / "work/runs").resolve()
    run = (run_root / a.run).resolve()
    run.relative_to(run_root)
    if run == run_root:
        raise SystemExit("Use a named run directory")
    if run.exists() and any(run.iterdir()):
        raise SystemExit("Run directory is not empty; choose a new run name")
    run.mkdir(parents=True, exist_ok=True)
    base = (vendor / "configs/lranet_pp/lranet_pp_totaltext.py").resolve()
    images, ann, ckpt = [str(Path(x).resolve()).replace("\\", "/")
                         for x in (a.images, a.annotations, a.checkpoint)]
    lines = ["_base_ = " + repr(base.as_posix()),
             "custom_imports = dict(imports=['extremeocr.lranet_protocol'" +
             (", 'extremeocr.lranet_plugin'" if a.adapter else "") +
             "], allow_failed_imports=False)"]
    if a.adapter:
        lines += ["model = dict(type='HighlightLRANetPP')"]
    if a.mode == "eval":
        lines += ["data = dict(test=dict(ann_file=" + repr(ann) +
                  ", img_prefix=" + repr(images) + "))"]
    else:
        # Training inherits the clean validation data, set up via the data symlink.
        lines += ["data = dict(train=dict(dataset=dict(ann_file=" + repr(ann) +
                  ", img_prefix=" + repr(images) + ")))" ]
        lines += ["load_from = " + repr(ckpt), "seed = " + str(a.seed)]
    config = run / "config.py"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(vendor), env.get("PYTHONPATH", "")])
    env["EXTREMEOCR_SCORE_THRESHOLD"] = str(a.score_threshold)
    if a.mode == "eval":
        command = [sys.executable, "tools/test.py", str(config), ckpt,
                   "--out", str(run / "predictions.pkl"), "--eval", "hmean-e2e"]
    else:
        command = ["bash", "tools/dist_train.sh", str(config), str(run), str(a.gpus),
                   "--load-from", ckpt, "--seed", str(a.seed)]
    record = dict(vars(a), command=command, cwd=str(vendor),
                  evaluation_protocol="fixed-threshold-totaltext-no-lexicon",
                  status="prepared", upstream=json.loads(
                      (ROOT / "configs/upstream.lock.json").read_text(encoding="utf-8"))["lranet_pp"])
    meta = run / "run.json"
    meta.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"command": command, "cwd": str(vendor), "config": str(config)}, indent=2))
    if a.execute:
        for path in (base, Path(images), Path(ann), Path(ckpt)):
            if not path.exists():
                raise SystemExit("Missing required resource: " + str(path))
        # Single-run lock: upstream evaluator writes temporary files in its cwd.
        lock = vendor / ".extremeocr-run.lock"
        try:
            with lock.open("x") as stream:
                stream.write(str(os.getpid()))
        except FileExistsError:
            raise SystemExit("Another run is active, or a stale lock needs inspection: " + str(lock))
        record["status"] = "running"
        meta.write_text(json.dumps(record, indent=2), encoding="utf-8")
        try:
            with (run / "console.log").open("w", encoding="utf-8") as output:
                code = subprocess.call(command, cwd=vendor, env=env, stdout=output, stderr=subprocess.STDOUT)
            record["status"] = "completed" if code == 0 else "failed"
            record["returncode"] = code
            meta.write_text(json.dumps(record, indent=2), encoding="utf-8")
        finally:
            lock.unlink()
        raise SystemExit(code)


if __name__ == "__main__":
    main()
