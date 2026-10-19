"""One view of which experiment links actually completed; never infer scores."""
import argparse
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from collect_metrics import parse_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    meta = json.loads((run / "run.json").read_text())
    result = {key: meta[key] for key in ("run", "mode", "module", "stage", "split", "status")}
    for key, filename in (("resources", "watchdog.json"), ("checkpoint", "checkpoint_keys_rank0.json"),
                          ("adapter", "adapter_health_rank0.json"), ("dev", "dev_metrics.json")):
        path = run / filename
        result[key] = json.loads(path.read_text()) if path.exists() else "not_run"
    result["metrics"] = None
    if meta["mode"] == "eval" and meta["status"] == "completed":
        result["metrics"] = parse_metrics((run / "console.log").read_text(errors="replace"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
