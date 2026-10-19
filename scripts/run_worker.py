"""A process-group-owned worker; no shell command evaluation."""
import json
from pathlib import Path
import subprocess
import sys
import hashlib


def main():
    run = Path(sys.argv[1]).resolve()
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    with (run / "pip-freeze.txt").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    # Both source-tree and installed-package extension paths matter.
    binaries = {}
    import importlib.util
    roots = [Path(meta["vendor"])]
    mmcv_spec = importlib.util.find_spec("mmcv")
    if mmcv_spec and mmcv_spec.origin:
        roots.append(Path(mmcv_spec.origin).parent)
    for root in roots:
        for path in root.rglob("*.so"):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            binaries[str(path)] = digest.hexdigest()
    (run / "extension_hashes.json").write_text(json.dumps(binaries, indent=2))
    code = subprocess.call(meta["command"])
    if code == 0 and meta["mode"] == "eval" and meta["split"] == "dev":
        code = subprocess.call([sys.executable, str(Path(__file__).with_name("evaluate_dev.py")),
                                "--run-dir", str(run)])
    if code == 0 and meta.get("score_probe"):
        code = subprocess.call([sys.executable, str(Path(__file__).with_name("summarize_scores.py")),
                                "--run-dir", str(run)])
    raise SystemExit(code)


if __name__ == "__main__":
    main()
