"""Compare installed source bytes with a checksum-pinned official archive.

An online refresh is explicit. Offline verification never trusts .source.json
alone and permits build outputs without treating them as upstream source.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremeocr.data import sha256
from extremeocr.provenance import write_json


def audit(root=ROOT):
    contract = json.loads((root / "configs/baseline_source.json").read_text(encoding="utf-8"))
    locked = json.loads((root / "configs/upstream.lock.json").read_text(encoding="utf-8"))["lranet_pp"]
    if any(contract[key] != locked[key] for key in ("repo", "commit", "archive_sha256")):
        raise ValueError("Baseline source manifest and version lock disagree")
    vendor = Path(os.environ.get("EXTREMEOCR_VENDOR", str(root / locked["path"])))
    changed = [name for name, expected in contract["files"].items()
               if not (vendor / name).is_file() or sha256(vendor / name) != expected]
    # The official evaluator writes this result JSON; keep the source tree pristine
    # by restoring it only manually, never hiding changes to tracked source.
    return {"commit": contract["commit"], "files_checked": len(contract["files"]),
            "changed_or_missing": changed, "passed": not changed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Download pinned archive and rebuild source manifest")
    args = parser.parse_args()
    if args.refresh:
        locked = json.loads((ROOT / "configs/upstream.lock.json").read_text(encoding="utf-8"))["lranet_pp"]
        url = "https://codeload.github.com/{}/zip/{}".format(locked["repo"], locked["commit"])
        payload = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(payload).hexdigest() != locked["archive_sha256"]:
            raise SystemExit("Official archive checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            files = {name.split("/", 1)[1]: hashlib.sha256(archive.read(name)).hexdigest()
                     for name in archive.namelist() if "/" in name and not name.endswith("/")}
        write_json(ROOT / "configs/baseline_source.json", {
            **{key: locked[key] for key in ("repo", "commit", "archive_sha256")},
            "files": files})
    report = audit()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
