"""Fetch the exact baseline snapshot without a nested Git repository."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "configs/upstream.lock.json").read_text(encoding="utf-8"))
    baseline = lock["lranet_pp"]
    dest = ROOT / "work/upstream/LRANet-PP"
    if (dest / ".source.json").exists():
        print("Baseline snapshot already present: " + str(dest))
        return
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit("Unregistered snapshot exists; inspect it before replacing: " + str(dest))
    url = "https://codeload.github.com/{}/zip/{}".format(baseline["repo"], baseline["commit"])
    payload = urllib.request.urlopen(url, timeout=60).read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != baseline["archive_sha256"]:
        raise SystemExit("Archive digest differs from upstream.lock.json")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            parts = Path(name).parts[1:]
            if not parts or name.endswith("/"):
                continue
            target = dest.joinpath(*parts).resolve()
            target.relative_to(dest.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    (dest / ".source.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print("Fetched " + baseline["commit"])


if __name__ == "__main__":
    main()
