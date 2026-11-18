"""Rebuild the reviewed compatibility source from pinned upstream archives."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def unpack(payload, destination, kind):
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        if kind == "zip":
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for name in archive.namelist():
                    target = (temporary / name).resolve()
                    target.relative_to(temporary)
                archive.extractall(temporary)
        else:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                archive.extractall(temporary, filter="data")
        children = list(temporary.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise ValueError("Expected a single upstream archive root")
        shutil.copytree(children[0], destination, symlinks=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--destination", default="work/ports")
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--patch-dir", default=str(ROOT / "docker/patches"))
    args = parser.parse_args()
    if not Path("/.dockerenv").exists() and os.environ.get("EXTREMEOCR_CONTAINER") != "1":
        raise SystemExit("Prepare the CUDA port inside Docker")
    configs, patches = Path(args.config_dir), Path(args.patch_dir)
    port = json.loads((configs / "modern_port.json").read_text())
    locked = json.loads((configs / "upstream.lock.json").read_text())["lranet_pp"]
    specs = dict(port["dependencies"])
    specs["lranet"] = dict(locked, url="https://codeload.github.com/{}/zip/{}".format(locked["repo"], locked["commit"]))
    root = Path(args.destination)
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
        destination = root / name
        if destination.exists():
            raise ValueError("Refusing to overwrite port source: " + str(destination))
        url = spec.get("url") or "https://codeload.github.com/{}/tar.gz/refs/tags/{}".format(spec["repo"], spec["tag"])
        payload = urllib.request.urlopen(url, timeout=90).read()
        if hashlib.sha256(payload).hexdigest() != spec["archive_sha256"]:
            raise ValueError("Upstream archive SHA256 changed: " + name)
        unpack(payload, destination, "zip" if name == "lranet" else "tar")
        patch_name = name + "-modern.patch"
        patch = patches / patch_name
        if hashlib.sha256(patch.read_bytes()).hexdigest() != port["patches"][patch_name]:
            raise ValueError("Reviewed patch SHA256 changed: " + name)
        subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(patch.resolve())], cwd=destination, check=True)
        if name == "lranet":
            for relative, expected in port["lranet_patched_files"].items():
                if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != expected:
                    raise ValueError("Patched LRANet file differs: " + relative)
        (destination / ".extremeocr-source.json").write_text(json.dumps(spec, indent=2) + "\n")
        print("prepared " + name, flush=True)


if __name__ == "__main__":
    main()
