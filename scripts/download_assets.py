"""Fetch author-provided Total-Text assets inside Docker, with resumable transfers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_http(url, target, attempts=12):
    import requests
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    for attempt in range(attempts):
        start = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(url, headers={"Range": "bytes={}-".format(start)} if start else {},
                              stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError("Expected asset bytes; received an HTML page")
                if response.status_code == 206:
                    content_range = response.headers["Content-Range"]
                    if not content_range.startswith("bytes {}-".format(start)):
                        raise ValueError("Server returned an unexpected byte range")
                    total = int(content_range.rsplit("/", 1)[1])
                    mode = "ab"
                else:
                    start = 0
                    total = int(response.headers["Content-Length"])
                    mode = "wb"
                with partial.open(mode) as handle:
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
                if partial.stat().st_size != total:
                    raise ValueError("Incomplete transfer")
                partial.replace(target)
                return
        except (requests.RequestException, ValueError) as error:
            print("transfer retry {}/{}: {}".format(attempt + 1, attempts, str(error)[:160]), flush=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(attempt + 1, 5))


def fetch(entry, target):
    target = Path(target)
    receipt = target.with_name(target.name + ".source.json")
    if target.exists() and receipt.exists():
        previous = json.loads(receipt.read_text())
        if sha256(target) != previous["sha256"]:
            raise ValueError("Existing asset changed: " + str(target))
        if entry.get("sha256") and previous["sha256"] != entry["sha256"]:
            raise ValueError("Existing asset differs from pinned source")
        print("verified " + str(target), flush=True)
        return previous
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if "drive_id" in entry:
            import gdown
            result = gdown.download(id=entry["drive_id"], output=str(target), quiet=True,
                                    resume=True, retries=32, timeout=(30, 90))
            if result is None:
                raise RuntimeError("Google Drive download failed")
        else:
            download_http(entry["url"], target)
    digest = sha256(target)
    if entry.get("sha256") and digest != entry["sha256"]:
        raise ValueError("Asset SHA256 differs from fixed source: " + str(target))
    report = dict(entry, sha256=digest, bytes=target.stat().st_size,
                  downloaded_at=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print("downloaded {}: {}".format(target, digest), flush=True)
    return report


def extract_images(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        bad = source.testzip()
        if bad:
            raise ValueError("Corrupt archive member: " + bad)
        count = 0
        for member in source.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX/"):
                continue
            parts = Path(member.filename).parts
            # Author archive stores totaltext/{training,test}/... .
            if "totaltext" in parts:
                parts = parts[parts.index("totaltext") + 1:]
            target = destination.joinpath(*parts).resolve()
            target.relative_to(destination)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not supported")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                expected = hashlib.sha256()
                with source.open(member) as input_file:
                    for block in iter(lambda: input_file.read(1024 * 1024), b""):
                        expected.update(block)
                if sha256(target) != expected.hexdigest():
                    raise ValueError("Extracted asset changed; refusing to overwrite: " + str(target))
            else:
                with source.open(member) as input_file, target.open("wb") as output_file:
                    import shutil
                    shutil.copyfileobj(input_file, output_file)
            count += 1
    return count


def validate_dataset(root):
    from PIL import Image
    report = {}
    for split in ("train", "test"):
        annotation = root / ("totaltext_" + split + ".json")
        data = json.loads(annotation.read_text())
        images = []
        for item in data["images"]:
            image_root = root / "training" if split == "train" else root
            path = (image_root / item["file_name"]).resolve()
            path.relative_to(root.resolve())
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.size != (item["width"], item["height"]):
                    raise ValueError("Image dimensions differ from annotation: " + str(path))
            images.append({"file_name": item["file_name"], "sha256": sha256(path)})
        expected = 1255 if split == "train" else 300
        if len(images) != expected:
            raise ValueError("Unexpected Total-Text {} size: {}".format(split, len(images)))
        report[split] = {"images": images, "image_count": len(images),
                         "annotation_count": len(data["annotations"]),
                         "annotation_sha256": sha256(annotation)}
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--only", choices=("all", "data", "checkpoint"), default="all")
    args = parser.parse_args()
    if os.environ.get("EXTREMEOCR_CONTAINER") != "1":
        raise SystemExit("Run this downloader inside the ExtremeOCR Docker container")
    os.chdir(ROOT)
    sources = json.loads((ROOT / "configs/asset_sources.json").read_text())
    if args.only in ("all", "checkpoint"):
        entry = sources["totaltext_checkpoint"]
        fetch(entry, ROOT / entry["path"])
    if args.only in ("all", "data"):
        for key in ("totaltext_train", "totaltext_test"):
            entry = sources[key]
            fetch(entry, ROOT / entry["path"])
        entry = sources["totaltext_images"]
        fetch(entry, ROOT / entry["archive"])
        root = ROOT / "data/raw/totaltext"
        extract_images(ROOT / entry["archive"], root)
        report = validate_dataset(root)
        report["source"] = "LRANet++ author-prepared Total-Text; original JSON order retained"
        report["license"] = "Original Total-Text dataset and author distribution terms apply; no new license granted"
        (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v["image_count"] for k, v in report.items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
