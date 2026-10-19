"""Export per-instance saturation and axis-aligned crops for a GT-crop probe.

Crops are a recognizer interface, not curved-text rectification or OCR results.
Feed the same frozen external recognizer on clean/degraded crops, then use
score_crops.py. Review readability manually without changing official labels.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremeocr.provenance import asset, write_json


def export(images, annotations, output):
    images, annotations, output = Path(images).resolve(), Path(annotations), Path(output)
    if output.exists():
        raise ValueError("Use a new probe output directory")
    doc = json.loads(annotations.read_text(encoding="utf-8"))
    by_id = {}
    for ann in doc["annotations"]:
        by_id.setdefault(ann["image_id"], []).append(ann)
    rows = []
    (output / "crops").mkdir(parents=True)
    for im in doc["images"]:
        path = (images / im["file_name"]).resolve()
        path.relative_to(images)
        with Image.open(path) as source:
            source = source.convert("RGB")
            rgb = np.asarray(source).astype(float) / 255
            for ann in by_id.get(im["id"], []):
                if ann.get("ignore") or ann.get("iscrowd") or ann.get("transcription") == "###":
                    continue
                polygons = ann.get("segmentation", [])
                if len(polygons) != 1:
                    raise ValueError("Probe needs one polygon per annotation")
                xy = np.asarray(polygons[0], dtype=float).reshape(-1, 2)
                if len(xy) < 4 or not np.isfinite(xy).all():
                    raise ValueError("Invalid instance polygon")
                mask = Image.new("1", source.size)
                ImageDraw.Draw(mask).polygon([tuple(p) for p in xy], fill=1)
                inside = np.asarray(mask, dtype=bool)
                if not inside.any():
                    raise ValueError("Empty instance mask")
                box = mask.getbbox()
                name = str(len(rows)).zfill(7) + ".png"
                source.crop(box).save(output / "crops" / name)
                rows.append({"image_id": im["id"], "annotation_id": ann["id"],
                             "source": im["file_name"], "crop": "crops/" + name,
                             "text": ann["transcription"], "box": box,
                             "any_clip_fraction": float((rgb.max(2)[inside] >= 0.98).mean()),
                             "all_clip_fraction": float((rgb.min(2)[inside] >= 0.98).mean()),
                             "readability": "unreviewed", "geometry": "axis-aligned polygon bounding crop"})
    (output / "instances.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(output / "probe.json", {"annotation": asset(annotations), "instances": len(rows),
                                       "ocr_executed": False, "pixel_saturation_threshold": 0.98,
                                       "care_policy": "nonignored original annotations, not official word-spotting care"})
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    for key in ("images", "annotations", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    print(json.dumps({"exported_instances": export(args.images, args.annotations, args.output)}))


if __name__ == "__main__":
    main()
