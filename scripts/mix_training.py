"""Merge variants by references, preserving all per-instance annotation fields."""
import argparse
import copy
import json
from pathlib import Path


def mix(suite, annotation_name, variants):
    suite = Path(suite)
    if "clean" not in variants or len(variants) < 2:
        raise ValueError("Include clean and at least one training corruption")
    if len(variants) != len(set(variants)):
        raise ValueError("Duplicate variants")
    result = None
    for variant in variants:
        folder = (suite / variant).resolve()
        folder.relative_to(suite.resolve())
        doc = json.loads((folder / annotation_name).read_text(encoding="utf-8"))
        if result is None:
            result = copy.deepcopy(doc)
            result["images"], result["annotations"] = [], []
        mapping = {}
        for image in doc["images"]:
            image = copy.deepcopy(image)
            old_id = image["id"]
            image["id"] = len(result["images"]) + 1
            mapping[old_id] = image["id"]
            image["file_name"] = (Path(variant) / "images" / image["file_name"]).as_posix()
            result["images"].append(image)
        for ann in doc["annotations"]:
            ann = copy.deepcopy(ann)
            ann["image_id"] = mapping[ann["image_id"]]
            ann["id"] = len(result["annotations"]) + 1
            result["annotations"].append(ann)
    result["extremeocr_training_variants"] = variants
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--annotation-name", required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise SystemExit("Output already exists")
    doc = mix(a.suite, a.annotation_name, a.variants)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"images": len(doc["images"]), "annotations": len(doc["annotations"])}))


if __name__ == "__main__":
    main()
