"""Split training sources before generating any correlated corruptions."""
import copy
import hashlib
import json
from pathlib import Path

from .data import sha256
from .provenance import write_json


def split_training(images, annotations, output, fraction=0.15, seed=42, groups=None):
    images, annotations, output = Path(images).resolve(), Path(annotations).resolve(), Path(output)
    if not 0 < fraction < 1:
        raise ValueError("Dev fraction must be between zero and one")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Split output must be new/empty")
    doc = json.loads(annotations.read_text(encoding="utf-8"))
    if doc.get("extremeocr_split", {}).get("role") == "test" or "test" in annotations.stem.lower():
        raise ValueError("Split only original training annotations")
    if doc.get("extremeocr_training_variants"):
        raise ValueError("Split source images before mixing variants")
    groups = groups or {}
    if len({im["id"] for im in doc["images"]}) != len(doc["images"]):
        raise ValueError("Duplicate image ids")
    # Union explicit scene groups with byte-identical sources. Different filenames
    # of the same image must never leak across train/dev.
    parents, seen, records = {}, {}, []
    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    for im in doc["images"]:
        source = (images / im["file_name"]).resolve()
        source.relative_to(images)
        digest = sha256(source)
        group = str(groups.get(im["file_name"], im["file_name"]))
        if digest in seen:
            parents[find(group)] = find(seen[digest])
        seen[digest] = group
        records.append((im, group, digest))
    keys = sorted({find(group) for _, group, _ in records},
                  key=lambda key: hashlib.sha256((str(seed) + ":" + key).encode()).hexdigest())
    if len(keys) < 2:
        raise ValueError("At least two distinct source groups are required")
    dev_groups = set(keys[:max(1, min(len(keys) - 1, round(len(keys) * fraction)))])
    summary = {"source_annotation_sha256": sha256(annotations), "seed": seed,
               "dev_group_fraction": fraction, "group_policy": "explicit scene groups + exact-byte duplicates",
               "scene_groups_supplied": bool(groups), "source_images": len(records)}
    for role in ("train", "dev"):
        result = copy.deepcopy(doc)
        result["images"] = []
        for im, group, digest in records:
            if (find(group) in dev_groups) == (role == "dev"):
                image = copy.deepcopy(im)
                image["extremeocr_source_sha256"] = digest
                image["extremeocr_group"] = find(group)
                result["images"].append(image)
        ids = {im["id"] for im in result["images"]}
        result["annotations"] = [ann for ann in result["annotations"] if ann["image_id"] in ids]
        result["extremeocr_split"] = dict(summary, role=role)
        write_json(output / ("totaltext_" + role + ".json"), result)
        summary[role + "_images"] = len(ids)
    write_json(output / "split.json", summary)
    return summary
