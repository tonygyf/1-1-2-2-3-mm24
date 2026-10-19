"""Prepare GT in upstream Total-Text matcher format for a project dev split.

Never overwrites official GT. The same matcher is reused but this generated-GT
protocol is separately named and is not a claim of official test reproduction.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from .provenance import write_json


def gt_lines(annotations):
    from shapely.geometry import Polygon
    lines = []
    for ann in annotations:
        parts = ann.get("segmentation", [])
        if not isinstance(parts, list) or len(parts) != 1:
            raise ValueError("Dev GT requires exactly one polygon per instance")
        flat = parts[0]
        if len(flat) < 8 or len(flat) % 2 or not all(math.isfinite(x) for x in flat):
            raise ValueError("Invalid dev GT polygon")
        points = np.asarray(flat, dtype=float).reshape(-1, 2).astype(int)
        # Match upstream's Cartesian LinearRing.is_ccw == False convention,
        # even though image coordinates have a downward-positive y axis.
        area = np.sum(points[:, 0] * np.roll(points[:, 1], -1) -
                      np.roll(points[:, 0], -1) * points[:, 1])
        if area == 0:
            raise ValueError("Degenerate dev GT polygon")
        if area > 0:
            points = points[::-1]
        if not Polygon(points).is_valid:
            raise ValueError("Invalid dev GT polygon; inspect source labels")
        text = ann.get("transcription")
        if not isinstance(text, str) or any(token in text for token in ("\n", "\r", ",####")):
            raise ValueError("Unsupported dev transcription; inspect source annotation")
        if ann.get("ignore") or ann.get("iscrowd"):
            text = "###"
        lines.append(",".join(str(x) for x in points.flatten()) + ",####" + text + "\n")
    return "".join(lines)


def prepare_dev(doc, predictions, output):
    if doc.get("extremeocr_split", {}).get("role") != "dev":
        raise ValueError("Generated GT evaluation is restricted to explicit dev splits")
    if len(predictions) != len(doc["images"]):
        raise ValueError("Prediction count differs from JSON image order")
    by_id = defaultdict(list)
    for ann in doc["annotations"]:
        by_id[ann["image_id"]].append(ann)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, order = [], []
    with zipfile.ZipFile(output / "gt_dev.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for index, (image, result) in enumerate(zip(doc["images"], predictions)):
            archive.writestr("{:07d}.txt".format(index), gt_lines(by_id[image["id"]]))
            order.append({"index": index, "image_id": image["id"], "file_name": image["file_name"]})
            boundaries, texts = result["boundary_result"], result.get("strs", [])
            if len(boundaries) != len(texts):
                raise ValueError("Boundary/recognition count mismatch")
            for boundary, text in zip(boundaries, texts):
                records.append({"image_id": index, "category_id": 1,
                                "polys": np.asarray(boundary[:-1]).reshape(-1, 2).tolist(),
                                "score": float(boundary[-1]), "rec": text})
    write_json(output / "order.json", order)
    write_json(output / "my_text_results.json", records)
    return output / "gt_dev.zip"


def detection_zip(records, output, threshold):
    """Portable equivalent of upstream integer/score/winding serialization.

    Matching remains the untouched upstream text_eval_script. Upstream's path
    replacement in sort_detection assumes POSIX separators and loses results on
    Windows, so project dev serialization avoids its temporary directory dance.
    """
    from shapely.geometry import LinearRing, Polygon
    by_image = defaultdict(list)
    for row in records:
        score = row["score"]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid prediction score")
        if score <= 0.1 or round(score, 3) < threshold:
            continue
        points = np.asarray(row["polys"]).astype(int).tolist()
        polygon = Polygon(points)
        if not polygon.is_valid:
            continue
        if LinearRing(points).is_ccw:
            points.reverse()
        text = row["rec"].strip()
        if any(token in text for token in ("\n", "\r", ",####")):
            raise ValueError("Unsupported prediction transcription delimiter")
        by_image[row["image_id"]].append(",".join(str(x) for p in points for x in p) + ",####" + text + "\n")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for image_id, lines in by_image.items():
            archive.writestr("{:07d}.txt".format(image_id), "".join(lines))


def parsed_metrics(result):
    result_metrics = {}
    for name, key in (("det-hmean", "det_only_method"), ("e2e-hmean", "e2e_method")):
        match = re.search(r"hmean: ([0-9.eE+-]+)", result[key])
        if not match:
            raise ValueError("Unrecognized upstream metric string")
        value = float(match.group(1))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid Hmean")
        result_metrics[name] = value
    return result_metrics
