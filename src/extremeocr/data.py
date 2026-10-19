import copy
import hashlib
import json
from pathlib import Path

from PIL import Image

from .corruptions import corrupt, stable_seed


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_suite(input_root, annotations, output_root, config_path):
    """COCO-style input: transform pixels, keep instance labels and image order.

    file_name alone changes to a PNG path. All variants, including clean, use PNG.
    The annotation basename is retained (upstream selects its evaluator by name).
    """
    input_root, annotations = Path(input_root).resolve(), Path(annotations).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Use a new/empty suite output directory")
    if input_root == output_root or input_root in output_root.parents:
        raise ValueError("Output must live outside the input image tree")
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    doc = json.loads(annotations.read_text(encoding="utf-8"))
    variants = [("clean", 0)] if cfg.get("include_clean", True) else []
    variants += [(k, s) for k in cfg["kinds"] for s in cfg["severities"]]
    names = [str(Path(item["file_name"]).with_suffix(".png")).lower()
             for item in doc["images"]]
    if len(names) != len(set(names)):
        raise ValueError("Image filenames collide after conversion to PNG")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, severity in variants:
        tag = "clean" if kind == "clean" else kind + "/s" + str(severity)
        target_root = output_root / tag
        derived = copy.deepcopy(doc)
        for item, original in zip(derived["images"], doc["images"]):
            rel = Path(original["file_name"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("file_name must be a relative dataset path: " + str(rel))
            source = (input_root / rel).resolve()
            source.relative_to(input_root)
            dest_rel = rel.with_suffix(".png")
            dest = target_root / "images" / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as im:
                if im.size != (original["width"], original["height"]):
                    raise ValueError("Annotation/image size mismatch: " + str(rel))
                out, meta = corrupt(im, kind, severity,
                                    stable_seed(cfg["seed"], rel.as_posix() + ":" + kind))
                out.save(dest, format="PNG")
            item["file_name"] = dest_rel.as_posix()
            rows.append(dict(meta, image_id=original["id"], source=rel.as_posix(),
                             output=dest.relative_to(output_root).as_posix(),
                             source_sha256=sha256(source), output_sha256=sha256(dest)))
        (target_root / annotations.name).write_text(
            json.dumps(derived, ensure_ascii=False), encoding="utf-8")
    (output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (output_root / "protocol.json").write_text(json.dumps({
        "config": cfg, "config_sha256": sha256(config_path),
        "annotation_sha256": sha256(annotations), "images": len(doc["images"]),
        "variants": len(variants), "records": len(rows),
        "label_policy": "Instance labels unchanged; review readability separately."
    }, indent=2), encoding="utf-8")
    return {"images": len(doc["images"]), "variants": len(variants), "records": len(rows)}
