"""Exercise software plumbing on a drawn fixture, never report OCR accuracy."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremeocr.corruptions import KINDS, corrupt
from extremeocr.data import build_suite


def main():
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    image = Image.new("RGB", (480, 160), (65, 80, 95))
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 455, 135), fill=(225, 223, 200))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 35)
    except OSError:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 35)
        except OSError:
            font = ImageFont.load_default()
    draw.text((45, 55), "ROAD 26 / EXIT 8", fill=(30, 38, 50), font=font)
    sheet = Image.new("RGB", (480 * 3, 190 * 7 + 50), "#edf1f5")
    sd = ImageDraw.Draw(sheet)
    sd.text((12, 12), "SOFTWARE FIXTURE ONLY - not a research dataset or OCR result", fill="black")
    kinds = ("clean",) + KINDS
    rows = []
    for row, kind in enumerate(kinds):
        for col, severity in enumerate((1, 2, 3)):
            out, meta = corrupt(image, kind, 0 if kind == "clean" else severity, seed=42)
            sheet.paste(out, (col * 480, row * 190 + 70))
            sd.text((col * 480 + 8, row * 190 + 50),
                    kind + " / level " + str(severity), fill="black")
            assert out.size == image.size
            rows.append(meta)
    sheet.save(outputs / "corruption_preview.png")
    # Exercise the same dataset builder as the CLI, including label inheritance.
    with tempfile.TemporaryDirectory(dir=ROOT / "work") as tmp:
        fixture = Path(tmp)
        raw = fixture / "raw"
        raw.mkdir()
        image.save(raw / "sign.png")
        annotation = {"images": [{"id": 1, "file_name": "sign.png", "width": 480, "height": 160}],
                      "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                       "bbox": [25, 25, 430, 110],
                                       "segmentation": [[25, 25, 455, 25, 455, 135, 25, 135]],
                                       "transcription": "ROAD 26 / EXIT 8", "area": 47300,
                                       "iscrowd": 0}],
                      "categories": [{"id": 1, "name": "text"}]}
        ann_path = fixture / "totaltext_fixture.json"
        ann_path.write_text(json.dumps(annotation), encoding="utf-8")
        built = build_suite(raw, ann_path, fixture / "suite", ROOT / "configs/screen.json")
        assert built["records"] == 19
        for path in (fixture / "suite").rglob(ann_path.name):
            assert json.loads(path.read_text(encoding="utf-8"))["annotations"] == annotation["annotations"]
    import torch
    from extremeocr.adapter import SaturationAdapter
    model = SaturationAdapter()
    features = tuple(torch.randn(1, 256, h, w) for h, w in [(20, 60), (10, 30), (5, 15), (3, 8)])
    rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
    result = model(features, rgb)
    assert all(torch.equal(a, b) for a, b in zip(features, result))
    sum(x.square().mean() for x in result).backward()
    status = {"software_smoke": "passed", "corruption_variants": 19,
              "dataset_builder": built, "label_inheritance": "passed",
              "adapter_parameters": sum(p.numel() for p in model.parameters()),
              "adapter_forward_backward": "passed", "initial_identity": True,
              "torch": torch.__version__, "device": "cpu",
              "baseline_inference": "not_run", "ocr_metrics": None,
              "fixture": "programmatically drawn sign; not real research data",
              "measurements": rows}
    (outputs / "smoke_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in status.items() if k != "measurements"}, indent=2))


if __name__ == "__main__":
    main()
