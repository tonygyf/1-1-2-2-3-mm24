import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from extremeocr.corruptions import KINDS, corrupt
from extremeocr.data import build_suite
from collect_metrics import parse_metrics
from mix_training import mix


class ProtocolTests(unittest.TestCase):
    def test_corruptions_are_deterministic_and_keep_geometry(self):
        image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (40, 100, 3), dtype=np.uint8))
        for kind in KINDS:
            for severity in (1, 2, 3):
                first, meta = corrupt(image, kind, severity, seed=7)
                second, _ = corrupt(image, kind, severity, seed=7)
                self.assertEqual(image.size, first.size)
                np.testing.assert_array_equal(first, second)
                self.assertTrue(0 <= meta["all_channel_clip_fraction"] <= 1)
        clean, _ = corrupt(image, "clean", 0)
        np.testing.assert_array_equal(clean, image)

    def test_glare_location_is_shared_across_levels(self):
        image = Image.new("RGB", (100, 50), (100, 100, 100))
        _, first = corrupt(image, "glare", 1, 42)
        _, third = corrupt(image, "glare", 3, 42)
        self.assertEqual(first["parameters"]["center_xy_normalized"],
                         third["parameters"]["center_xy_normalized"])

    def test_suite_labels_and_mixed_training_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            images = base / "raw"
            (images / "test").mkdir(parents=True)
            Image.new("RGB", (50, 20), (70, 80, 90)).save(images / "test/a.jpg")
            annotation = {"images": [{"id": 19, "file_name": "test/a.jpg", "width": 50, "height": 20}],
                          "annotations": [{"id": 44, "image_id": 19, "transcription": "ABC",
                                           "bbox": [1, 2, 20, 10], "category_id": 1}],
                          "categories": [{"id": 1, "name": "text"}]}
            ann = base / "totaltext_test.json"
            ann.write_text(json.dumps(annotation))
            suite = base / "derived"
            result = build_suite(images, ann, suite, ROOT / "configs/screen.json")
            self.assertEqual(result["variants"], 19)
            for path in suite.rglob("totaltext_test.json"):
                doc = json.loads(path.read_text())
                self.assertEqual(doc["annotations"], annotation["annotations"])
                self.assertEqual(doc["images"][0]["id"], 19)
                self.assertTrue((path.parent / "images" / doc["images"][0]["file_name"]).exists())
            mixed = mix(suite, ann.name, ["clean", "glare/s2"])
            self.assertEqual(len({i["id"] for i in mixed["images"]}), 2)
            self.assertEqual([a["image_id"] for a in mixed["annotations"]], [1, 2])
            for image in mixed["images"]:
                self.assertTrue((suite / image["file_name"]).exists())

    def test_metrics_do_not_fill_missing_results(self):
        self.assertEqual(parse_metrics("{'e2e-hmean': 0.8, 'det-hmean': 0.9}")["e2e-hmean"], 0.8)
        for log in ("training started", "{'e2e-hmean': 80, 'det-hmean': 90}"):
            with self.assertRaises(ValueError):
                parse_metrics(log)

    def test_adapter_identity_then_trainable(self):
        import torch
        from extremeocr.adapter import SaturationAdapter
        torch.manual_seed(42)
        model = SaturationAdapter(channels=8, levels=2, bottleneck=4)
        features = [torch.randn(2, 8, 8, 16), torch.randn(2, 8, 4, 8)]
        rgb = torch.rand(2, 3, 32, 64)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        first = model(features, rgb)
        for actual, original in zip(first, features):
            self.assertTrue(torch.equal(actual, original))
        loss = sum((out - torch.ones_like(out)).square().mean() for out in first)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        optimizer.step()
        second = model(features, rgb)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(first, second)))


if __name__ == "__main__":
    unittest.main()
