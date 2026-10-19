import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import io

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from extremeocr.adapter import SaturationAdapter
from extremeocr.checkpoint_contract import checked_state, verify_architecture
from extremeocr.dev_evaluation import prepare_dev
from extremeocr.provenance import asset, write_json
from extremeocr.splits import split_training
from extremeocr.threshold_protocol import freeze_threshold
from collect_metrics import verify_inputs
from docker_run import docker_command
from lranet_run import dataset_record, parser, prepare, validate_args
from score_crops import score
from summarize_scores import summarize


def fake_evaluator():
    values = []
    for t in [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.35, 0.4, 0.5]:
        values.append(t)
    return values


class ResearchPipelineTests(unittest.TestCase):
    def test_checkpoint_shapes_and_only_verified_ema_backups_allowed(self):
        expected = {"backbone.weight": torch.zeros(2, 3), "highlight_adapter.weight": torch.zeros(2, 2)}
        ordinary = {"backbone.weight": torch.ones(2, 3)}
        actual = dict(ordinary, ema_backbone_weight=torch.ones(2, 3))
        clean, report = checked_state(expected, actual, allow_new_adapter=True, official_origin=True)
        self.assertTrue(report["passed"])
        self.assertEqual(set(clean), set(ordinary))
        actual["backbone.weight"] = torch.zeros(3, 3)
        self.assertFalse(checked_state(expected, actual, True, True)[1]["passed"])
        actual["backbone.weight"] = torch.ones(2, 3)
        actual["ema_arbitrary"] = torch.zeros(1)
        self.assertFalse(checked_state(expected, actual, True, True)[1]["passed"])
        actual = dict(expected)
        self.assertFalse(checked_state(expected, actual, False, True)[1]["passed"])
        actual = {"backbone.weight": torch.full((2, 3), float("nan"))}
        self.assertFalse(checked_state(expected, actual, True, True)[1]["passed"])

    def test_embedded_checkpoint_architecture_matches_sidecar(self):
        run = {"checkpoint_origin": "experiment", "module": "saturation", "levels": [0, 1, 2, 3],
               "checkpoint_contract": {"prepared_config": {"sha256": "expected"}}}
        checkpoint = {"meta": {"extremeocr_architecture": {
            "module": "saturation", "levels": [0, 1, 2, 3], "prepared_config_sha256": "expected"}}}
        verify_architecture(checkpoint, run)
        checkpoint["meta"]["extremeocr_architecture"]["module"] = "global"
        with self.assertRaises(ValueError):
            verify_architecture(checkpoint, run)

    def test_dev_bridge_explicitly_disables_upstream_default_evaluation(self):
        import importlib.util
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "test.py"
            output = Path(tmp) / "value.json"
            fake.write_text("import argparse, json\nfrom pathlib import Path\ndef parse_args():\n    return argparse.Namespace(eval='hmean-e2e')\ndef main():\n    Path(" + repr(str(output)) + ").write_text(json.dumps({'eval': parse_args().eval}))\n")
            spec = importlib.util.spec_from_file_location("bridge", ROOT / "scripts/predict_dev.py")
            bridge = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bridge)
            with patch.object(sys, "argv", ["predict_dev.py", str(fake)]):
                bridge.main()
            self.assertIsNone(json.loads(output.read_text())["eval"])

    def test_controls_equal_capacity_identity_and_input_dependency(self):
        sizes = []
        features = (torch.randn(1, 8, 4, 8),)
        dark, bright = torch.zeros(1, 3, 8, 16), torch.ones(1, 3, 8, 16)
        for mode in SaturationAdapter.MODES:
            torch.manual_seed(42)
            module = SaturationAdapter(8, 1, 4, conditioning=mode)
            sizes.append(sum(p.numel() for p in module.parameters()))
            self.assertTrue(torch.equal(module(features, bright)[0], features[0]))
            with torch.no_grad():
                module.branches[0][-1].weight.fill_(0.1)
            difference = (module(features, bright)[0] - module(features, dark)[0]).abs().max().item()
            self.assertEqual(difference == 0, mode == "zeros")
        self.assertEqual(len(set(sizes)), 1)

    def test_level_ablation_preserves_unselected_features(self):
        model = SaturationAdapter(8, 2, 4, active_levels=[1])
        features = (torch.randn(1, 8, 4, 8), torch.randn(1, 8, 2, 4))
        with torch.no_grad():
            model.branches[1][-1].bias.fill_(1)
        result = model(features, torch.ones(1, 3, 8, 16))
        self.assertIs(result[0], features[0])
        self.assertTrue(torch.equal(result[1], features[1] + 1))
        with self.assertRaises(ValueError):
            SaturationAdapter(active_levels=[4])

    def test_adapter_trained_state_roundtrip_is_exact(self):
        first = SaturationAdapter(8, 1, 4)
        with torch.no_grad():
            first.branches[0][-1].bias.fill_(0.123)
        buffer = io.BytesIO()
        torch.save(first.state_dict(), buffer)
        buffer.seek(0)
        second = SaturationAdapter(8, 1, 4)
        second.load_state_dict(torch.load(buffer), strict=True)
        features, rgb = (torch.rand(1, 8, 4, 8),), torch.rand(1, 3, 8, 16)
        self.assertTrue(torch.equal(first(features, rgb)[0], second(features, rgb)[0]))

    def test_split_keeps_duplicate_sources_and_explicit_groups_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            images = folder / "images"
            images.mkdir()
            docs = []
            for i in range(6):
                # 0/1 are byte-identical, and 2/3 share an explicit scene.
                value = 20 if i < 2 else 30 + i
                name = str(i) + ".png"
                Image.new("RGB", (10, 10), (value, value, value)).save(images / name)
                docs.append({"id": i, "file_name": name, "width": 10, "height": 10})
            ann = folder / "totaltext_train.json"
            write_json(ann, {"images": docs, "annotations": [], "categories": []})
            split_training(images, ann, folder / "split", 0.5, 42, {"2.png": "scene", "3.png": "scene"})
            train = json.loads((folder / "split/totaltext_train.json").read_text())
            dev = json.loads((folder / "split/totaltext_dev.json").read_text())
            train_ids, dev_ids = [{im["id"] for im in doc["images"]} for doc in (train, dev)]
            self.assertFalse(train_ids & dev_ids)
            for first, second in ((0, 1), (2, 3)):
                self.assertEqual(first in train_ids, second in train_ids)
            self.assertEqual(train_ids | dev_ids, set(range(6)))
            dataset_record(images, folder / "split/totaltext_dev.json", "dev")
            with self.assertRaises(ValueError):
                dataset_record(images, ann, "train")

    def test_official_test_reordering_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            images = [{"id": i, "file_name": "test/" + str(i) + ".jpg", "width": 10, "height": 10}
                      for i in range(300)]
            reference, candidate = folder / "original_totaltext_test.json", folder / "totaltext_test.json"
            write_json(reference, {"images": images, "annotations": []})
            write_json(candidate, {"images": images[::-1], "annotations": []})
            with self.assertRaisesRegex(ValueError, "identity/order"):
                dataset_record(folder, candidate, "test", reference)

    def test_train_defaults_are_bounded_and_validation_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for relative in ("configs/stages.json", "configs/upstream.lock.json"):
                path = folder / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / relative).read_bytes())
            args = parser().parse_args(["train", "--images", "data/missing", "--annotations", "train.json",
                                       "--checkpoint", "official.pth", "--run", "probe", "--module", "residual"])
            with patch("lranet_run.ROOT", folder), patch("lranet_run.audit", return_value={"passed": True}):
                run, meta = prepare(args)
            text = (run / "config.py").read_text()
            self.assertIn("--no-validate", meta["command"])
            self.assertEqual(meta["stage_budget"]["max_iters"], 30)
            self.assertIn("'conditioning': 'zeros'", text)
            self.assertNotIn("lranet_protocol", text)
            self.assertIsNone(meta["dataset"])

    def test_official_protocol_cannot_include_candidate_modules(self):
        args = parser().parse_args(["eval", "--images", "x", "--annotations", "x", "--checkpoint", "x",
                                   "--run", "x", "--module", "saturation", "--protocol", "official-sweep"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_fixed_threshold_changes_only_pinned_loop(self):
        fixed = freeze_threshold(fake_evaluator, 0.3)
        self.assertEqual(fixed(), [0.3])
        self.assertEqual(len(fake_evaluator()), 14)
        for invalid in (float("nan"), 0.01, 1.1):
            with self.assertRaises(ValueError):
                freeze_threshold(fake_evaluator, invalid)

    def test_docker_has_finite_limits_and_no_swap_or_host_namespaces(self):
        limits = json.loads((ROOT / "configs/resource_limits.json").read_text())
        command = docker_command(ROOT, "test:image", limits, ["python", "scripts/preflight.py"])
        self.assertEqual(command[command.index("--memory") + 1], command[command.index("--memory-swap") + 1])
        self.assertIn("--init", command)
        self.assertIn("--pids-limit", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--ipc=host", command)
        self.assertIn("--network=none", command)
        limits["watchdog_memory_gib"] = limits["container_memory_gib"]
        with self.assertRaises(ValueError):
            docker_command(ROOT, "test:image", limits, ["python"])

    def test_changed_run_assets_rejected_at_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"
            path.write_bytes(b"original")
            record = asset(path)
            meta = {key: record for key in ("checkpoint_asset", "basis_asset", "config_asset")}
            meta["dataset"] = {"annotation": record, "images": []}
            meta["images"] = tmp
            verify_inputs(meta)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_inputs(meta)

    def test_dev_gt_order_and_empty_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = {"extremeocr_split": {"role": "dev"},
                   "images": [{"id": 11, "file_name": "a"}, {"id": 99, "file_name": "b"}],
                   "annotations": [{"id": 1, "image_id": 99, "transcription": "ROAD",
                                    "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]}]}
            gt = prepare_dev(doc, [{"boundary_result": [], "strs": []}] * 2, Path(tmp) / "dev")
            with zipfile.ZipFile(gt) as archive:
                self.assertEqual(archive.read("0000000.txt"), b"")
                self.assertIn(b"ROAD", archive.read("0000001.txt"))
            with self.assertRaises(ValueError):
                prepare_dev(doc, [], Path(tmp) / "other")

    def test_crop_scoring_rejects_partial_predictions(self):
        truth = [{"image_id": 1, "annotation_id": 1, "text": "ROAD"}]
        result = score(truth, [{"image_id": 1, "annotation_id": 1, "text": "road"}])
        self.assertEqual(result["word_accuracy"], 1)
        self.assertEqual(result["cer"], 0)
        with self.assertRaises(ValueError):
            score(truth, [])

    def test_score_diagnostic_distinguishes_confidence_loss(self):
        result = summarize([{"diagnostic_detection_scores": [0.9, 0.8],
                             "boundary_result": [[0, 0, 0.4], [0, 0, 0.2]]}], 0.3)
        self.assertEqual(result["pre_rec_kept"], 2)
        self.assertEqual(result["lost_after_recognition_confidence"], 1)


if __name__ == "__main__":
    unittest.main()
