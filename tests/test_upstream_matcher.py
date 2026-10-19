"""Real upstream matcher on gold/empty/swapped dev predictions, no model."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "work/qa_deps")]
from extremeocr.dev_evaluation import detection_zip, parsed_metrics, prepare_dev


@unittest.skipUnless(importlib.util.find_spec("Levenshtein"), "Matcher QA requires local work/qa_deps or baseline runtime")
class UpstreamMatcherTests(unittest.TestCase):
    def evaluate(self, variant):
        vendor = Path(os.environ.get("EXTREMEOCR_VENDOR", str(ROOT / "work/upstream/LRANet-PP")))
        directory = vendor / "mmocr/core/evaluation/evaluation_e2e"
        sys.path.insert(0, str(directory))
        from text_eval_script import text_eval_main
        polygon = [0, 0, 100, 0, 100, 40, 0, 40]
        texts = ["ROAD", "EXIT"]
        doc = {"extremeocr_split": {"role": "dev"},
               "images": [{"id": 11, "file_name": "a"}, {"id": 99, "file_name": "b"}],
               "annotations": [{"id": i, "image_id": image_id, "transcription": text,
                                "segmentation": [polygon]} for i, image_id, text in [(1, 11, texts[0]), (2, 99, texts[1])]]}
        predictions = [{"boundary_result": [polygon + [0.9]], "strs": [text]} for text in texts]
        if variant == "empty":
            predictions = [{"boundary_result": [], "strs": []}] * 2
        if variant == "swapped":
            predictions.reverse()
        if variant == "ignore":
            doc["annotations"][0]["iscrowd"] = 1
        if variant == "short_numeric":
            doc["annotations"] += [{"id": 3, "image_id": 11, "transcription": "12",
                                    "segmentation": [[200, 0, 220, 0, 220, 40, 200, 40]]}]
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dev"
            gt = prepare_dev(doc, predictions, output)
            try:
                os.chdir(output)
                records = json.loads((output / "my_text_results.json").read_text(encoding="utf-8"))
                detection_zip(records, output / "det.zip", 0.3)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = text_eval_main(str(output / "det.zip"), str(gt), True)
            finally:
                os.chdir(previous)
        metrics = parsed_metrics(result)
        return {"E2E_RESULTS": {"hmean": metrics["e2e-hmean"]},
                "DETECTION_ONLY_RESULTS": {"hmean": metrics["det-hmean"]}}

    def test_gold_predictions(self):
        result = self.evaluate("gold")
        self.assertEqual(result["E2E_RESULTS"]["hmean"], 1)
        self.assertEqual(result["DETECTION_ONLY_RESULTS"]["hmean"], 1)

    def test_empty_predictions(self):
        result = self.evaluate("empty")
        self.assertEqual(result["E2E_RESULTS"]["hmean"], 0)

    def test_wrong_image_recognitions_are_not_perfect(self):
        result = self.evaluate("swapped")
        self.assertEqual(result["E2E_RESULTS"]["hmean"], 0)
        self.assertEqual(result["DETECTION_ONLY_RESULTS"]["hmean"], 1)

    def test_ignored_instance(self):
        self.assertEqual(self.evaluate("ignore")["E2E_RESULTS"]["hmean"], 1)

    def test_numeric_short_word_caresets_differ(self):
        result = self.evaluate("short_numeric")
        self.assertEqual(result["E2E_RESULTS"]["hmean"], 1)
        self.assertLess(result["DETECTION_ONLY_RESULTS"]["hmean"], 1)
