"""Fix the confidence threshold for robustness runs using upstream evaluator.

Upstream sweeps thresholds on every evaluated split. This import changes only
the sweep to a single frozen value, preserving upstream matching/word filtering.
Use the original tools/test.py directly for the paper's sweep protocol.
"""
import importlib
import os

from .threshold_protocol import freeze_threshold

evaluator_module = importlib.import_module("mmocr.core.evaluation.hmean_e2e")
dataset_module = importlib.import_module("mmocr.datasets.icdar_e2e_dataset")


def install():
    threshold = float(os.environ["EXTREMEOCR_SCORE_THRESHOLD"])
    function = evaluator_module.eval_hmean_e2e
    fixed = freeze_threshold(function, threshold)
    evaluator_module.eval_hmean_e2e = fixed
    dataset_module.eval_hmean_e2e = fixed
    print("[extremeocr] fixed confidence threshold =", threshold)


install()
