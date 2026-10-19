"""Fix the confidence threshold for robustness runs using upstream evaluator.

Upstream sweeps thresholds on every evaluated split. This import changes only
the sweep to a single frozen value, preserving upstream matching/word filtering.
Use the original tools/test.py directly for the paper's sweep protocol.
"""
import ast
import inspect
import os
import textwrap

import mmocr.core.evaluation.hmean_e2e as evaluator_module
import mmocr.datasets.icdar_e2e_dataset as dataset_module


def install():
    threshold = float(os.environ["EXTREMEOCR_SCORE_THRESHOLD"])
    if not 0.1 <= threshold <= 1.0:
        raise ValueError("Threshold must be >=0.1: upstream prefilters at 0.1")
    function = evaluator_module.eval_hmean_e2e
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "t":
            node.iter = ast.List(elts=[ast.Constant(value=threshold)], ctx=ast.Load())
            replacements += 1
    if replacements != 1:
        raise RuntimeError("Upstream evaluator changed; inspect threshold integration")
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, "<extremeocr-fixed-threshold>", "exec"), namespace)
    fixed = namespace[function.__name__]
    evaluator_module.eval_hmean_e2e = fixed
    dataset_module.eval_hmean_e2e = fixed
    print("[extremeocr] fixed confidence threshold =", threshold)


install()
