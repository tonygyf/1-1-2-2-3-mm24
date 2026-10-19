"""Constrain only the pinned upstream threshold loop, with explicit guards."""
import ast
import inspect
import math
import textwrap

EXPECTED_THRESHOLDS = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22,
                       0.24, 0.26, 0.28, 0.3, 0.35, 0.4, 0.5]


def freeze_threshold(function, threshold):
    if not math.isfinite(threshold) or not 0.1 <= threshold <= 1:
        raise ValueError("Threshold must be finite and >= upstream prefilter 0.1")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "t"]
    if len(loops) != 1 or ast.literal_eval(loops[0].iter) != EXPECTED_THRESHOLDS:
        raise RuntimeError("Upstream evaluator threshold structure changed")
    loops[0].iter = ast.List(elts=[ast.Constant(value=threshold)], ctx=ast.Load())
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, "<extremeocr-fixed-threshold>", "exec"), namespace)
    return namespace[function.__name__]
