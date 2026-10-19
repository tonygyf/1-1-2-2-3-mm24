"""Use upstream inference with its default test evaluation explicitly disabled.

Upstream --eval defaults to hmean-e2e even when the option is absent. This small
bridge changes that parsed option only; model, loader and prediction code remain
the pinned upstream functions. Project dev evaluation follows in run_worker.py.
"""
import importlib.util
from pathlib import Path
import sys


def main():
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script)] + sys.argv[2:]
    spec = importlib.util.spec_from_file_location("extremeocr_upstream_predict", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.parse_args
    def parse_without_eval():
        args = original()
        args.eval = None
        return args
    module.parse_args = parse_without_eval
    module.main()


if __name__ == "__main__":
    main()
