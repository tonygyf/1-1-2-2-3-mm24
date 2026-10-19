"""Make upstream non-strict loading fail on unapproved architecture mismatch."""
import json
import os
from pathlib import Path

import mmcv.runner.checkpoint as checkpoint_module

from .provenance import write_json
from .checkpoint_contract import checked_state, verify_architecture


def install():
    run = Path(os.environ["EXTREMEOCR_RUN_DIR"])
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    original = checkpoint_module.load_state_dict
    original_load = checkpoint_module._load_checkpoint

    def load_checked(*args, **kwargs):
        checkpoint = original_load(*args, **kwargs)
        verify_architecture(checkpoint, meta)
        return checkpoint

    def checked(module, state_dict, strict=False, logger=None):
        target = module.module if hasattr(module, "module") else module
        # Backbone init_cfg can load torchvision weights before full-model load.
        if not hasattr(target, "recog_head") or not hasattr(target, "bbox_head"):
            return original(module, state_dict, strict=strict, logger=logger)
        allowed = meta["checkpoint_origin"] == "official" and meta["module"] != "none"
        clean, report = checked_state(target.state_dict(), state_dict, allowed,
                                      meta["checkpoint_origin"] == "official")
        write_json(run / ("checkpoint_keys_rank" + os.environ.get("RANK", "0") + ".json"), report)
        if not report["passed"]:
            raise RuntimeError("Checkpoint architecture mismatch: " + repr(report))
        return original(module, clean, strict=strict, logger=logger)

    checkpoint_module.load_state_dict = checked
    checkpoint_module._load_checkpoint = load_checked


install()
