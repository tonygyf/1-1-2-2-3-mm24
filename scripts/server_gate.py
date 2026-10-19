"""Explicit GPU lifecycle gate, without optimizer steps or research metrics."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremeocr.provenance import asset, write_json
from extremeocr.checkpoint_contract import checked_state
from audit_baseline import audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise ValueError("Use a new lifecycle report path")
    if not audit()["passed"]:
        raise ValueError("Official source mismatch")
    import torch
    import mmcv
    from mmdet.models import build_detector
    from extremeocr import lranet_plugin  # registry side effect
    from mmcv.ops import modulated_deform_conv2d
    vendor = Path(os.environ.get("EXTREMEOCR_VENDOR", str(ROOT / "work/upstream/LRANet-PP")))
    cfg = mmcv.Config.fromfile(str(vendor / "configs/lranet_pp/lranet_pp_totaltext.py"))
    basis = str(vendor / "orthanchors/syntext_14.npz")
    cfg.model.backbone.init_cfg = None
    cfg.model.bbox_head.path_lra = basis
    cfg.model.bbox_head.loss.path_lra = basis
    cfg.model.recog_head.path_lra = basis
    torch.manual_seed(42)
    baseline = build_detector(copy.deepcopy(cfg.model)).cuda().eval()
    baseline.init_weights()
    checkpoint = torch.load(a.checkpoint, map_location="cpu")
    if checkpoint.get("meta", {}).get("extremeocr_architecture") is not None:
        raise ValueError("Lifecycle initialization gate requires an official checkpoint")
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module.") if hasattr(key, "removeprefix") else
             (key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    state, key_report = checked_state(baseline.state_dict(), state, official_origin=True)
    if not key_report["passed"]:
        raise ValueError("Official checkpoint mismatch: " + repr(key_report))
    baseline.load_state_dict(state, strict=False)
    cfg.model.type = "HighlightLRANetPP"
    adapted = build_detector(copy.deepcopy(cfg.model)).cuda().eval()
    adapted.init_weights()
    result = adapted.load_state_dict(baseline.state_dict(), strict=False)
    if result.unexpected_keys or any(not key.startswith("highlight_adapter.") for key in result.missing_keys):
        raise ValueError("Unexpected lifecycle checkpoint keys")
    image = torch.zeros(1, 3, 256, 512, device="cuda")
    with torch.no_grad():
        reference, actual = baseline.extract_feat(image), adapted.extract_feat(image)
    if not all(torch.isfinite(value).all().item() for value in tuple(reference) + tuple(actual)):
        raise ValueError("Nonfinite FPN output")
    error = max((x - y).abs().max().item() for x, y in zip(reference, actual))
    if not math.isfinite(error) or error > 1e-6:
        raise ValueError("Adapter initialization changed FPN output")
    # Force both CUDA extension families through forward/backward.
    x = torch.randn(1, 2, 8, 8, device="cuda", requires_grad=True)
    offset = torch.zeros(1, 18, 8, 8, device="cuda")
    mask = torch.ones(1, 9, 8, 8, device="cuda")
    weight = torch.randn(2, 2, 3, 3, device="cuda")
    dcn_output = modulated_deform_conv2d(x, offset, mask, weight, padding=1)
    dcn_output.sum().backward()
    if not torch.isfinite(dcn_output).all().item() or not torch.isfinite(x.grad).all().item():
        raise ValueError("Nonfinite DCNv2 forward/backward")
    from mmocr.models.textend2end.utils.grid_sample_batch.grid_sample_batch import grid_sample_batch
    sampler_input = torch.randn(1, 2, 8, 8, device="cuda", requires_grad=True)
    grid = torch.zeros(1, 2, 3, 2, device="cuda", requires_grad=True)
    sampled = grid_sample_batch(sampler_input, grid, torch.zeros(1, device="cuda"), 0, 0, False)
    raw_sampled = grid_sample_batch(sampler_input, grid.detach(), torch.zeros(1, device="cuda"), 0, 0, None)
    if not torch.equal(raw_sampled, sampled):
        raise ValueError("Upstream align_corners=None differs from False; investigate environment")
    sampled.sum().backward()
    if not torch.isfinite(sampler_input.grad).all().item() or not torch.isfinite(grid.grad).all().item():
        raise ValueError("Custom sampler backward is nonfinite")
    reference_input = sampler_input.detach().clone().requires_grad_(True)
    reference_grid = grid.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.grid_sample(reference_input, reference_grid, align_corners=False)
    expected.sum().backward()
    sampler_error = max((sampled - expected).abs().max().item(),
                        (sampler_input.grad - reference_input.grad).abs().max().item(),
                        (grid.grad - reference_grid.grad).abs().max().item())
    if not math.isfinite(sampler_error) or sampler_error > 1e-5:
        raise ValueError("Custom sampler differs from native forward/backward reference")
    write_json(output, {"checkpoint": asset(a.checkpoint), "fpn_initial_max_abs_error": error,
                        "fpn_identity_passed": True, "dcn_forward_backward_passed": True,
                        "grid_sampler_forward_backward_passed": True, "ddp": "pending probe-training",
                        "grid_sampler_native_max_abs_error": sampler_error,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                        "ocr_metrics": None})
    print(json.dumps({"lifecycle": "passed", "ddp": "pending probe-training"}))


if __name__ == "__main__":
    main()
