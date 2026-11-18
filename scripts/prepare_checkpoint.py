"""Canonicalize this author's known release mismatch without relaxing model checks."""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremeocr.checkpoint_contract import checked_state
from extremeocr.provenance import asset, write_json


def canonical_state(expected, actual, parameter_names, policy):
    missing = set(expected) - set(actual)
    extras = set(actual) - set(expected)
    if missing != set(policy["missing_constant_buffers"]) or extras != set(policy["unused_checkpoint_keys"]):
        raise ValueError("Checkpoint mismatch differs from the reviewed release policy")
    if missing & set(parameter_names):
        raise ValueError("A trained parameter is missing; constant-buffer policy cannot cover it")
    result = OrderedDict((name, actual[name] if name in actual else value.detach().cpu())
                         for name, value in expected.items())
    result, report = checked_state(expected, result, official_origin=True)
    if not report["passed"]:
        raise ValueError("Canonicalized checkpoint still fails strict validation: " + repr(report))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", default="checkpoints/upstream/lranet_pp/totaltext.pth")
    parser.add_argument("--output", default="checkpoints/upstream/lranet_pp/totaltext_runtime.pth")
    args = parser.parse_args()
    if os.environ.get("EXTREMEOCR_CONTAINER") != "1":
        raise SystemExit("Run inside Docker")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing checkpoint")
    policy_path = ROOT / "configs/checkpoint_compatibility.json"
    policy = json.loads(policy_path.read_text())
    original = asset(args.input)
    if original["sha256"] != policy["source_sha256"]:
        raise ValueError("Only the reviewed author checkpoint is covered by this policy")
    from audit_baseline import audit
    source_report = audit()
    if not source_report["passed"]:
        raise ValueError("Runtime source audit failed")
    import torch
    import mmcv
    import mmocr.models
    from mmdet.models import build_detector
    vendor = Path(os.environ["EXTREMEOCR_VENDOR"])
    cfg = mmcv.Config.fromfile(str(vendor / "configs/lranet_pp/lranet_pp_totaltext.py"))
    cfg.model.backbone.init_cfg = None
    basis = str(vendor / "orthanchors/syntext_14.npz")
    cfg.model.bbox_head.path_lra = basis
    cfg.model.bbox_head.loss.path_lra = basis
    cfg.model.recog_head.path_lra = basis
    torch.manual_seed(42)
    model = build_detector(cfg.model)
    actual = torch.load(args.input, map_location="cpu", weights_only=True)["state_dict"]
    clean = canonical_state(model.state_dict(), actual, dict(model.named_parameters()), policy)
    report = {"source_checkpoint": original, "compatibility_policy": asset(policy_path),
              "source_audit": source_report, "missing_constant_buffers_initialized": policy["missing_constant_buffers"],
              "unused_checkpoint_keys_omitted": policy["unused_checkpoint_keys"],
              "semantics": policy["semantics"], "training_profile": policy["training_profile"],
              "retained_tensor_count": len(set(actual) & set(clean)),
              "retained_tensors_modified": 0, "seed": 42, "training_budget": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": clean, "meta": {"extremeocr_canonicalization": report}}, output)
    report["runtime_checkpoint"] = asset(output)
    write_json(output.with_name(output.name + ".source.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
