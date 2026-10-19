"""Validate full-model key/shape compatibility, including known EMA backups."""
from collections import OrderedDict
import torch


def checked_state(expected, actual, allow_new_adapter=False, official_origin=False):
    missing = sorted(set(expected) - set(actual))
    allowed_missing = lambda key: (allow_new_adapter and key.startswith("highlight_adapter.")) or key.endswith("num_batches_tracked")
    invalid_missing = [key for key in missing if not allowed_missing(key)]
    mapping = {"ema_" + key.replace(".", "_"): key for key in expected}
    if len(mapping) != len(expected):
        raise ValueError("Ambiguous EMA buffer mapping")
    extras = set(actual) - set(expected)
    ema_backups = sorted(key for key in extras if key in mapping
                         and tuple(actual[key].shape) == tuple(expected[mapping[key]].shape))
    unexpected = sorted(extras - set(ema_backups))
    mismatch = sorted(key for key in set(actual) & set(expected)
                      if tuple(actual[key].shape) != tuple(expected[key].shape))
    nonfinite = sorted(key for key in set(actual) & set(expected)
                       if actual[key].is_floating_point() and not torch.isfinite(actual[key]).all().item())
    if official_origin and any(key.startswith("highlight_adapter.") for key in actual):
        unexpected.append("adapter weights require an experiment checkpoint_contract.json")
    report = {"missing": missing, "unexpected": unexpected, "shape_mismatch": mismatch,
              "nonfinite_weights": nonfinite,
              "verified_ema_backups_ignored": ema_backups,
              "weight_policy": "ordinary parameter keys exactly as saved, matching upstream test loading",
              "passed": not invalid_missing and not unexpected and not mismatch and not nonfinite}
    clean = OrderedDict((key, value) for key, value in actual.items() if key not in ema_backups)
    if hasattr(actual, "_metadata"):
        clean._metadata = actual._metadata
    return clean, report


def verify_architecture(checkpoint, run):
    architecture = checkpoint.get("meta", {}).get("extremeocr_architecture")
    if run["checkpoint_origin"] == "official":
        if architecture is not None:
            raise ValueError("Project checkpoint requires its experiment contract")
        return
    contract = run["checkpoint_contract"]
    expected = {"module": run["module"], "levels": run["levels"],
                "prepared_config_sha256": contract["prepared_config"]["sha256"]}
    if architecture != expected:
        raise ValueError("Embedded checkpoint architecture/source differs from its sidecar")
