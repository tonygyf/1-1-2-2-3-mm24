"""Read-only report of the currently selected Python/GPU environment."""
import importlib.util
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata
import json
import platform
import sys

report = {"python": sys.version, "platform": platform.platform()}
report["distribution_versions"] = {}
for distribution in ("torch", "torchvision", "mmcv-full", "mmdet", "mmocr", "numpy", "rapidfuzz", "Levenshtein"):
    try:
        report["distribution_versions"][distribution] = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        report["distribution_versions"][distribution] = None
for name in ("numpy", "PIL", "torch", "mmcv", "mmdet", "mmocr"):
    report[name] = importlib.util.find_spec(name) is not None
if report["torch"]:
    import torch
    report["torch_version"] = torch.__version__
    report["cuda_build"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    report["gpus"] = [dict(name=torch.cuda.get_device_name(i),
                           capability=torch.cuda.get_device_capability(i),
                           total_memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                      for i in range(torch.cuda.device_count())]
    report["torch_compiled_arches"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
report["full_model_and_extensions"] = "not_tested_by_preflight; run server_gate.py explicitly"
print(json.dumps(report, indent=2))
