"""Read-only report of the currently selected Python/GPU environment."""
import importlib.util
import json
import platform
import sys

report = {"python": sys.version, "platform": platform.platform()}
for name in ("numpy", "PIL", "torch", "mmcv", "mmdet", "mmocr"):
    report[name] = importlib.util.find_spec(name) is not None
if report["torch"]:
    import torch
    report["torch_version"] = torch.__version__
    report["cuda_build"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    report["gpus"] = [dict(name=torch.cuda.get_device_name(i),
                           capability=torch.cuda.get_device_capability(i))
                      for i in range(torch.cuda.device_count())]
print(json.dumps(report, indent=2))
