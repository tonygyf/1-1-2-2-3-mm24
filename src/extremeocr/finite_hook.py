"""Fail a research run promptly on nonfinite loss (no clipping/mutation)."""
import torch
import os
import json
from pathlib import Path
from mmcv.runner import HOOKS, Hook
from .provenance import write_json


@HOOKS.register_module()
class FiniteLossHook(Hook):
    def after_train_iter(self, runner):
        loss = runner.outputs.get("loss")
        if loss is None or not torch.isfinite(loss.detach()).all().item():
            raise FloatingPointError("Nonfinite/missing loss at iter {}".format(runner.iter))


@HOOKS.register_module()
class AdapterHealthHook(Hook):
    def __init__(self, interval=20):
        self.interval = interval

    def before_run(self, runner):
        run = Path(os.environ["EXTREMEOCR_RUN_DIR"])
        meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
        if runner.meta is None:
            runner.meta = {}
        runner.meta["extremeocr_architecture"] = {
            "module": meta["module"], "levels": meta["levels"],
            "prepared_config_sha256": meta["config_asset"]["sha256"]}
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        self.adapter = getattr(model, "highlight_adapter", None)
        self.initial = ({name: value.detach().cpu().clone() for name, value in self.adapter.named_parameters()}
                        if self.adapter is not None else {})

    def after_train_iter(self, runner):
        final = runner.iter + 1 == runner.max_iters
        if self.adapter is None or (not final and (runner.iter + 1) % self.interval):
            return
        gradients = [p.grad for p in self.adapter.parameters() if p.grad is not None]
        finite = all(torch.isfinite(grad).all().item() for grad in gradients)
        nonzero = sum(grad.abs().sum().item() > 0 for grad in gradients)
        change = max((value.detach().cpu() - self.initial[name]).abs().max().item()
                     for name, value in self.adapter.named_parameters())
        report = {"iteration": runner.iter + 1, "finite_gradients": finite,
                  "nonzero_gradient_tensors": nonzero, "max_parameter_change": change,
                  "passed": finite and nonzero > 0 and change > 0}
        write_json(Path(os.environ["EXTREMEOCR_RUN_DIR"]) /
                   ("adapter_health_rank" + os.environ.get("RANK", "0") + ".json"), report)
        if not finite or (final and not report["passed"]):
            raise RuntimeError("Adapter failed gradient/update health check: " + repr(report))
