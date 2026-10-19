"""Prepare auditable LRANet runs; execution is explicit and container-guarded."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremeocr.data import sha256
from extremeocr.provenance import asset, environment, utc_now, write_json
from audit_baseline import audit

MODULES = {"none": None, "residual": "zeros", "saturation": "saturation",
           "global": "mean", "flipped": "flip"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["eval", "train"])
    p.add_argument("--images", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--initialization-note", default="Official checkpoint training-data exposure must be verified; Total-Text finetuned weights may have seen clean dev sources")
    p.add_argument("--run", required=True)
    p.add_argument("--module", choices=MODULES, default="none")
    p.add_argument("--adapter", action="store_true", help="Alias for --module saturation")
    p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--stage", choices=["probe", "pilot", "full"], default="probe")
    p.add_argument("--split", choices=["train", "dev", "test"])
    p.add_argument("--protocol", choices=["fixed", "official-sweep"], default="fixed")
    p.add_argument("--official-annotations", help="Unmodified official full test JSON for ordered identity check")
    p.add_argument("--score-probe", action="store_true", help="Record aligned pre/post recognition scores")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--execute", action="store_true")
    return p


def validate_args(a):
    a.split = a.split or ("train" if a.mode == "train" else "test")
    if a.adapter:
        if a.module not in ("none", "saturation"):
            raise ValueError("--adapter conflicts with --module")
        a.module = "saturation"
    if a.mode == "train" and (a.split != "train" or a.protocol != "fixed"):
        raise ValueError("Research training requires train/fixed; official training uses upstream commands")
    if a.mode == "eval" and a.split == "train":
        raise ValueError("Use dev or test for evaluation")
    if a.protocol == "official-sweep" and (a.mode != "eval" or a.module != "none" or a.split != "test"):
        raise ValueError("Official sweep is only for baseline full-test reproduction")
    if a.score_probe and (a.mode != "eval" or a.protocol != "fixed"):
        raise ValueError("Score probe is restricted to fixed-protocol evaluation")
    if not 0.1 <= a.score_threshold <= 1 or not 1 <= a.gpus <= 4:
        raise ValueError("Expected threshold in [0.1,1] and 1..4 GPUs")
    if not a.levels or len(set(a.levels)) != len(a.levels) or any(i not in range(4) for i in a.levels):
        raise ValueError("Select distinct FPN indices 0..3")


def dataset_record(images, annotation, role, reference=None):
    doc = json.loads(annotation.read_text(encoding="utf-8"))
    ids = [image["id"] for image in doc["images"]]
    id_set = set(ids)
    annotation_ids = [instance["id"] for instance in doc["annotations"]]
    if not ids or len(set(ids)) != len(ids) or len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("Dataset must be nonempty with unique image/annotation ids")
    if any(instance["image_id"] not in id_set for instance in doc["annotations"]):
        raise ValueError("Annotation references an unknown image")
    declared = doc.get("extremeocr_split", {}).get("role")
    if role in ("train", "dev") and declared != role:
        raise ValueError("Research train/dev must come from split_training.py before corruption/mixing")
    if role == "test":
        if reference is None:
            raise ValueError("Full-test evaluator requires --official-annotations to verify order")
        official = json.loads(reference.read_text(encoding="utf-8"))
        def identity(value):
            return [(im["id"], Path(im["file_name"]).with_suffix("").as_posix(),
                     im["width"], im["height"]) for im in value["images"]]
        if len(official["images"]) != 300 or identity(doc) != identity(official):
            raise ValueError("Official Total-Text test identity/order differs (expected 300)")
        if doc["annotations"] != official["annotations"]:
            raise ValueError("Official test labels changed")
        if "totaltext" not in str(annotation).lower() or "ctw" in str(annotation).lower():
            raise ValueError("Upstream dispatch requires unambiguous totaltext annotation path")
    rows = []
    for im in doc["images"]:
        path = (images / im["file_name"]).resolve()
        path.relative_to(images)
        with Image.open(path) as decoded:
            if decoded.size != (im["width"], im["height"]):
                raise ValueError("Image dimensions differ from annotation: " + str(path))
            decoded.verify()
        rows.append({"image_id": im["id"], "file_name": im["file_name"], "sha256": sha256(path)})
    return {"images": rows, "annotation": asset(annotation), "split": doc.get("extremeocr_split"),
            "official_reference": asset(reference) if reference else None}


def prepare(a):
    validate_args(a)
    vendor = Path(os.environ.get("EXTREMEOCR_VENDOR", str(ROOT / "work/upstream/LRANet-PP"))).resolve()
    run_root = (ROOT / "work/runs").resolve()
    run = (run_root / a.run).resolve()
    run.relative_to(run_root)
    if run == run_root or run.exists():
        raise ValueError("Choose a new named run directory")
    images, ann, ckpt = [Path(x).resolve() for x in (a.images, a.annotations, a.checkpoint)]
    a.images, a.annotations, a.checkpoint = map(str, (images, ann, ckpt))
    reference = Path(a.official_annotations).resolve() if a.official_annotations else None
    if a.execute and os.environ.get("EXTREMEOCR_CONTAINER") != "1":
        raise ValueError("Execute inside the limited Docker container; prepare locally without --execute")
    source = audit()
    if not source["passed"]:
        raise ValueError("Upstream source changed: " + repr(source["changed_or_missing"]))
    data = dataset_record(images, ann, a.split, reference) if a.execute else None
    if a.execute and not ckpt.is_file():
        raise ValueError("Checkpoint does not exist")
    contract_path = ckpt.parent / "checkpoint_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else None
    if contract and (contract["module"] != a.module or contract["levels"] != a.levels):
        raise ValueError("Checkpoint module/level contract differs from requested architecture")
    stage = json.loads((ROOT / "configs/stages.json").read_text(encoding="utf-8"))[a.stage]
    basis = vendor / "orthanchors/syntext_14.npz"
    imports = ["extremeocr.checkpoint_guard"]
    if a.score_probe:
        imports.append("extremeocr.score_probe")
    if a.module != "none":
        imports.append("extremeocr.lranet_plugin")
    if a.mode == "eval" and a.split == "test" and a.protocol == "fixed":
        imports.append("extremeocr.lranet_protocol")
    if a.mode == "train":
        imports.append("extremeocr.finite_hook")
    lines = ["_base_ = " + repr((vendor / "configs/lranet_pp/lranet_pp_totaltext.py").as_posix()),
             "custom_imports = " + repr(dict(imports=imports, allow_failed_imports=False))]
    model = {"bbox_head": {"path_lra": basis.as_posix(), "loss": {"path_lra": basis.as_posix()}},
             "recog_head": {"path_lra": basis.as_posix()}}
    if a.module != "none":
        model.update(type="HighlightLRANetPP", conditioning=MODULES[a.module], active_levels=a.levels)
    lines.append("model = " + repr(model))
    if a.mode == "eval":
        lines.append("data = " + repr(dict(test=dict(ann_file=ann.as_posix(), img_prefix=images.as_posix()))))
        command = [sys.executable, str(vendor / "tools/test.py"), str(run / "config.py"), str(ckpt),
                   "--out", str(run / "predictions.pkl")]
        if a.split == "dev":
            command.insert(1, str(ROOT / "scripts/predict_dev.py"))
        if a.split == "test":
            command += ["--eval", "hmean-e2e"]
    else:
        # A complete official checkpoint supplies every backbone tensor. Avoid
        # a redundant torchvision download inside the offline container.
        lines.append("model['backbone'] = dict(init_cfg=None)")
        lines += ["data = " + repr(dict(train=dict(dataset=dict(ann_file=ann.as_posix(), img_prefix=images.as_posix())))),
                  "runner = " + repr(dict(type="IterBasedRunner", max_iters=stage["max_iters"])),
                  "lr_config = " + repr(dict(policy="step", step=stage["lr_steps"], by_epoch=False)),
                  "checkpoint_config = " + repr(dict(interval=stage["checkpoint_interval"], by_epoch=False)),
                  "log_config = dict(interval=" + str(stage["log_interval"]) + ")",
                  "custom_hooks = [dict(type='ExpMomentumEMAHook', momentum=0.0002, priority=49), dict(type='FiniteLossHook', priority=10), dict(type='AdapterHealthHook', priority=60, interval=" + str(1 if a.stage == "probe" else 20) + ")]",
                  "load_from = " + repr(ckpt.as_posix()), "seed = " + str(a.seed)]
        command = [sys.executable]
        if a.gpus > 1:
            command += ["-m", "torch.distributed.launch", "--nproc_per_node=" + str(a.gpus),
                        "--master_port=" + os.environ.get("PORT", "29500")]
        command += [str(vendor / "tools/train.py"), str(run / "config.py"),
                    "--work-dir", str(run / "training"), "--load-from", str(ckpt),
                    "--seed", str(a.seed), "--no-validate"]
        if a.gpus > 1:
            command += ["--launcher", "pytorch"]
    run.mkdir(parents=True)
    (run / "config.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = dict(vars(a), vendor=str(vendor), command=command,
                  cwd=str(run / "runtime"), status="prepared", created_at=utc_now(),
                  stage_budget=stage if a.mode == "train" else None,
                  evaluation_protocol=("project-dev-upstream-word-spotting-matcher" if a.split == "dev" else a.protocol),
                  checkpoint_origin="experiment" if contract else "official",
                  checkpoint_contract=contract,
                  source=source, dataset=data, checkpoint_asset=asset(ckpt), basis_asset=asset(basis),
                  config_asset=asset(run / "config.py"), environment=environment(ROOT), adapter=a.module != "none")
    write_json(run / "run.json", record)
    if a.mode == "train":
        write_json(run / "training/checkpoint_contract.json", {
            "module": a.module, "levels": a.levels, "stage": a.stage, "seed": a.seed,
            "initialization": asset(ckpt), "prepared_config": asset(run / "config.py"), "budget": stage,
            "ema": "upstream hook unchanged; same last-artifact policy across methods"})
    return run, record


def execute(run, record):
    runtime = run / "runtime"
    # Run project imports from a snapshot so later workspace edits cannot change
    # one DDP worker or evaluation relative to its siblings.
    snapshot = run / "project"
    for folder in ("src", "scripts", "configs"):
        shutil.copytree(ROOT / folder, snapshot / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    record["project_snapshot"] = {path.relative_to(snapshot).as_posix(): sha256(path)
                                  for path in snapshot.rglob("*") if path.is_file()}
    record["image_id"] = os.environ.get("EXTREMEOCR_IMAGE_ID", "unrecorded-direct-container-invocation")
    record["command"] = [str(snapshot / "scripts/predict_dev.py") if value == str(ROOT / "scripts/predict_dev.py") else value
                         for value in record["command"]]
    shutil.copytree(Path(record["vendor"]) / "mmocr/core/evaluation/evaluation_e2e",
                    runtime / "mmocr/core/evaluation/evaluation_e2e")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(snapshot / "src"), record["vendor"], env.get("PYTHONPATH", "")])
    env["EXTREMEOCR_SCORE_THRESHOLD"] = str(record["score_threshold"])
    env["EXTREMEOCR_RUN_DIR"] = str(run)
    record.update(status="running", started_at=utc_now())
    write_json(run / "run.json", record)
    try:
        with (run / "console.log").open("w", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, str(snapshot / "scripts/resource_guard.py"),
                                    "--run-dir", str(run), "--", sys.executable,
                                    str(snapshot / "scripts/run_worker.py"), str(run)],
                                   cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(status="completed" if code == 0 else "failed", returncode=code)
        return code
    except BaseException as exc:
        record.update(status="interrupted", error=type(exc).__name__)
        raise
    finally:
        record["finished_at"] = utc_now()
        write_json(run / "run.json", record)


def main():
    args = parser().parse_args()
    run, record = prepare(args)
    print(json.dumps({"run_dir": str(run), "command": record["command"],
                      "status": record["status"], "module": record["module"]}, indent=2))
    if args.execute:
        raise SystemExit(execute(run, record))


if __name__ == "__main__":
    main()
