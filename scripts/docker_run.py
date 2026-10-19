"""Render or explicitly execute a bounded, nonprivileged Docker command."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def docker_command(root, image, limits, command, gpus="all"):
    memory = int(limits["container_memory_gib"])
    if memory <= 0 or not 0 < limits["watchdog_memory_gib"] < memory:
        raise ValueError("Require 0 < watchdog RAM < container RAM")
    if limits["shm_gib"] >= memory or min(limits["cpus"], limits["pids_limit"], limits["shm_gib"]) <= 0:
        raise ValueError("Invalid CPU/PID/shared-memory limits")
    return ["docker", "run", "--rm", "--init", "--restart=no", "--gpus", gpus,
            "--memory", str(memory) + "g", "--memory-swap", str(memory) + "g",
            "--cpus", str(limits["cpus"]), "--pids-limit", str(limits["pids_limit"]),
            "--shm-size", str(limits["shm_gib"]) + "g", "--stop-timeout", "45",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--network=none",
            "--mount", "type=bind,source=" + str(root) + ",target=/workspace",
            "--workdir", "/workspace", "--env", "EXTREMEOCR_CONTAINER=1",
            "--env", "EXTREMEOCR_VENDOR=/opt/lranet",
            "--env", "EXTREMEOCR_WATCHDOG_GIB=" + str(limits["watchdog_memory_gib"]),
            "--env", "EXTREMEOCR_LIMITS_JSON=" + json.dumps(limits, separators=(",", ":")),
            "--env", "PYTHONUNBUFFERED=1", "--env", "OMP_NUM_THREADS=1",
            "--env", "PYTHONPATH=/workspace/src:/opt/lranet", image] + list(command)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="extremeocr:lranet-reference")
    p.add_argument("--limits", default=str(ROOT / "configs/resource_limits.json"))
    p.add_argument("--gpus", default="all")
    p.add_argument("--execute", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    tail = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not tail:
        tail = ["python", "scripts/preflight.py"]
    command = docker_command(ROOT, args.image, json.loads(Path(args.limits).read_text()), tail, args.gpus)
    print(shlex.join(command))
    if args.execute:
        image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
        if not image_id.startswith("sha256:"):
            raise ValueError("Unable to resolve immutable local image id")
        position = command.index(args.image)
        command[position:position + 1] = ["--env", "EXTREMEOCR_IMAGE_ID=" + image_id, image_id]
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
