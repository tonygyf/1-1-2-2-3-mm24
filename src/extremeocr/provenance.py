"""Small, dependency-free records shared by preparation and execution."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

from .data import sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment(root):
    def read(command):
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "recorded_at": utc_now(),
            "git_commit": read(["git", "rev-parse", "HEAD"]),
            "git_dirty": read(["git", "status", "--porcelain"]) != ""}


def asset(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path) if path.is_file() else None}
