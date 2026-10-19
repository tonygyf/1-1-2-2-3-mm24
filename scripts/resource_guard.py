"""Linux cgroup watchdog; terminate only the session launched by this guard."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremeocr.provenance import utc_now, write_json


def memory_files():
    for used, limit in (("memory.current", "memory.max"),
                        ("memory/memory.usage_in_bytes", "memory/memory.limit_in_bytes")):
        current, maximum = Path("/sys/fs/cgroup") / used, Path("/sys/fs/cgroup") / limit
        if current.is_file() and maximum.is_file():
            value = maximum.read_text().strip()
            if value == "max" or int(value) >= 2 ** 60:
                raise RuntimeError("Container must have a finite cgroup memory limit")
            return current, int(value)
    raise RuntimeError("Container cgroup memory accounting is unavailable; refusing unguarded execution")


def terminate_group(pid, grace):
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def supervise(command, usage_path, limits, report_path, log_path=None):
    started = time.monotonic()
    process = None
    reason, peak, over = "completed", 0, 0
    received = []
    log_offset, data_errors = 0, 0
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda number, frame: received.append(number))
    try:
        process = subprocess.Popen(command, start_new_session=True)
        while process.poll() is None:
            used = int(Path(usage_path).read_text().strip())
            peak = max(peak, used)
            over = over + 1 if used > limits["watchdog_memory_gib"] * 1024 ** 3 else 0
            elapsed = time.monotonic() - started
            if log_path and Path(log_path).exists():
                with Path(log_path).open("rb") as stream:
                    stream.seek(log_offset)
                    chunk = stream.read(1024 * 1024)
                    log_offset = stream.tell()
                    data_errors += chunk.count(b"data error")
            if received:
                reason = "signal-" + str(received[0])
            elif over >= limits["memory_consecutive_polls"]:
                reason = "memory-limit"
            elif elapsed > limits["max_hours"] * 3600:
                reason = "walltime-limit"
            elif data_errors >= limits.get("max_data_errors", 100):
                reason = "data-resampling-errors"
            elif log_path and Path(log_path).exists() and time.time() - Path(log_path).stat().st_mtime > limits.get("stall_minutes", 30) * 60:
                reason = "log-stalled"
            with Path(report_path).with_name("resource_trace.jsonl").open("a") as trace:
                trace.write(json.dumps({"elapsed_seconds": round(elapsed, 2), "memory_bytes": used,
                                        "data_errors": data_errors, "reason": reason}) + "\n")
            write_json(report_path, {"status": "running", "reason": reason, "session": process.pid,
                                     "cgroup_memory_bytes": used, "peak_bytes": peak,
                                     "elapsed_seconds": round(elapsed, 2), "updated_at": utc_now()})
            if reason != "completed":
                break
            time.sleep(limits["poll_seconds"])
    except BaseException:
        reason = "watchdog-error"
        raise
    finally:
        if process is not None:
            # Even a successful parent may leave workers behind. Docker --init
            # reaps orphaned zombies; signals clean up live members of this group.
            terminate_group(process.pid, limits["termination_grace_seconds"])
            code = process.wait()
            write_json(report_path, {"status": "completed" if reason == "completed" and code == 0 else "failed",
                                     "reason": reason, "child_returncode": code, "peak_bytes": peak,
                                     "effective_limits": limits,
                                     "elapsed_seconds": round(time.monotonic() - started, 2), "updated_at": utc_now()})
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code if reason == "completed" else 125


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if sys.platform != "linux" or os.environ.get("EXTREMEOCR_CONTAINER") != "1":
        raise SystemExit("Guard requires a Linux experiment container")
    limits = json.loads((ROOT / "configs/resource_limits.json").read_text(encoding="utf-8"))
    limits.update(json.loads(os.environ.get("EXTREMEOCR_LIMITS_JSON", "{}")))
    for key in ("poll_seconds", "termination_grace_seconds", "memory_consecutive_polls", "max_hours", "stall_minutes"):
        if limits[key] <= 0:
            raise ValueError("Resource limit must be positive: " + key)
    current, maximum = memory_files()
    requested = float(os.environ.get("EXTREMEOCR_WATCHDOG_GIB", limits["watchdog_memory_gib"]))
    limits["watchdog_memory_gib"] = min(requested, maximum / 1024 ** 3 * 0.875)
    if limits["watchdog_memory_gib"] <= 0:
        raise ValueError("Watchdog memory limit must be positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Missing child command")
    run = Path(args.run_dir)
    raise SystemExit(supervise(command, current, limits, run / "watchdog.json", run / "console.log"))


if __name__ == "__main__":
    main()
