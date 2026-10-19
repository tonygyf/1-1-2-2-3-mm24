"""Single bounded local acceptance command. Never starts OCR training."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from extremeocr.provenance import utc_now, write_json
from audit_baseline import audit
from docker_run import docker_command


def main():
    checks = {"official_source": audit()}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "work/qa_deps"), environment.get("PYTHONPATH", "")])
    for key, command in (("unit_and_protocol_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]),
                         ("fixture_smoke", [sys.executable, "scripts/smoke.py"])):
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
        checks[key] = {"passed": result.returncode == 0, "command": command,
                       "stdout": result.stdout, "stderr": result.stderr}
        print(key + ": " + ("PASS" if result.returncode == 0 else "FAIL"), flush=True)
        if result.returncode:
            print(result.stdout + result.stderr)
    limits = json.loads((ROOT / "configs/resource_limits.json").read_text())
    docker_command(ROOT, "test:config-only", limits, ["python", "scripts/preflight.py"])
    checks["docker_command_configuration"] = {"passed": True, "runtime": "not_run"}
    # Skipped matcher tests cannot silently satisfy a claimed evaluator gate.
    tests = checks["unit_and_protocol_tests"]
    matcher_skipped = "Matcher QA requires" in tests["stderr"]
    checks["matcher_fixture"] = {"passed": not matcher_skipped and tests["passed"],
                                  "scope": "official matcher on software gold/empty/wrong-order fixtures"}
    passed = all(check.get("passed", False) for check in checks.values())
    report = {"status": "passed" if passed else "failed", "verified_at": utc_now(), "checks": checks,
              "server_gpu_environment": "not_run", "docker_build_and_watchdog_runtime": "not_run",
              "baseline_accuracy_reproduction": "not_run", "research_metrics": None}
    write_json(ROOT / "outputs/local_verification.json", report)
    print(json.dumps({"local_software_acceptance": report["status"], "research_metrics": None}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
