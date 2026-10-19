"""Real child-process fault scenarios; Linux only, no OCR/GPU or memory stress."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from resource_guard import supervise


@unittest.skipUnless(sys.platform == "linux", "Process-group runtime requires Linux server/container")
class ResourceGuardTests(unittest.TestCase):
    def run_case(self, script, memory=0, hours=1):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            usage, report = folder / "memory", folder / "watchdog.json"
            usage.write_text(str(memory))
            limits = {"watchdog_memory_gib": 1, "memory_consecutive_polls": 2,
                      "poll_seconds": 0.02, "termination_grace_seconds": 0.2,
                      "max_hours": hours}
            code = supervise([sys.executable, "-c", script], usage, limits, report)
            return code, json.loads(report.read_text())

    def test_success_and_nonzero_exit(self):
        self.assertEqual(self.run_case("pass")[0], 0)
        code, report = self.run_case("raise SystemExit(7)")
        self.assertEqual(code, 7)
        self.assertEqual(report["status"], "failed")

    def test_memory_threshold_without_allocating_memory(self):
        code, report = self.run_case("import time; time.sleep(10)", memory=2 * 1024 ** 3)
        self.assertEqual(code, 125)
        self.assertEqual(report["reason"], "memory-limit")

    def test_walltime(self):
        code, report = self.run_case("import time; time.sleep(10)", hours=0.00002)
        self.assertEqual(code, 125)
        self.assertEqual(report["reason"], "walltime-limit")
