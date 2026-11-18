"""Compatibility patches are explicit; arbitrary vendor edits still fail audit."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_baseline import audit


class SourceVariantTests(unittest.TestCase):
    def test_port_requires_matching_hash_and_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            vendor = root / "vendor"
            vendor.mkdir()
            original = hashlib.sha256(b"original").hexdigest()
            patched = hashlib.sha256(b"patched").hexdigest()
            lock = {"repo": "author/model", "commit": "fixed", "archive_sha256": "archive", "path": "vendor"}
            (root / "configs/upstream.lock.json").write_text(json.dumps({"lranet_pp": lock}))
            (root / "configs/baseline_source.json").write_text(json.dumps(dict(lock, files={"model.py": original})))
            port = {"lranet_upstream_commit": "fixed", "lranet_patched_files": {"model.py": patched}}
            (root / "configs/modern_port.json").write_text(json.dumps(port))
            source = vendor / "model.py"
            source.write_bytes(b"patched")
            self.assertFalse(audit(root, vendor=vendor, variant="official")["passed"])
            report = audit(root, vendor=vendor, variant="modern")
            self.assertTrue(report["passed"])
            self.assertFalse(report["official_bytes_unchanged"])
            source.write_bytes(b"unreviewed")
            self.assertFalse(audit(root, vendor=vendor, variant="modern")["passed"])
            port["lranet_upstream_commit"] = "different"
            (root / "configs/modern_port.json").write_text(json.dumps(port))
            with self.assertRaises(ValueError):
                audit(root, vendor=vendor, variant="modern")
