"""Download/extraction failures must not silently become accepted research assets."""
import io
from pathlib import Path
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_assets import download_http, extract_images
from prepare_modern_port import unpack


class DownloadTests(unittest.TestCase):
    def test_ignored_range_restarts_without_appending(self):
        class Response:
            status_code = 200
            headers = {"Content-Type": "application/octet-stream", "Content-Length": "4"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, size): yield b"good"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.pth"
            target.with_suffix(".part").write_bytes(b"old")
            with patch("requests.get", return_value=Response()) as request:
                download_http("https://example.invalid/model", target, attempts=1)
            self.assertEqual(request.call_args.kwargs["headers"], {"Range": "bytes=3-"})
            self.assertEqual(target.read_bytes(), b"good")

    def test_resumed_range_must_match_offset(self):
        class Response:
            status_code = 206
            headers = {"Content-Range": "bytes 1-3/4"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.pth"
            partial = target.with_suffix(".part")
            partial.write_bytes(b"old")
            with patch("requests.get", return_value=Response()), self.assertRaises(ValueError):
                download_http("https://example.invalid/model", target, attempts=1)
            self.assertEqual(partial.read_bytes(), b"old")
            self.assertFalse(target.exists())

    def test_archive_cannot_escape_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.txt", b"bad")
            with self.assertRaises(ValueError):
                extract_images(archive, root / "dataset")
            self.assertFalse((root / "escape.txt").exists())

    def test_port_preserves_upstream_dangling_documentation_symlink(self):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            directory = tarfile.TarInfo("source")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            link = tarfile.TarInfo("source/logo.png")
            link.type = tarfile.SYMTYPE
            link.linkname = "missing.png"
            archive.addfile(link)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source"
            unpack(payload.getvalue(), destination, "tar")
            self.assertTrue((destination / "logo.png").is_symlink())

    def test_port_zip_cannot_escape_temporary_root(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("../escape.txt", b"bad")
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            unpack(payload.getvalue(), Path(directory) / "source", "zip")

    def test_existing_extraction_must_equal_pinned_archive_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("totaltext/test/image.jpg", b"original")
            destination = root / "dataset"
            extract_images(archive, destination)
            self.assertEqual(extract_images(archive, destination), 1)
            target = destination / "test/image.jpg"
            target.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                extract_images(archive, destination)
            self.assertEqual(target.read_bytes(), b"modified")
