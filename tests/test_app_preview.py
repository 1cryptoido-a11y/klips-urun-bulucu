import base64
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app import make_query_preview


class QueryPreviewTests(unittest.TestCase):
    def test_preview_is_compact_browser_safe_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.png"
            Image.new("RGB", (1800, 1200), (180, 120, 20)).save(path)
            preview = make_query_preview(path)

        prefix = "data:image/jpeg;base64,"
        self.assertTrue(preview.startswith(prefix))
        decoded = base64.b64decode(preview.removeprefix(prefix))
        self.assertLess(len(decoded), 250_000)
