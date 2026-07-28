import base64
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app import create_app, make_query_preview


class QueryPreviewTests(unittest.TestCase):
    def test_home_offers_camera_and_gallery_inputs(self):
        app = create_app()
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="kamera"', response.data)
        self.assertIn(b'capture="environment"', response.data)
        self.assertIn(b'name="foto"', response.data)

    def test_preview_is_compact_browser_safe_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.png"
            Image.new("RGB", (1800, 1200), (180, 120, 20)).save(path)
            preview = make_query_preview(path)

        prefix = "data:image/jpeg;base64,"
        self.assertTrue(preview.startswith(prefix))
        decoded = base64.b64decode(preview.removeprefix(prefix))
        self.assertLess(len(decoded), 250_000)
