import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from ai.query_processor import normalize_uploaded_image, prepare_query_views


class QueryProcessorTests(unittest.TestCase):
    def test_phone_exif_orientation_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "phone.jpg"
            destination = Path(directory) / "normalized.jpg"
            raw = Image.new("RGB", (40, 80), "white")
            exif = raw.getexif()
            exif[274] = 6
            raw.save(source, exif=exif)

            normalize_uploaded_image(source, destination)

            with Image.open(destination) as normalized:
                self.assertEqual(normalized.size, (80, 40))

    def test_product_on_plain_background_creates_focused_view(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.jpg"
            image = Image.new("RGB", (800, 600), "white")
            draw = ImageDraw.Draw(image)
            draw.ellipse((280, 160, 520, 440), fill=(180, 120, 20))
            image.save(path)

            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).ellipse((280, 160, 520, 440), fill=255)
            with patch("ai.query_processor._background_mask", return_value=mask):
                views = prepare_query_views(path)

        self.assertEqual(len(views), 3)
        self.assertLess(views[1].width, views[0].width)
        self.assertLess(views[1].height, views[0].height)
        self.assertEqual(views[2].size, (1024, 1024))
        self.assertEqual(views[2].getpixel((0, 0)), (255, 255, 255))

    def test_necklace_category_adds_overlapping_vertical_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two-necklaces.jpg"
            Image.new("RGB", (600, 1000), "white").save(path)
            with patch("ai.query_processor._background_mask", return_value=None):
                views = prepare_query_views(path, "KOLYE")

        self.assertEqual(len(views), 3)
        self.assertEqual(views[-2].size, (600, 590))
        self.assertEqual(views[-1].size, (600, 590))

    def test_bracelet_category_adds_sides_and_rotated_view(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vertical-bracelet.jpg"
            Image.new("RGB", (600, 1000), "white").save(path)
            with patch("ai.query_processor._background_mask", return_value=None):
                views = prepare_query_views(path, "BİLEKLİK")

        self.assertEqual(len(views), 5)
        self.assertEqual(views[-4].size, (276, 1000))
        self.assertEqual(views[-3].size, (276, 1000))
        self.assertEqual(views[-2].size, (1000, 276))
        self.assertEqual(views[-1].size, (1000, 276))


if __name__ == "__main__":
    unittest.main()
