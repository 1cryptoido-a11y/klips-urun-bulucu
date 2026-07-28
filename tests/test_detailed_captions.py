import unittest

from tools.apply_detailed_captions import (
    ACCENT_COLORS,
    MOTIFS,
    attribute_caption,
    build_description,
    first_match,
)


class DetailedCaptionTests(unittest.TestCase):
    def test_background_color_is_not_product_color(self):
        caption = (
            "A necklace is displayed on a black tag. "
            "The necklace has a blue starfish and small beads."
        )
        relevant = attribute_caption(caption)
        self.assertNotIn("black tag", relevant)
        self.assertEqual(first_match(relevant, ACCENT_COLORS), "mavi")
        self.assertEqual(first_match(relevant, MOTIFS), "denizyıldızı")

    def test_description_uses_correct_turkish_lowercase(self):
        product = {"kategori": "BİLEKLİK"}
        value = build_description(
            product, "altın tonlu", "mavi", "balık", ["boncuk detaylı"], ""
        )
        self.assertEqual(
            value,
            "Altın tonlu, mavi renkli, balık figürlü, boncuk detaylı bileklik.",
        )


if __name__ == "__main__":
    unittest.main()
