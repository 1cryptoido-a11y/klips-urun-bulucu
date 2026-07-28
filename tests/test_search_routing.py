import unittest

import numpy as np
from PIL import Image, ImageDraw

from ai.search_engine import (
    count_blue_motif_centers,
    select_necklace_dino_queries,
    structural_weights,
)


class NecklaceDinoRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queries = np.arange(8 * 3, dtype="float32").reshape(8, 3)

    def test_macro_photo_keeps_every_query_view(self) -> None:
        selected = select_necklace_dino_queries(
            self.queries, macro_necklace=True, layered_necklace=False
        )
        np.testing.assert_array_equal(selected, self.queries)

    def test_layered_portrait_uses_whole_product_crops(self) -> None:
        selected = select_necklace_dino_queries(
            self.queries, macro_necklace=False, layered_necklace=True
        )
        np.testing.assert_array_equal(selected, self.queries[4:6])

    def test_single_necklace_portrait_skips_dino(self) -> None:
        selected = select_necklace_dino_queries(
            self.queries, macro_necklace=False, layered_necklace=False
        )
        self.assertIsNone(selected)

    def test_blue_motif_counter_ignores_shade_but_keeps_count(self) -> None:
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        for index in range(10):
            center_x = 40 + (index * 47)
            color = (20, 100 + index, 180 + index * 3)
            draw.ellipse((center_x - 9, 245, center_x + 9, 263), fill=color)
        self.assertEqual(count_blue_motif_centers(image), 10)

    def test_bracelet_profile_prefers_grayscale_structure(self) -> None:
        object_weight, color_weight = structural_weights(
            "BİLEKLİK", macro_necklace=False
        )
        self.assertGreater(object_weight, color_weight)

    def test_necklace_profile_preserves_proven_instance_weight(self) -> None:
        self.assertEqual(
            structural_weights("KOLYE", macro_necklace=True), (1.04, 1.18)
        )


if __name__ == "__main__":
    unittest.main()
