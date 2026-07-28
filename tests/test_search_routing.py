import unittest

import numpy as np

from ai.search_engine import select_necklace_dino_queries


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


if __name__ == "__main__":
    unittest.main()
