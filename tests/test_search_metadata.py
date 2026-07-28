import unittest

from ai.search_engine import lexical_score, search_terms


class SearchMetadataTests(unittest.TestCase):
    def test_turkish_terms_are_preserved(self):
        self.assertEqual(search_terms("Gümüş dört yapraklı yonca kolye"), {
            "gümüş", "dört", "yapraklı", "yonca", "kolye"
        })

    def test_structured_motif_increases_lexical_score(self):
        product = {
            "kategori": "KOLYE",
            "aciklama": "Altın tonlu, balık figürlü kolye.",
            "arama_etiketleri": ["altın tonlu", "balık"],
        }
        self.assertEqual(lexical_score("altın balık kolye", product), 1.0)
        self.assertLess(lexical_score("gümüş kalp yüzük", product), 0.5)


if __name__ == "__main__":
    unittest.main()
