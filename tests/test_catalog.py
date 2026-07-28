import unittest

from tools.import_catalog import infer_category, normalize_category, product_from_row


class CatalogNormalizationTests(unittest.TestCase):
    def test_category_aliases_are_merged(self):
        self.assertEqual(normalize_category("BILEKLIK"), "BİLEKLİK")
        self.assertEqual(normalize_category("Sahmaran"), "ŞAHMARAN")
        self.assertEqual(normalize_category("Toka"), "TOKA")

    def test_report_row_becomes_product(self):
        from datetime import date

        product = product_from_row(
            {
                "barkod": "001559",
                "stok_ismi": "ÜL-KP 5",
                "grup": "KÜPE",
                "firma": "GÜÇLÜ BİJUTERİ",
                "image": "https://example.test/001559.jpg",
            },
            date(2026, 7, 28),
        )
        self.assertIsNotNone(product)
        self.assertEqual(product["kod"], "001559")
        self.assertEqual(product["kategori"], "KÜPE")

    def test_missing_category_is_inferred(self):
        self.assertEqual(infer_category("000007", "İPM-KLY 7", ""), "KOLYE")
        self.assertEqual(infer_category("004453", "2", ""), "YÜZÜK")
        self.assertEqual(infer_category("127921", "2", ""), "DİĞER")


if __name__ == "__main__":
    unittest.main()
