import unittest

from tools.build_pdf_catalog import classify


class PdfImportTests(unittest.TestCase):
    def test_plural_earrings_are_classified(self):
        self.assertEqual(classify("A pair of silver earrings on a card."), "KÜPE")

    def test_pink_does_not_match_pin(self):
        self.assertEqual(classify("A pink children's book."), "HEDİYELİK")

    def test_hair_pin_precedes_generic_pin(self):
        self.assertEqual(classify("A pair of silver hair pins."), "TOKA")

    def test_belt_chain_is_accessory(self):
        self.assertEqual(classify("A black belt with a gold chain."), "AKSESUAR")


if __name__ == "__main__":
    unittest.main()
