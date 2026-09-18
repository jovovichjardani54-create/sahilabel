import unittest

from field_extractor import FOUND, NOT_FOUND, UNCERTAIN, extract_fields


def words_from_lines(*lines):
    words = []
    for top, line in enumerate(lines, start=10):
        left = 10
        for text, conf in line:
            words.append({"text": text, "conf": conf, "left": left, "top": top * 20,
                          "width": max(8, len(text) * 8), "height": 12})
            left += max(8, len(text) * 8) + 5
    return {"full_text": "", "words": words}


class FieldExtractorTests(unittest.TestCase):
    def test_generic_declarations_and_coordinates(self):
        ocr = words_from_lines(
            [("Marketed", 90), ("by", 90), ("Example", 92), ("Foods", 90), ("Ltd", 88)],
            [("BISCUITS", 96), ("NET", 91), ("WEIGHT", 90), ("100", 94), ("g", 94)],
            [("FOR", 85), ("MRP", 90)],
            [("45.00", 92)],
            [("PKD", 85), ("22/07/26", 93)],
            [("Customer", 91), ("Care", 91), ("1800123456", 93)],
        )
        fields = extract_fields(ocr)
        self.assertEqual(fields["manufacturer_or_packer"]["status"], FOUND)
        self.assertEqual(fields["generic_product_name"]["value"], "BISCUITS")
        self.assertEqual(fields["net_quantity"]["value"], "100g")
        self.assertEqual(fields["mrp"]["value"], "45.00")
        self.assertEqual(fields["manufacture_or_packing_date"]["value"], "22/07/26")
        self.assertEqual(fields["consumer_care"]["value"], "1800123456")
        self.assertEqual(fields["mrp"]["bbox"]["top"], 240)

    def test_partial_marker_and_absent_consumer_care(self):
        ocr = words_from_lines(
            [("Marketed", 76), ("Example", 82), ("Products", 80)],
            [("0.45/g", 88), ("123456789012", 91)],
        )
        fields = extract_fields(ocr)
        self.assertEqual(fields["manufacturer_or_packer"]["status"], UNCERTAIN)
        self.assertEqual(fields["mrp"]["status"], NOT_FOUND)
        self.assertEqual(fields["consumer_care"]["status"], NOT_FOUND)

    def test_net_quantity_ranking_rejects_serving_size(self):
        fields = extract_fields(words_from_lines(
            [("Nutrition", 90), ("Facts", 90), ("Serving", 94), ("size", 94), ("28g", 95)],
            [("Net", 93), ("Weight", 93), ("250g", 91)],
        ))
        self.assertEqual(fields["net_quantity"]["value"], "250g")
        self.assertIn("Net Weight", fields["net_quantity"]["evidence_text"])

    def test_contextual_final_nine_normalization(self):
        fields = extract_fields(words_from_lines(
            [("Net", 92), ("Weight", 92), ("2509", 78)],
        ))
        self.assertEqual(fields["net_quantity"]["value"], "250g")
        self.assertEqual(fields["net_quantity"]["status"], FOUND)
        self.assertIn("2509", fields["net_quantity"]["evidence_text"])
        self.assertIn("Normalized", fields["net_quantity"]["reason"])

    def test_arbitrary_number_ending_nine_is_not_normalized(self):
        fields = extract_fields(words_from_lines([("Batch", 91), ("2509", 92)]))
        self.assertEqual(fields["net_quantity"]["status"], NOT_FOUND)

    def test_nutrition_sugar_does_not_beat_product_title(self):
        fields = extract_fields(words_from_lines(
            [("PISTACHIOS", 72)],
            [("Nutrition", 95), ("Facts", 95), ("Total", 94), ("SUGAR", 96)],
        ))
        self.assertEqual(fields["generic_product_name"]["value"], "PISTACHIOS")

    def test_company_phrase_is_preferred_over_marker_verb(self):
        fields = extract_fields(words_from_lines(
            [("Packed", 90), ("and", 88), ("Marketed", 90), ("By", 91)],
            [("Example", 89), ("Foods", 93), ("Pvt", 91), ("Ltd", 92)],
        ))
        manufacturer = fields["manufacturer_or_packer"]
        self.assertEqual(manufacturer["status"], FOUND)
        self.assertIn("Foods", manufacturer["value"])
        self.assertNotEqual(manufacturer["value"].casefold(), "packed")

    def test_partial_markers_become_uncertain_without_invented_values(self):
        fields = extract_fields(words_from_lines(
            [("MRP", 86)],
            [("Date", 84), ("of", 83), ("Packing", 85)],
            [("Quality", 88), ("Care", 89), ("info@example", 57), ("74330", 65)],
        ))
        for key in ("mrp", "manufacture_or_packing_date", "consumer_care"):
            self.assertEqual(fields[key]["status"], UNCERTAIN)
            self.assertIsNone(fields[key]["value"])
            self.assertTrue(fields[key]["evidence_text"])


if __name__ == "__main__":
    unittest.main()
