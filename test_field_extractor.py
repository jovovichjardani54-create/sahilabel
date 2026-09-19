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

    def test_multi_word_commodity_beats_flavour_term_in_title(self):
        fields = extract_fields(words_from_lines(
            [("CHOCOLATE", 96), ("ICE", 90), ("CREAM", 90)],
        ))
        self.assertEqual(fields["generic_product_name"]["value"], "ICE CREAM")
        self.assertEqual(fields["generic_product_name"]["evidence_text"], "ICE CREAM")

    def test_chocolate_remains_a_valid_single_word_commodity(self):
        fields = extract_fields(words_from_lines(
            [("CHOCOLATE", 96)],
        ))
        self.assertEqual(fields["generic_product_name"]["value"], "CHOCOLATE")

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

    def test_real_label_nutrition_panel_does_not_supply_product_or_net_quantity(self):
        fields = extract_fields(words_from_lines(
            [("Nutrition", 95), ("Facts", 95)],
            [("Per", 95), ("100g", 95)],
            [("Serving", 95), ("size:", 95), ("54g", 95)],
            [("Typical", 95), ("Values", 95), ("Per", 95), ("100g", 95)],
            [("Total", 95), ("Sugar", 95), ("4.10", 95)],
            [("MARKETED", 95), ("BY:", 95)],
            [("Nutri", 95), ("Being", 95), ("Pvt", 95), ("Ltd.", 95)],
        ))
        self.assertEqual(fields["generic_product_name"]["status"], NOT_FOUND)
        self.assertEqual(fields["net_quantity"]["status"], NOT_FOUND)
        self.assertEqual(fields["manufacturer_or_packer"]["value"], "Nutri Being Pvt")
        self.assertNotIn("servings", fields["manufacturer_or_packer"]["value"].casefold())

    def test_merged_net_weight_token_and_final_nine_select_net_quantity(self):
        fields = extract_fields(words_from_lines(
            [("BISCUMTSNETWeIGHT", 88), ("2009", 82)],
        ))
        self.assertEqual(fields["generic_product_name"]["value"], "BISCUITS")
        self.assertEqual(fields["net_quantity"]["value"], "200g")
        self.assertIn("BISCUMTSNETWeIGHT", fields["net_quantity"]["evidence_text"])

    def test_mrp_ranking_rejects_nearby_unit_price(self):
        fields = extract_fields(words_from_lines(
            [("MRP", 95), ("50.00", 95), ("0.25", 95), ("per", 95), ("g", 95)],
        ))
        self.assertEqual(fields["mrp"]["value"], "50.00")
        self.assertNotIn("0.25", fields["mrp"]["evidence_text"])

    def test_invalid_numeric_date_is_not_treated_as_packing_date(self):
        fields = extract_fields(words_from_lines(
            [("PKD", 90), ("16/44/23", 92)],
        ))
        self.assertEqual(fields["manufacture_or_packing_date"]["status"], UNCERTAIN)
        self.assertIsNone(fields["manufacture_or_packing_date"]["value"])

    def test_low_confidence_unmarked_month_year_is_not_accepted(self):
        fields = extract_fields(words_from_lines(
            [("04-33", 33)],
        ))
        self.assertEqual(fields["manufacture_or_packing_date"]["status"], UNCERTAIN)
        self.assertIsNone(fields["manufacture_or_packing_date"]["value"])

    def test_reliable_marker_backed_month_year_is_supported(self):
        fields = extract_fields(words_from_lines(
            [("MFD", 92), ("04/2024", 94)],
        ))
        self.assertEqual(fields["manufacture_or_packing_date"]["status"], FOUND)
        self.assertEqual(fields["manufacture_or_packing_date"]["value"], "04/2024")

    def test_currency_price_without_mrp_marker_is_uncertain_and_excludes_unit_price(self):
        fields = extract_fields(words_from_lines(
            [("₹300.00", 96), ("₹0.40", 96), ("/", 96), ("ml", 96)],
        ))
        mrp = fields["mrp"]
        self.assertEqual(mrp["status"], UNCERTAIN)
        self.assertEqual(mrp["value"], "₹300.00")
        self.assertNotIn("₹0.40", mrp["evidence_text"])

    def test_marker_backed_mrp_prefers_package_price_over_nearby_unit_price(self):
        fields = extract_fields(words_from_lines(
            [("MRP", 94), ("₹300.00", 95), ("₹0.40", 96), ("/", 96), ("ml", 96)],
        ))

        self.assertEqual(fields["mrp"]["status"], FOUND)
        self.assertEqual(fields["mrp"]["value"], "₹300.00")
        self.assertNotIn("₹0.40", fields["mrp"]["evidence_text"])

    def test_consumer_care_can_use_contiguous_digit_fragments(self):
        fields = extract_fields(words_from_lines(
            [("Customer", 94), ("Care", 94), ("1800", 94), ("123", 94), ("456", 94)],
        ))
        self.assertEqual(fields["consumer_care"]["value"], "1800123456")
        self.assertIn("1800", fields["consumer_care"]["evidence_text"])

    def test_nutrition_column_cannot_borrow_net_or_mrp_heading_to_its_right(self):
        ocr = words_from_lines(
            [("Energy", 91), ("160", 94), ("kcal", 91),
             ("Net", 94), ("Weight", 94), ("250g", 91),
             ("MRP", 91), ("Rs.600.00", 92)],
            [("Fat", 90), ("1.5g", 95)],
        )
        fields = extract_fields(ocr)
        self.assertEqual(fields["net_quantity"]["value"], "250g")
        self.assertEqual(fields["net_quantity"]["status"], FOUND)
        self.assertEqual(fields["mrp"]["value"], "Rs.600.00")
        self.assertEqual(fields["mrp"]["status"], FOUND)

    def test_ingredient_word_does_not_beat_product_title(self):
        fields = extract_fields(words_from_lines(
            [("CALIFORNIA", 90), ("PISTACHIOS", 79)],
            [("Ingredients:", 94), ("Pistachios,", 95), ("Salt", 99)],
        ))
        self.assertEqual(fields["generic_product_name"]["value"], "PISTACHIOS")

    def test_unmarked_quantity_is_uncertain_not_compliant(self):
        fields = extract_fields(words_from_lines([("250g", 92)]))
        self.assertEqual(fields["net_quantity"]["status"], UNCERTAIN)

    def test_date_and_regulatory_context_are_not_product_names(self):
        fields = extract_fields(words_from_lines(
            [("PKD", 96), ("04/2025", 96)],
            [("MRP", 96), ("FSSAI", 96), ("Licence", 96)],
        ))
        self.assertEqual(fields["generic_product_name"]["status"], NOT_FOUND)

    def test_unrelated_similar_ocr_word_is_not_a_commodity(self):
        fields = extract_fields(words_from_lines(
            [("Recommended", 96), ("average", 96), ("adult", 96)],
        ))
        self.assertEqual(fields["generic_product_name"]["status"], NOT_FOUND)

    def test_barcode_digits_near_care_heading_are_not_a_contact(self):
        fields = extract_fields(words_from_lines(
            [("Customer", 95), ("Care", 95)],
            [("9014600363", 98)],
        ))
        self.assertEqual(fields["consumer_care"]["status"], UNCERTAIN)
        self.assertIsNone(fields["consumer_care"]["value"])

    def test_drained_quantity_cannot_beat_net_quantity(self):
        fields = extract_fields(words_from_lines(
            [("NET", 95), ("QUANTITY", 95), ("450g", 95)],
            [("DRAINED", 95), ("QUANTITY", 95), ("212g", 98)],
        ))
        self.assertEqual(fields["net_quantity"]["value"], "450g")

    def test_separate_drained_heading_cannot_beat_net_quantity(self):
        fields = extract_fields(words_from_lines(
            [("NET", 95), ("QUANTITY", 95), ("450g", 95)],
            [("DRAINED", 95)],
            [("212g", 98)],
        ))
        self.assertEqual(fields["net_quantity"]["value"], "450g")

    def test_malformed_three_digit_year_is_not_a_packing_date(self):
        fields = extract_fields(words_from_lines([("Mfg", 95), ("Date", 95), ("04/225", 96)]))
        self.assertEqual(fields["manufacture_or_packing_date"]["status"], UNCERTAIN)
        self.assertIsNone(fields["manufacture_or_packing_date"]["value"])

    def test_manufacturer_does_not_absorb_nutrition_column(self):
        fields = extract_fields(words_from_lines(
            [("Marketed", 96), ("By", 96)],
            [("Nutrition", 96), ("Foods", 96), ("Pvt", 96), ("Ltd", 96)],
            [("Acme", 96), ("Foods", 96), ("Pvt", 96), ("Ltd", 96)],
        ))
        manufacturer = fields["manufacturer_or_packer"]
        self.assertIn("Acme", manufacturer["value"])
        self.assertNotIn("Nutrition", manufacturer["value"])

    def test_unlabelled_nearby_street_is_not_consumer_care_address(self):
        fields = extract_fields(words_from_lines(
            [("Customer", 96), ("Care", 96)],
            [("Bharuch", 96), ("Road", 96), ("Gujarat", 96)],
        ))
        self.assertEqual(fields["consumer_care"]["status"], UNCERTAIN)
        self.assertIsNone(fields["consumer_care"]["value"])


if __name__ == "__main__":
    unittest.main()
