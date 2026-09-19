import unittest

from product_record import CoverageStatus, OCRResult, ProductRecord


class ProductRecordTests(unittest.TestCase):
    def test_one_image_is_available_but_does_not_complete_coverage(self):
        record = ProductRecord(product_id="product-1", session_id="session-1")
        source = "captures/front.jpg"
        record.add_image("front", source)

        self.assertEqual(record.product_id, "product-1")
        self.assertEqual(record.session_id, "session-1")
        self.assertEqual(record.front_image.source, source)
        self.assertIsNone(record.back_image)
        self.assertEqual(record.coverage_status, CoverageStatus.INCOMPLETE)

    def test_multiple_images_expose_front_back_side_and_image_list(self):
        record = ProductRecord(product_id="product-2")
        record.add_image("front", "front.png")
        record.add_image("back", "back.png")
        record.add_image("side", "side.png")

        self.assertEqual([image.label for image in record.image_list], ["front", "back", "side"])
        self.assertEqual(record.back_image.source, "back.png")
        self.assertEqual(record.side_image.source, "side.png")
        self.assertEqual(record.coverage_status, CoverageStatus.COMPLETE)

    def test_image_labels_are_normalized_and_preserved_per_image(self):
        record = ProductRecord(product_id="product-labels")
        image = record.add_image(" FRONT ", "front.png")

        self.assertEqual(image.label, "front")
        self.assertEqual(record.image_list[0].label, "front")

    def test_duplicate_image_labels_are_rejected_case_insensitively(self):
        record = ProductRecord(product_id="product-3")
        record.add_image("front", "first.jpg")

        with self.assertRaisesRegex(ValueError, "Duplicate image label"):
            record.add_image(" FRONT ", "second.jpg")

    def test_same_source_with_a_different_label_is_rejected(self):
        record = ProductRecord(product_id="product-source")
        record.add_image("front", "camera://same-image")

        with self.assertRaisesRegex(ValueError, "Duplicate image source"):
            record.add_image("back", "camera://same-image")

    def test_incomplete_coverage_requires_all_required_package_views(self):
        record = ProductRecord(product_id="product-4")
        record.add_image("front", "front.jpg")
        record.add_image("back", "back.jpg")

        self.assertEqual(record.coverage_status, CoverageStatus.INCOMPLETE)

    def test_invalid_labels_and_sources_are_rejected(self):
        record = ProductRecord(product_id="product-invalid")

        with self.assertRaisesRegex(ValueError, "Image label must be one of"):
            record.add_image("top", "top.jpg")
        with self.assertRaisesRegex(ValueError, "Image source must not be empty"):
            record.add_image("front", "")
        with self.assertRaisesRegex(TypeError, "path, URL, or bytes"):
            record.add_image("front", object())

    def test_merged_ocr_keeps_per_image_evidence_and_original_coordinates(self):
        front_boxes = ((12, 34, 56, 78),)
        back_boxes = ((90, 10, 20, 30),)
        record = ProductRecord(product_id="product-5")
        front_metadata = {"provider": "ocr-a"}
        record.add_image(
            "front",
            "camera://front",
            [OCRResult("Front evidence", front_boxes, front_metadata)],
            {"capture": "front"},
        )
        record.add_image("back", "camera://back", [OCRResult("Back evidence", back_boxes)])

        self.assertEqual(record.front_image.source, "camera://front")
        self.assertEqual(record.front_image.ocr_results[0].bounding_boxes, front_boxes)
        self.assertEqual(record.back_image.ocr_results[0].bounding_boxes, back_boxes)
        self.assertIs(record.front_image.ocr_results[0].metadata, front_metadata)
        self.assertEqual(record.front_image.metadata, {"capture": "front"})
        self.assertEqual(record.merged_ocr_text, "Front evidence\nBack evidence")

    def test_session_only_and_single_image_records_are_safe(self):
        record = ProductRecord(session_id="session-only")
        record.add_image("front", "legacy-single-image.jpg")

        self.assertIsNone(record.product_id)
        self.assertEqual(record.session_id, "session-only")
        self.assertEqual(len(record.images), 1)
        self.assertEqual(record.coverage_status.value, "incomplete")

    def test_empty_record_identifiers_and_malformed_ocr_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "product_id or session_id"):
            ProductRecord()

        record = ProductRecord(product_id="product-malformed")
        with self.assertRaisesRegex(TypeError, "OCRResult"):
            record.add_image("front", "front.jpg", [{"full_text": "raw"}])


if __name__ == "__main__":
    unittest.main()
