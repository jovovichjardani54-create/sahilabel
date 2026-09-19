import unittest

from compliance_checker import check_fields
from product_evidence import aggregate_product_evidence, evaluate_product_record
from product_record import CoverageStatus, OCRResult, ProductRecord
from test_field_extractor import words_from_lines


def ocr(*lines):
    payload = words_from_lines(*lines)
    return OCRResult(" ".join(word["text"] for word in payload["words"]), payload["words"])


def complete_ocr(confidence=90):
    return ocr(
        [("Marketed", confidence), ("by", confidence), ("Example", confidence), ("Foods", confidence)],
        [("BISCUITS", confidence), ("Net", confidence), ("Weight", confidence), ("100", confidence), ("g", confidence)],
        [("MRP", confidence), ("45.00", confidence)],
        [("PKD", confidence), ("22/07/26", confidence)],
        [("Customer", confidence), ("Care", confidence), ("1800123456", confidence)],
    )


class ProductEvidenceTests(unittest.TestCase):
    def record_with(self, *images):
        record = ProductRecord(product_id="product-evidence")
        for label, source, result in images:
            record.add_image(label, source, [result])
        return record

    def test_front_only_is_incomplete_and_preserves_single_image_decisions(self):
        result = complete_ocr()
        record = self.record_with(("front", "camera://front", result))

        evaluation = evaluate_product_record(record)

        self.assertEqual(evaluation.evidence.coverage_status, CoverageStatus.INCOMPLETE)
        self.assertEqual(evaluation.report["overall_result"], "PASS")
        self.assertEqual(check_fields({"words": list(result.bounding_boxes)}, False, True)["overall_result"], "PASS")

    def test_incomplete_coverage_keeps_missing_declarations_in_review(self):
        record = self.record_with(("front", "camera://front", ocr([("BISCUITS", 90)])))

        evaluation = evaluate_product_record(record)

        self.assertEqual(evaluation.report["field_decisions"]["mrp"]["status"], "REVIEW")
        self.assertEqual(evaluation.report["overall_result"], "REVIEW")

    def test_back_only_declaration_satisfies_front_absence_with_source_provenance(self):
        front = ocr([("BISCUITS", 90)])
        back = ocr([("Net", 97), ("Weight", 97), ("500", 97), ("g", 97)])
        record = self.record_with(("front", "camera://front", front), ("back", "camera://back", back))

        selected = aggregate_product_evidence(record).declarations["net_quantity"]

        self.assertEqual(selected.value, "500g")
        self.assertEqual(selected.source_label, "back")
        self.assertEqual(selected.source_identity, record.back_image.source_identity)
        self.assertEqual(selected.ocr_words, back.bounding_boxes)

    def test_side_only_declaration_satisfies_front_absence(self):
        record = self.record_with(
            ("front", "camera://front", ocr([("BISCUITS", 90)])),
            ("back", "camera://back", ocr([("MRP", 90), ("45.00", 90)])),
            ("side", "camera://side", ocr([("Customer", 95), ("Care", 95), ("1800123456", 95)])),
        )

        selected = aggregate_product_evidence(record).declarations["consumer_care"]

        self.assertEqual(selected.value, "1800123456")
        self.assertEqual(selected.source_label, "side")
        self.assertEqual(record.coverage_status, CoverageStatus.COMPLETE)

    def test_higher_confidence_competing_value_wins_and_retains_original_bbox(self):
        front = ocr([("Net", 80), ("Weight", 80), ("250", 80), ("g", 80)])
        back = ocr([("Net", 96), ("Weight", 96), ("500", 96), ("g", 96)])
        record = self.record_with(("front", "camera://front", front), ("back", "camera://back", back))

        selected = aggregate_product_evidence(record).declarations["net_quantity"]

        self.assertEqual(selected.value, "500g")
        self.assertEqual(selected.source_label, "back")
        self.assertEqual(selected.confidence, 96.0)
        self.assertEqual(selected.bounding_box, {"left": 10, "top": 200, "width": 119, "height": 12})
        self.assertEqual(len(aggregate_product_evidence(record).candidates["net_quantity"]), 2)

    def test_context_then_fixed_label_order_breaks_confidence_ties(self):
        front = ocr([("100", 90), ("g", 90)])
        side = ocr([("Net", 90), ("Weight", 90), ("100", 90), ("g", 90)])
        record = self.record_with(("front", "camera://front", front), ("side", "camera://side", side))

        selected = aggregate_product_evidence(record).declarations["net_quantity"]

        self.assertEqual(selected.source_label, "side")

    def test_complete_coverage_produces_deterministic_pass_and_source_evidence(self):
        record = self.record_with(
            ("front", "camera://front", complete_ocr()),
            ("back", "camera://back", complete_ocr()),
            ("side", "camera://side", complete_ocr()),
        )

        evaluation = evaluate_product_record(record)
        decision = evaluation.report["field_decisions"]["net_quantity"]

        self.assertEqual(evaluation.report["overall_result"], "PASS")
        self.assertEqual(decision["source_evidence"]["source_label"], "front")
        self.assertEqual(decision["source_evidence"]["ocr_words"], record.front_image.ocr_results[0].bounding_boxes)

    def test_coordinates_are_never_merged_between_images(self):
        front = ocr([("Net", 90), ("Weight", 90), ("100", 90), ("g", 90)])
        back_words = words_from_lines([("Net", 96), ("Weight", 96), ("500", 96), ("g", 96)])["words"]
        for word in back_words:
            word["left"] += 1000
            word["top"] += 500
        back = OCRResult("Net Weight 500 g", back_words)
        record = self.record_with(("front", "camera://front", front), ("back", "camera://back", back))

        selected = aggregate_product_evidence(record).declarations["net_quantity"]

        self.assertGreater(selected.bounding_box["left"], 1000)
        self.assertGreater(selected.bounding_box["top"], 500)
        self.assertEqual(selected.ocr_words, back.bounding_boxes)

    def test_invalid_input_and_empty_records_are_rejected(self):
        with self.assertRaises(TypeError):
            aggregate_product_evidence(object())
        with self.assertRaisesRegex(ValueError, "at least one image"):
            aggregate_product_evidence(ProductRecord(product_id="empty"))


if __name__ == "__main__":
    unittest.main()
