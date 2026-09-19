import unittest

from compliance_checker import check_fields
from test_field_extractor import words_from_lines


def complete_ocr():
    return words_from_lines(
        [("Marketed", 90), ("by", 90), ("Example", 90), ("Foods", 90)],
        [("BISCUITS", 95), ("NET", 95), ("WEIGHT", 95), ("100", 95), ("g", 95)], [("MRP", 90)],
        [("45.00", 90)], [("PKD", 90), ("22/07/26", 90)],
        [("Customer", 90), ("Care", 90), ("1800123456", 90)],
    )


class ComplianceDecisionTests(unittest.TestCase):
    def test_all_found_passes(self):
        report = check_fields(complete_ocr(), True, True)
        self.assertEqual(report["overall_result"], "PASS")
        self.assertTrue(report["overall_compliant"])

    def test_uncertain_field_requires_review(self):
        ocr = complete_ocr()
        ocr["words"] = [word for word in ocr["words"] if word["text"] != "by"]
        report = check_fields(ocr, True, True)
        self.assertEqual(report["field_decisions"]["manufacturer_or_packer"]["status"], "REVIEW")
        self.assertEqual(report["overall_result"], "REVIEW")

    def test_unmarked_date_requires_review(self):
        ocr = complete_ocr()
        ocr["words"] = [word for word in ocr["words"] if word["text"] != "PKD"]
        report = check_fields(ocr, True, True)
        self.assertEqual(report["field_decisions"]["manufacture_or_packing_date"]["status"], "REVIEW")

    def test_missing_with_incomplete_coverage_requires_review(self):
        report = check_fields(words_from_lines([("BISCUITS", 90)]), False, True)
        self.assertEqual(report["field_decisions"]["mrp"]["status"], "REVIEW")

    def test_missing_with_complete_coverage_and_quality_is_violation(self):
        report = check_fields(words_from_lines([("BISCUITS", 90)]), True, True)
        self.assertEqual(report["field_decisions"]["mrp"]["status"], "VIOLATION")
        self.assertEqual(report["overall_result"], "VIOLATION")

    def test_ocr_failure_requires_review(self):
        report = check_fields({"full_text": "", "words": []}, True)
        self.assertEqual(report["overall_result"], "REVIEW")
        self.assertEqual(report["field_decisions"]["mrp"]["status"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
