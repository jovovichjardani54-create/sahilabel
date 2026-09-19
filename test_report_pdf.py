import os
import tempfile
import unittest
from PIL import Image
from report_pdf import generate_pdf_report


class PdfReportTests(unittest.TestCase):
    def test_structured_review_report_uses_decisions_and_missing_fallback(self):
        report = {"overall_result": "REVIEW", "overall_compliant": False, "decision_support_message": "Inspector review is required.", "field_decisions": {"generic_product_name": {"field_key": "generic_product_name", "rule_id": "Rule 6(1)(b)", "status": "PASS", "extracted_value": "BISCUITS", "confidence": 96.0, "reason": "Matched a configurable generic commodity term."}, "consumer_care": {"field_key": "consumer_care", "rule_id": "Rule 6(2)", "status": "REVIEW", "extracted_value": None, "confidence": 0.0, "reason": "Declaration was not found, but package-side coverage is incomplete."}}}
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "report.pdf")
            generate_pdf_report(report, "label.jpg", output)
            with open(output, "rb") as generated:
                header = generated.read(5)
            self.assertGreater(os.path.getsize(output), 1000)
        self.assertEqual(header, b"%PDF-")

    def test_pdf_generates_when_visual_evidence_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "missing-evidence.pdf")
            generate_pdf_report({"overall_result": "REVIEW", "field_decisions": {}}, "label.jpg", output, "missing.png")
            self.assertGreater(os.path.getsize(output), 1000)

    def test_pdf_accepts_existing_annotated_evidence_image(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = os.path.join(directory, "evidence.png")
            output = os.path.join(directory, "with-evidence.pdf")
            Image.new("RGB", (100, 300), "white").save(evidence)
            generate_pdf_report({"overall_result": "PASS", "field_decisions": {}}, "label.jpg", output, evidence)
            self.assertGreater(os.path.getsize(output), 1500)


if __name__ == "__main__":
    unittest.main()
