import os
import unittest
from PIL import Image
from report_pdf import generate_pdf_report


class PdfReportTests(unittest.TestCase):
    def test_structured_review_report_uses_decisions_and_missing_fallback(self):
        report = {"overall_result": "REVIEW", "overall_compliant": False, "decision_support_message": "Inspector review is required.", "field_decisions": {"generic_product_name": {"field_key": "generic_product_name", "rule_id": "Rule 6(1)(b)", "status": "PASS", "extracted_value": "BISCUITS", "confidence": 96.0, "reason": "Matched a configurable generic commodity term."}, "consumer_care": {"field_key": "consumer_care", "rule_id": "Rule 6(2)", "status": "REVIEW", "extracted_value": None, "confidence": 0.0, "reason": "Declaration was not found, but package-side coverage is incomplete."}}}
        output = os.path.join("reports", "_pdf_test_report.pdf")
        try:
            generate_pdf_report(report, "label.jpg", output)
            with open(output, "rb") as generated:
                header = generated.read(5)
            self.assertGreater(os.path.getsize(output), 1000)
        finally:
            if os.path.exists(output):
                os.remove(output)
        self.assertEqual(header, b"%PDF-")

    def test_pdf_generates_when_visual_evidence_is_unavailable(self):
        output = os.path.join("reports", "_pdf_missing_evidence.pdf")
        try:
            generate_pdf_report({"overall_result": "REVIEW", "field_decisions": {}}, "label.jpg", output, "missing.png")
            self.assertGreater(os.path.getsize(output), 1000)
        finally:
            if os.path.exists(output):
                os.remove(output)

    def test_pdf_accepts_existing_annotated_evidence_image(self):
        evidence = os.path.join("reports", "_pdf_test_evidence.png")
        output = os.path.join("reports", "_pdf_with_evidence.pdf")
        try:
            Image.new("RGB", (100, 300), "white").save(evidence)
            generate_pdf_report({"overall_result": "PASS", "field_decisions": {}}, "label.jpg", output, evidence)
            self.assertGreater(os.path.getsize(output), 1500)
        finally:
            for path in (evidence, output):
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()
