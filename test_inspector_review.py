import unittest
from datetime import datetime, timezone

from inspector_review import InspectorReviewError, create_inspector_review


class InspectorReviewTests(unittest.TestCase):
    def test_confirmation_preserves_automated_decision(self):
        automated = {"overall": "PASS", "status": "PASS"}
        review = create_inspector_review(
            reviewer="insp-42",
            automated_decision=automated,
            inspector_decision="confirmed",
            reason="Label matches Legal Metrology checklist.",
            field_confirmations={"mrp": True, "net_quantity": True},
        )
        automated["overall"] = "FAIL"
        payload = review.to_dict()
        self.assertEqual("CONFIRMED", review.inspector_decision)
        self.assertEqual("PASS", review.automated_decision["overall"])
        self.assertEqual("insp-42", payload["reviewer"])
        self.assertEqual("PASS", payload["automated_decision"]["overall"])
        self.assertEqual("CONFIRMED", payload["inspector_decision"])
        self.assertEqual("Label matches Legal Metrology checklist.", payload["reason"])
        self.assertEqual({"mrp": True, "net_quantity": True}, payload["field_confirmations"])
        self.assertIn("reviewed_at", payload)
        self.assertNotEqual(payload["automated_decision"], payload["inspector_decision"])

    def test_override_keeps_original_automated_result(self):
        reviewed_at = datetime(2026, 9, 19, 1, 0, tzinfo=timezone.utc)
        review = create_inspector_review(
            reviewer="Jay",
            automated_decision="FAIL",
            inspector_decision="OVERRIDDEN",
            reason="Physical package shows a valid MRP declaration.",
            reviewed_at=reviewed_at,
        )
        payload = review.to_dict()
        self.assertEqual("Jay", payload["reviewer"])
        self.assertEqual("FAIL", payload["automated_decision"])
        self.assertEqual("OVERRIDDEN", payload["inspector_decision"])
        self.assertEqual("Physical package shows a valid MRP declaration.", payload["reason"])
        self.assertEqual("2026-09-19T01:00:00+00:00", payload["reviewed_at"])
        self.assertIsNone(payload["field_confirmations"])

    def test_pending_review_allows_missing_reason(self):
        review = create_inspector_review(
            reviewer="inspector-7",
            automated_decision={"status": "REVIEW"},
            inspector_decision="PENDING",
        )
        payload = review.to_dict()
        self.assertEqual("PENDING", payload["inspector_decision"])
        self.assertEqual({"status": "REVIEW"}, payload["automated_decision"])
        self.assertIsNone(payload["reason"])
        self.assertIsNone(payload["field_confirmations"])

    def test_missing_reason_is_rejected_for_final_decisions(self):
        for decision in ("CONFIRMED", "OVERRIDDEN"):
            with self.subTest(decision=decision):
                with self.assertRaisesRegex(InspectorReviewError, "reason is required"):
                    create_inspector_review(
                        reviewer="insp-1",
                        automated_decision="PASS",
                        inspector_decision=decision,
                        reason="   ",
                    )

    def test_invalid_status_is_rejected(self):
        with self.assertRaisesRegex(InspectorReviewError, "CONFIRMED, OVERRIDDEN, or PENDING"):
            create_inspector_review(
                reviewer="insp-1",
                automated_decision="PASS",
                inspector_decision="APPROVED",
                reason="Looks good",
            )


if __name__ == "__main__":
    unittest.main()
