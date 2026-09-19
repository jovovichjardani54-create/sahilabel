import unittest
from datetime import datetime, timezone

from inspector_review import (
    InspectorReviewError,
    REGISTRY,
    confirm_review,
    create_inspector_review,
    create_pending_review,
    get_review_status,
    override_review,
    requires_human_inspection,
)


class InspectorReviewTests(unittest.TestCase):
    def setUp(self):
        REGISTRY.clear()

    def tearDown(self):
        REGISTRY.clear()

    def test_confirmation_preserves_automated_decision(self):
        automated = {"overall": "REVIEW", "status": "REVIEW"}
        review = create_inspector_review(
            reviewer="insp-42",
            automated_decision=automated,
            inspector_decision="confirmed",
            reason="Label matches Legal Metrology checklist.",
            field_confirmations={"mrp": True, "net_quantity": True},
            inspector_id="insp-42",
            reviewer_name="Jay",
        )
        automated["overall"] = "FAIL"
        payload = review.to_dict()
        self.assertEqual("CONFIRMED", review.inspector_decision)
        self.assertEqual("REVIEW", review.automated_decision["overall"])
        self.assertEqual("insp-42", payload["inspector_id"])
        self.assertEqual("Jay", payload["reviewer_name"])
        self.assertEqual("REVIEW", payload["automated_decision"]["overall"])
        self.assertEqual("CONFIRMED", payload["inspector_decision"])
        self.assertEqual("Label matches Legal Metrology checklist.", payload["reason"])
        self.assertEqual({"mrp": True, "net_quantity": True}, payload["field_confirmations"])
        self.assertIn("reviewed_at", payload)
        self.assertTrue(payload["audit_trail"])
        self.assertNotEqual(payload["automated_decision"], payload["inspector_decision"])

    def test_override_keeps_original_automated_result(self):
        reviewed_at = datetime(2026, 9, 19, 1, 0, tzinfo=timezone.utc)
        review = create_inspector_review(
            reviewer="Jay",
            automated_decision="REVIEW",
            inspector_decision="OVERRIDDEN",
            reason="Physical package shows a valid MRP declaration.",
            reviewed_at=reviewed_at,
            inspector_id="insp-7",
            reviewer_name="Jay",
        )
        payload = review.to_dict()
        self.assertEqual("insp-7", payload["inspector_id"])
        self.assertEqual("Jay", payload["reviewer_name"])
        self.assertEqual("REVIEW", payload["automated_decision"])
        self.assertEqual("OVERRIDDEN", payload["inspector_decision"])
        self.assertEqual("Physical package shows a valid MRP declaration.", payload["reason"])
        self.assertEqual("2026-09-19T01:00:00+00:00", payload["reviewed_at"])
        self.assertEqual("2026-09-19T01:00:00+00:00", payload["created_at"])
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

    def test_override_without_reason_is_rejected(self):
        with self.assertRaisesRegex(InspectorReviewError, "override reason is required"):
            create_inspector_review(
                reviewer="insp-1",
                automated_decision="REVIEW",
                inspector_decision="OVERRIDDEN",
                reason="   ",
                inspector_id="insp-1",
                reviewer_name="Jay",
            )

    def test_invalid_status_is_rejected(self):
        with self.assertRaisesRegex(InspectorReviewError, "CONFIRMED, OVERRIDDEN, or PENDING"):
            create_inspector_review(
                reviewer="insp-1",
                automated_decision="REVIEW",
                inspector_decision="APPROVED",
                reason="Looks good",
                inspector_id="insp-1",
                reviewer_name="Jay",
            )

    def test_creating_a_pending_review(self):
        result = create_pending_review(
            "item-1",
            {"overall_status": "REVIEW", "mrp": "unclear"},
            inspector_id="insp-9",
            reviewer_name="Priya",
        )
        self.assertTrue(result["created"])
        self.assertTrue(result["review_required"])
        self.assertEqual("PENDING", result["inspector_decision"])
        self.assertEqual("insp-9", result["inspector_id"])
        self.assertEqual("Priya", result["reviewer_name"])
        self.assertEqual("REVIEW", result["system_status"])
        self.assertEqual({"overall_status": "REVIEW", "mrp": "unclear"}, result["automated_decision"])
        self.assertTrue(result["reviewed_at"])
        self.assertEqual("created", result["audit_trail"][0]["action"])

    def test_retrieving_a_pending_review(self):
        create_pending_review("item-2", {"status": "REVIEW"}, inspector_id="insp-2", reviewer_name="Alex")
        status = get_review_status("item-2")
        self.assertTrue(status["found"])
        self.assertEqual("PENDING", status["inspector_decision"])
        self.assertEqual("item-2", status["review_id"])
        self.assertEqual("insp-2", status["inspector_id"])
        self.assertIsNone(status["message"])

    def test_inspector_confirmation(self):
        create_pending_review("item-3", {"status": "REVIEW"}, inspector_id="insp-3", reviewer_name="Sam")
        confirmed = confirm_review(
            "item-3",
            inspector_id="insp-3",
            reviewer_name="Sam",
            reason="System REVIEW is correct",
        )
        self.assertEqual("CONFIRMED", confirmed["inspector_decision"])
        self.assertEqual({"status": "REVIEW"}, confirmed["automated_decision"])
        self.assertEqual("confirmed", confirmed["audit_trail"][-1]["action"])
        self.assertEqual("PENDING", confirmed["audit_trail"][0]["inspector_decision"])

    def test_inspector_override(self):
        original = {"status": "REVIEW", "rule": "C12"}
        create_pending_review("item-4", original, inspector_id="insp-4", reviewer_name="Jay")
        overridden = override_review(
            "item-4",
            inspector_id="insp-4",
            reviewer_name="Jay",
            reason="Physical inspection found a valid declaration",
        )
        self.assertEqual("OVERRIDDEN", overridden["inspector_decision"])
        self.assertEqual(original, overridden["automated_decision"])
        self.assertEqual("Physical inspection found a valid declaration", overridden["reason"])
        self.assertEqual(2, len(overridden["audit_trail"]))

    def test_override_reason_presence_validation(self):
        create_pending_review("item-5", "REVIEW", inspector_id="insp-5", reviewer_name="Jay")
        with self.assertRaisesRegex(InspectorReviewError, "override reason is required"):
            override_review("item-5", inspector_id="insp-5", reviewer_name="Jay", reason="")
        pending = get_review_status("item-5")
        self.assertEqual("PENDING", pending["inspector_decision"])
        self.assertEqual("REVIEW", pending["automated_decision"])

    def test_timestamp_presence_and_correctness(self):
        before = datetime.now(timezone.utc)
        created = create_pending_review("item-6", {"status": "REVIEW"}, inspector_id="insp-6", reviewer_name="Jay")
        confirmed = confirm_review("item-6", inspector_id="insp-6", reviewer_name="Jay")
        after = datetime.now(timezone.utc)
        created_at = datetime.fromisoformat(created["created_at"])
        reviewed_at = datetime.fromisoformat(confirmed["reviewed_at"])
        trail_at = datetime.fromisoformat(confirmed["audit_trail"][-1]["at"])
        self.assertLessEqual(before, created_at)
        self.assertLessEqual(created_at, reviewed_at)
        self.assertLessEqual(reviewed_at, after)
        self.assertEqual(reviewed_at, trail_at)
        self.assertEqual(created["created_at"], confirmed["created_at"])

    def test_invalid_inspector_input(self):
        create_pending_review("item-7", "REVIEW", inspector_id="insp-7", reviewer_name="Jay")
        with self.assertRaisesRegex(InspectorReviewError, "inspector ID is required"):
            confirm_review("item-7", inspector_id="  ", reviewer_name="Jay")
        with self.assertRaisesRegex(InspectorReviewError, "reviewer name is required"):
            override_review(
                "item-7",
                inspector_id="insp-7",
                reviewer_name="",
                reason="Package is compliant after physical check",
            )
        with self.assertRaisesRegex(InspectorReviewError, "inspector ID or reviewer name is required"):
            create_inspector_review(
                automated_decision="REVIEW",
                inspector_decision="PENDING",
            )

    def test_missing_review_safe_behavior(self):
        status = get_review_status("does-not-exist")
        self.assertFalse(status["found"])
        self.assertIsNone(status["review"])
        self.assertIsNone(status["inspector_decision"])
        self.assertIsNone(status["automated_decision"])
        self.assertEqual([], status["audit_trail"])
        self.assertEqual("No inspector review exists for this item", status["message"])
        with self.assertRaisesRegex(InspectorReviewError, "No inspector review exists"):
            confirm_review("does-not-exist", inspector_id="insp-1", reviewer_name="Jay")

    def test_preserving_original_system_result_after_override(self):
        original = {"overall_status": "REVIEW", "score": 40}
        create_pending_review("item-8", original, inspector_id="insp-8", reviewer_name="Jay")
        original["overall_status"] = "PASS"
        original["score"] = 99
        overridden = override_review(
            "item-8",
            inspector_id="insp-8",
            reviewer_name="Jay",
            reason="Inspector found the missing MRP on the inner flap",
        )
        self.assertEqual("REVIEW", overridden["automated_decision"]["overall_status"])
        self.assertEqual(40, overridden["automated_decision"]["score"])
        self.assertEqual("OVERRIDDEN", overridden["inspector_decision"])
        self.assertEqual("REVIEW", overridden["audit_trail"][-1]["automated_decision"]["overall_status"])

    def test_pass_cases_do_not_require_unnecessary_review(self):
        self.assertFalse(requires_human_inspection({"status": "PASS"}))
        self.assertFalse(requires_human_inspection("VIOLATION"))
        self.assertFalse(requires_human_inspection("FAIL"))
        self.assertTrue(requires_human_inspection("REVIEW"))
        result = create_pending_review("item-pass", {"status": "PASS"}, inspector_id="insp-9", reviewer_name="Jay")
        self.assertFalse(result["created"])
        self.assertFalse(result["review_required"])
        self.assertIsNone(result["inspector_decision"])
        self.assertEqual({"status": "PASS"}, result["automated_decision"])
        self.assertIn("do not require inspector review", result["message"])
        self.assertFalse(get_review_status("item-pass")["found"])
        violation = create_pending_review("item-fail", "VIOLATION", inspector_id="insp-9", reviewer_name="Jay")
        self.assertFalse(violation["review_required"])
        self.assertFalse(get_review_status("item-fail")["found"])


if __name__ == "__main__":
    unittest.main()
