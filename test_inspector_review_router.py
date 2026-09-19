import unittest

from inspector_review import REGISTRY
from inspector_review_router import (
    handle_confirm_review,
    handle_create_pending_review,
    handle_get_review,
    handle_list_pending_reviews,
    handle_override_review,
)


class InspectorReviewRouterTests(unittest.TestCase):
    def setUp(self):
        REGISTRY.clear()

    def tearDown(self):
        REGISTRY.clear()

    def test_create_pending_review_endpoint(self):
        status, payload = handle_create_pending_review(
            {
                "review_id": "api-1",
                "automated_decision": {"status": "REVIEW"},
                "inspector_id": "insp-api",
                "reviewer_name": "Nia",
            }
        )
        self.assertEqual(201, status)
        self.assertTrue(payload["created"])
        self.assertEqual("PENDING", payload["inspector_decision"])
        self.assertFalse(payload["is_final_legal_decision"])

    def test_list_pending_reviews_endpoint(self):
        handle_create_pending_review(
            {
                "review_id": "api-list-1",
                "automated_decision": "REVIEW",
                "inspector_id": "insp-1",
                "reviewer_name": "Ada",
            }
        )
        handle_create_pending_review(
            {
                "review_id": "api-list-2",
                "automated_decision": "REVIEW",
                "inspector_id": "insp-2",
                "reviewer_name": "Bea",
            }
        )
        handle_confirm_review(
            "api-list-2",
            {"inspector_id": "insp-2", "reviewer_name": "Bea"},
        )
        status, payload = handle_list_pending_reviews()
        self.assertEqual(200, status)
        self.assertEqual(1, payload["count"])
        self.assertEqual("api-list-1", payload["reviews"][0]["review_id"])

    def test_retrieve_one_review_endpoint(self):
        handle_create_pending_review(
            {
                "review_id": "api-get",
                "automated_decision": "REVIEW",
                "inspector_id": "insp-g",
                "reviewer_name": "Gus",
            }
        )
        status, payload = handle_get_review("api-get")
        self.assertEqual(200, status)
        self.assertTrue(payload["found"])
        self.assertEqual("PENDING", payload["inspector_decision"])
        missing_status, missing = handle_get_review("no-such-review")
        self.assertEqual(200, missing_status)
        self.assertFalse(missing["found"])
        self.assertEqual("No inspector review exists for this item", missing["message"])

    def test_confirm_result_endpoint(self):
        handle_create_pending_review(
            {
                "review_id": "api-confirm",
                "automated_decision": {"status": "REVIEW", "mrp": "unclear"},
                "inspector_id": "insp-c",
                "reviewer_name": "Cid",
            }
        )
        status, payload = handle_confirm_review(
            "api-confirm",
            {
                "inspector_id": "insp-c",
                "reviewer_name": "Cid",
                "reason": "Matches the physical label",
                "field_confirmations": {"mrp": True},
            },
        )
        self.assertEqual(200, status)
        self.assertEqual("CONFIRMED", payload["inspector_decision"])
        self.assertEqual({"status": "REVIEW", "mrp": "unclear"}, payload["automated_decision"])
        self.assertEqual({"mrp": True}, payload["field_confirmations"])
        self.assertTrue(payload["is_final_legal_decision"])

    def test_override_with_reason_endpoint(self):
        handle_create_pending_review(
            {
                "review_id": "api-override",
                "automated_decision": "REVIEW",
                "inspector_id": "insp-o",
                "reviewer_name": "Ora",
            }
        )
        status, payload = handle_override_review(
            "api-override",
            {
                "inspector_id": "insp-o",
                "reviewer_name": "Ora",
                "reason": "MRP is present on the inner flap",
            },
        )
        self.assertEqual(200, status)
        self.assertEqual("OVERRIDDEN", payload["inspector_decision"])
        self.assertEqual("REVIEW", payload["automated_decision"])
        self.assertEqual("MRP is present on the inner flap", payload["reason"])

    def test_override_without_reason_is_rejected(self):
        handle_create_pending_review(
            {
                "review_id": "api-no-reason",
                "automated_decision": "REVIEW",
                "inspector_id": "insp-n",
                "reviewer_name": "Ned",
            }
        )
        status, payload = handle_override_review(
            "api-no-reason",
            {"inspector_id": "insp-n", "reviewer_name": "Ned", "reason": "  "},
        )
        self.assertEqual(400, status)
        self.assertFalse(payload["ok"])
        self.assertIn("override reason is required", payload["error"])
        found_status, found = handle_get_review("api-no-reason")
        self.assertEqual(200, found_status)
        self.assertEqual("PENDING", found["inspector_decision"])

    def test_reviewer_identity_and_timestamp_on_endpoints(self):
        _, created = handle_create_pending_review(
            {
                "review_id": "api-ident",
                "automated_decision": "REVIEW",
                "inspector_id": "insp-id",
                "reviewer_name": "Ida",
            }
        )
        self.assertEqual("insp-id", created["inspector_id"])
        self.assertEqual("Ida", created["reviewer_name"])
        self.assertTrue(created["created_at"])
        _, confirmed = handle_confirm_review(
            "api-ident",
            {"inspector_id": "insp-final", "reviewer_name": "Fay"},
        )
        self.assertEqual("insp-final", confirmed["inspector_id"])
        self.assertEqual("Fay", confirmed["reviewer_name"])
        self.assertEqual(created["created_at"], confirmed["created_at"])
        self.assertGreaterEqual(confirmed["reviewed_at"], created["reviewed_at"])

    def test_invalid_transitions_and_double_completion(self):
        handle_create_pending_review(
            {
                "review_id": "api-done",
                "automated_decision": "REVIEW",
                "inspector_id": "insp-x",
                "reviewer_name": "Xan",
            }
        )
        handle_confirm_review(
            "api-done",
            {"inspector_id": "insp-x", "reviewer_name": "Xan"},
        )
        again_status, again = handle_confirm_review(
            "api-done",
            {"inspector_id": "insp-x", "reviewer_name": "Xan"},
        )
        self.assertEqual(200, again_status)
        self.assertTrue(again["idempotent"])
        self.assertEqual(2, len(again["audit_trail"]))
        override_status, override_payload = handle_override_review(
            "api-done",
            {
                "inspector_id": "insp-x",
                "reviewer_name": "Xan",
                "reason": "Must not replace a completed confirmation",
            },
        )
        self.assertEqual(400, override_status)
        self.assertIn("already confirmed", override_payload["error"])
        missing_status, missing = handle_confirm_review(
            "missing-id",
            {"inspector_id": "insp-x", "reviewer_name": "Xan"},
        )
        self.assertEqual(400, missing_status)
        self.assertIn("No inspector review exists", missing["error"])

    def test_pass_does_not_create_review_via_endpoint(self):
        status, payload = handle_create_pending_review(
            {
                "review_id": "api-pass",
                "automated_decision": {"status": "PASS"},
                "inspector_id": "insp-p",
                "reviewer_name": "Pat",
            }
        )
        self.assertEqual(200, status)
        self.assertFalse(payload["created"])
        self.assertFalse(payload["review_required"])
        listed_status, listed = handle_list_pending_reviews()
        self.assertEqual(200, listed_status)
        self.assertEqual(0, listed["count"])
        get_status, fetched = handle_get_review("api-pass")
        self.assertEqual(200, get_status)
        self.assertFalse(fetched["found"])


if __name__ == "__main__":
    unittest.main()
