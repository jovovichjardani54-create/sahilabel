import unittest

from plugins.scoring import compute_score


class ScoringTests(unittest.TestCase):
    def test_review_is_pending_not_a_zero_or_grade_f(self):
        score = compute_score({
            "overall_result": "REVIEW",
            "fields": {},
            "readability_flags": [{"reason": "Retake image"}],
        })

        self.assertTrue(score["pending_review"])
        self.assertIsNone(score["score"])
        self.assertIsNone(score["grade"])
        self.assertEqual(score["message"], "Pending review")

    def test_pass_still_has_numeric_score_and_grade(self):
        score = compute_score({
            "overall_result": "PASS",
            "fields": {"mrp": {"severity": "critical", "present": True}},
        })

        self.assertFalse(score["pending_review"])
        self.assertEqual(score["score"], 80)
        self.assertEqual(score["grade"], "B")


if __name__ == "__main__":
    unittest.main()
