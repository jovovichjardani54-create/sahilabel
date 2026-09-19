import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import main
from inspector_review import REGISTRY
from plugins import highlighting


def words_from_lines(*lines):
    words = []
    for row, line in enumerate(lines, start=1):
        left = 10
        for text, confidence in line:
            words.append({
                "text": text,
                "conf": confidence,
                "left": left,
                "top": row * 30,
                "width": max(8, len(text) * 8),
                "height": 14,
            })
            left += max(8, len(text) * 8) + 5
    return {"full_text": " ".join(word["text"] for word in words), "words": words}


def image_bytes():
    image = Image.new("RGB", (100, 100), "white")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    return payload.getvalue()


GOOD_QUALITY = {
    "quality_status": "GOOD",
    "quality_score": 100.0,
    "checks": {},
    "explanation": "Image quality is good and the label is likely readable.",
    "recommendation": "PROCEED",
}
REVIEW_QUALITY = {
    "quality_status": "POOR",
    "quality_score": 0.0,
    "checks": {},
    "explanation": "Image quality needs review.",
    "recommendation": "REVIEW",
}


class MainIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.original_database_path = main.HISTORY_STORE.database_path
        self.original_upload_dir = main.UPLOAD_DIR
        self.original_report_dir = main.REPORT_DIR
        self.original_annotated_dir = highlighting.ANNOTATED_DIR
        main.HISTORY_STORE.database_path = str(root / "history.sqlite3")
        main.HISTORY_STORE._create_table()
        main.UPLOAD_DIR = str(root / "uploads")
        main.REPORT_DIR = str(root / "reports")
        highlighting.ANNOTATED_DIR = str(root / "annotated")
        for directory in (main.UPLOAD_DIR, main.REPORT_DIR, highlighting.ANNOTATED_DIR):
            Path(directory).mkdir(parents=True, exist_ok=True)
        main.HISTORY.clear()
        main.PRODUCT_RECORDS.clear()
        main._PLUGIN_STORE.clear()
        REGISTRY.clear()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        main.HISTORY.clear()
        main.PRODUCT_RECORDS.clear()
        main._PLUGIN_STORE.clear()
        REGISTRY.clear()
        main.HISTORY_STORE.database_path = self.original_database_path
        main.UPLOAD_DIR = self.original_upload_dir
        main.REPORT_DIR = self.original_report_dir
        highlighting.ANNOTATED_DIR = self.original_annotated_dir
        self.temp_dir.cleanup()

    def test_multiview_check_uses_back_and_side_source_evidence(self):
        front = words_from_lines(
            [("Marketed", 95), ("by", 95), ("Example", 95), ("Foods", 95), ("Ltd", 95)],
            [("BISCUITS", 95)],
        )
        back = words_from_lines(
            [("Net", 96), ("Weight", 96), ("100", 96), ("g", 96)],
            [("MRP", 96), ("45.00", 96)],
            [("PKD", 96), ("22/07/26", 96)],
        )
        side = words_from_lines(
            [("Customer", 97), ("Care", 97), ("1800123456", 97)],
        )
        with patch("main.assess_image_quality", return_value=GOOD_QUALITY), patch(
            "main.extract_text", side_effect=[front, back, side]
        ):
            response = self.client.post(
                "/check",
                data={"image_labels": ["front", "back", "side"]},
                files=[
                    ("files", ("front.png", io.BytesIO(image_bytes()), "image/png")),
                    ("files", ("back.png", io.BytesIO(image_bytes()), "image/png")),
                    ("files", ("side.png", io.BytesIO(image_bytes()), "image/png")),
                ],
            )

        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertEqual(record["product_record"]["coverage_status"], "complete")
        self.assertEqual(record["report"]["overall_result"], "PASS")
        decisions = record["report"]["field_decisions"]
        self.assertEqual(decisions["mrp"]["source_evidence"]["source_label"], "back")
        self.assertEqual(decisions["consumer_care"]["source_evidence"]["source_label"], "side")
        self.assertEqual(set(record["annotated_evidence"]), {"front", "back", "side"})

        item_id = record["id"]
        self.assertEqual(self.client.get(f"/annotated/{item_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/annotated/{item_id}/back").status_code, 200)
        self.assertEqual(self.client.get(f"/annotated/{item_id}/side").status_code, 200)
        self.assertEqual(self.client.get(f"/report/{item_id}/pdf").status_code, 200)
        self.assertEqual(self.client.get("/history/search", params={"query": item_id}).status_code, 200)
        self.assertEqual(
            self.client.get(f"/analytics/history/reports/{item_id}").status_code, 200
        )
        dashboard = self.client.get("/dashboard/stats").json()
        self.assertGreaterEqual(dashboard["persistent_analytics"]["total_inspections"], 1)

    def test_poor_image_stays_review_and_uses_router_backed_inspection(self):
        with patch("main.assess_image_quality", return_value=REVIEW_QUALITY), patch(
            "main.extract_text"
        ) as extract_text:
            response = self.client.post(
                "/check",
                data={"inspector_id": "insp-1"},
                files={"file": ("poor.png", io.BytesIO(image_bytes()), "image/png")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertEqual(record["report"]["overall_result"], "REVIEW")
        self.assertEqual(record["inspector_review"]["inspector_decision"], "PENDING")
        extract_text.assert_not_called()

        item_id = record["id"]
        score = self.client.get(f"/score/{item_id}").json()
        self.assertTrue(score["pending_review"])
        self.assertIsNone(score["score"])
        self.assertIsNone(score["grade"])
        confirmed = self.client.post(
            f"/inspector-reviews/{item_id}/confirm",
            json={"inspector_id": "insp-1", "reviewer_name": "Asha"},
        )
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        self.assertEqual(confirmed.json()["inspector_decision"], "CONFIRMED")

        created = self.client.post(
            "/inspector-reviews",
            json={
                "review_id": "manual-override",
                "automated_decision": "REVIEW",
                "inspector_id": "insp-2",
                "reviewer_name": "Bala",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.assertTrue(created.json()["created"])
        overridden = self.client.post(
            "/inspector-reviews/manual-override/override",
            json={
                "inspector_id": "insp-2",
                "reviewer_name": "Bala",
                "reason": "Verified against the physical package.",
            },
        )
        self.assertEqual(overridden.status_code, 200, overridden.text)
        self.assertEqual(overridden.json()["inspector_decision"], "OVERRIDDEN")

    def test_html_dashboard_and_camera_ui_keep_json_analytics_available(self):
        index = self.client.get("/")
        dashboard = self.client.get("/analytics")
        camera_script = self.client.get("/static/camera.js")

        self.assertEqual(index.status_code, 200)
        self.assertIn('data-camera-open="front"', index.text)
        self.assertIn("Inspector review", index.text)
        self.assertIn("Compliance score:</strong> Pending review", index.text)
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("Inspection analytics", dashboard.text)
        self.assertIn("/analytics/history/analytics", dashboard.text)
        self.assertEqual(camera_script.status_code, 200)
        self.assertIn("navigator.mediaDevices.getUserMedia", camera_script.text)
        self.assertEqual(self.client.get("/analytics/history/analytics").status_code, 200)


if __name__ == "__main__":
    unittest.main()
