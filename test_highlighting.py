import unittest

from PIL import Image
from plugins.highlighting import STATUS_COLORS, _draw_decisions


class HighlightingTests(unittest.TestCase):
    def test_draws_valid_bbox_with_status_color(self):
        image = Image.new("RGB", (200, 120), "white")
        overlays = _draw_decisions(image, [{"field_key": "net_quantity", "status": "PASS", "extracted_value": "100g", "bbox": {"left": 20, "top": 40, "width": 50, "height": 20}}])
        self.assertEqual(len(overlays), 1)
        self.assertEqual(overlays[0]["color"], STATUS_COLORS["PASS"])
        self.assertEqual(image.getpixel((20, 40)), STATUS_COLORS["PASS"])

    def test_review_and_violation_use_their_status_colors(self):
        image = Image.new("RGB", (200, 120), "white")
        overlays = _draw_decisions(image, [
            {"field_key": "manufacturer_or_packer", "status": "REVIEW", "bbox": {"left": 10, "top": 20, "width": 30, "height": 15}},
            {"field_key": "mrp", "status": "VIOLATION", "bbox": {"left": 80, "top": 20, "width": 30, "height": 15}},
        ])
        self.assertEqual(overlays[0]["color"], STATUS_COLORS["REVIEW"])
        self.assertEqual(overlays[1]["color"], STATUS_COLORS["VIOLATION"])

    def test_skips_missing_bbox_without_fabricating_box(self):
        image = Image.new("RGB", (200, 120), "white")
        overlays = _draw_decisions(image, [{"field_key": "consumer_care", "status": "REVIEW", "extracted_value": None, "bbox": None}])
        self.assertEqual(overlays, [])
        self.assertEqual(image.getpixel((100, 60)), (255, 255, 255))

    def test_edge_labels_and_boxes_remain_inside_image(self):
        image = Image.new("RGB", (120, 70), "white")
        overlays = _draw_decisions(image, [{"field_key": "manufacture_or_packing_date", "status": "VIOLATION", "extracted_value": "22/07/26", "bbox": {"left": 110, "top": 58, "width": 20, "height": 20}}])
        box, label = overlays[0]["box"], overlays[0]["label_box"]
        self.assertTrue(all(value >= 0 for value in box + label))
        self.assertLess(box[2], image.width)
        self.assertLess(box[3], image.height)
        self.assertLessEqual(label[2], image.width)
        self.assertLessEqual(label[3], image.height)

    def test_nearby_evidence_labels_do_not_overlap(self):
        image = Image.new("RGB", (300, 120), "white")
        overlays = _draw_decisions(image, [
            {"field_key": "generic_product_name", "status": "PASS", "extracted_value": "BISCUITS", "bbox": {"left": 20, "top": 55, "width": 80, "height": 20}},
            {"field_key": "net_quantity", "status": "PASS", "extracted_value": "100g", "bbox": {"left": 120, "top": 55, "width": 50, "height": 20}},
        ])
        first, second = (overlay["label_box"] for overlay in overlays)
        self.assertTrue(first[2] <= second[0] or second[2] <= first[0] or first[3] <= second[1] or second[3] <= first[1])


if __name__ == "__main__":
    unittest.main()
