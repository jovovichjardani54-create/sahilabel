import unittest

import cv2
import numpy as np

from quality_gate import GOOD, POOR, REVIEW, WARNING, assess_image_quality


def readable_label(width=1600, height=1200):
    image = np.full((height, width, 3), 170, dtype=np.uint8)
    cv2.putText(image, "NET WEIGHT 500g", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (15, 15, 15), 8)
    cv2.putText(image, "MRP Rs. 45.00", (100, 650), cv2.FONT_HERSHEY_SIMPLEX, 3, (15, 15, 15), 8)
    return image


class ImageQualityGateTests(unittest.TestCase):
    def test_good_image(self):
        result = assess_image_quality(readable_label())
        self.assertEqual(result["quality_status"], GOOD)
        self.assertEqual(result["recommendation"], "PROCEED")

    def test_blurry_image_requires_review(self):
        result = assess_image_quality(cv2.GaussianBlur(readable_label(), (51, 51), 0))
        self.assertIn(result["quality_status"], (WARNING, POOR))
        self.assertNotEqual(result["checks"]["blur"]["status"], GOOD)
        self.assertEqual(result["recommendation"], REVIEW)

    def test_tiny_image_is_poor(self):
        result = assess_image_quality(cv2.resize(readable_label(), (200, 150)))
        self.assertEqual(result["quality_status"], POOR)
        self.assertEqual(result["checks"]["resolution"]["status"], POOR)
        self.assertEqual(result["recommendation"], REVIEW)

    def test_dark_image_is_poor(self):
        dark = np.full((1200, 1600, 3), 20, dtype=np.uint8)
        cv2.putText(dark, "LABEL", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (35, 35, 35), 8)
        result = assess_image_quality(dark)
        self.assertEqual(result["quality_status"], POOR)
        self.assertEqual(result["checks"]["brightness"]["status"], POOR)
        self.assertEqual(result["recommendation"], REVIEW)

    def test_overexposed_image_is_poor(self):
        result = assess_image_quality(np.full((1200, 1600, 3), 255, dtype=np.uint8))
        self.assertEqual(result["quality_status"], POOR)
        self.assertEqual(result["checks"]["glare"]["status"], POOR)
        self.assertEqual(result["recommendation"], REVIEW)


if __name__ == "__main__":
    unittest.main()
