import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

    def test_borderline_blur_requires_review(self):
        # The mocked Laplacian must have variance 50, not a constant value.
        with patch("quality_gate.cv2.Laplacian", return_value=np.array([[0.0, 10.0], [10.0, 20.0]])):
            result = assess_image_quality(readable_label())
        self.assertEqual(result["checks"]["blur"]["status"], WARNING)
        self.assertEqual(result["recommendation"], REVIEW)

    def test_tiny_image_is_poor(self):
        result = assess_image_quality(cv2.resize(readable_label(), (200, 150)))
        self.assertEqual(result["quality_status"], POOR)
        self.assertEqual(result["checks"]["resolution"]["status"], POOR)
        self.assertEqual(result["recommendation"], REVIEW)

    def test_borderline_resolution_requires_review(self):
        result = assess_image_quality(readable_label(800, 800))
        self.assertEqual(result["checks"]["resolution"]["status"], WARNING)
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

    def test_glare_heavy_image_requires_review(self):
        image = readable_label()
        image[:600, :800] = 255
        result = assess_image_quality(image)
        self.assertIn(result["checks"]["glare"]["status"], (WARNING, POOR))
        self.assertEqual(result["recommendation"], REVIEW)

    def test_low_contrast_image_requires_review(self):
        result = assess_image_quality(np.full((1200, 1600, 3), 150, dtype=np.uint8))
        self.assertEqual(result["checks"]["readability"]["status"], POOR)
        self.assertEqual(result["recommendation"], REVIEW)

    def test_combined_poor_quality_image_requires_review(self):
        result = assess_image_quality(np.full((150, 200, 3), 20, dtype=np.uint8))
        self.assertEqual(result["quality_status"], POOR)
        self.assertEqual(result["recommendation"], REVIEW)

    def test_corrupt_image_is_reviewed_safely(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.jpg"
            path.write_bytes(b"not an image")
            result = assess_image_quality(path)
        self.assertEqual(result["recommendation"], REVIEW)

    def test_missing_and_invalid_images_are_reviewed_safely(self):
        self.assertEqual(assess_image_quality("not-a-real-image.jpg")["recommendation"], REVIEW)
        self.assertEqual(assess_image_quality(np.array([], dtype=np.uint8))["recommendation"], REVIEW)
        self.assertEqual(assess_image_quality(np.zeros((20, 20, 2), dtype=np.uint8))["recommendation"], REVIEW)


if __name__ == "__main__":
    unittest.main()
