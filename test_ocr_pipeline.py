import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import ocr_pipeline
from compliance_checker import check_fields


class OcrPipelineTests(unittest.TestCase):
    def test_tile_word_coordinates_map_back_to_original_image(self):
        data = {"text": ["small"], "conf": ["92.5"], "left": [20], "top": [10], "width": [30], "height": [12]}
        with patch("ocr_pipeline.pytesseract.image_to_data", return_value=data):
            words = ocr_pipeline._ocr_words(Image.new("L", (100, 100)), scale=2, offset_left=300, offset_top=400)
        self.assertEqual(words[0], {"text": "small", "conf": 92.5, "left": 310, "top": 405, "width": 15, "height": 6})

    def test_overlapping_tiles_keep_the_highest_confidence_word(self):
        first = {"text": "Weight", "conf": 55.0, "left": 100, "top": 100, "width": 40, "height": 12}
        overlap = {"text": "Weight", "conf": 91.0, "left": 102, "top": 100, "width": 40, "height": 12}
        separate = {"text": "Weight", "conf": 80.0, "left": 300, "top": 100, "width": 40, "height": 12}
        merged = ocr_pipeline._merge_words([first], [overlap, separate])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["conf"], 91.0)

    def test_return_structure_is_preserved(self):
        image = Image.new("RGB", (40, 40))
        word = {"text": "Label", "conf": 90.0, "left": 2, "top": 3, "width": 10, "height": 8}
        with patch("ocr_pipeline.assess_image_quality", return_value={"recommendation": "PROCEED"}), patch("ocr_pipeline.Image.open", return_value=image), patch("ocr_pipeline.enhance_image", return_value=(np.zeros((80, 80), dtype=np.uint8), 2.0)), patch("ocr_pipeline._ocr_words", side_effect=[[word], []]), patch("ocr_pipeline._tiled_enhanced_words", return_value=[]):
            result = ocr_pipeline.extract_text("example.jpeg")
        self.assertEqual(set(result), {"full_text", "words"})
        self.assertEqual(result["words"], [word])
        self.assertIsInstance(result["full_text"], str)

    def test_review_quality_skips_all_ocr_processing(self):
        with patch("ocr_pipeline.assess_image_quality", return_value={"recommendation": "REVIEW"}), patch("ocr_pipeline.Image.open") as image_open, patch("ocr_pipeline._ocr_words") as ocr_words, patch("ocr_pipeline.enhance_image") as enhance, patch("ocr_pipeline._tiled_enhanced_words") as tiled:
            result = ocr_pipeline.extract_text("poor-image.jpg")
        self.assertEqual(result, {"full_text": "", "words": []})
        image_open.assert_not_called()
        ocr_words.assert_not_called()
        enhance.assert_not_called()
        tiled.assert_not_called()

    def test_quality_review_cannot_become_a_violation(self):
        with patch("ocr_pipeline.assess_image_quality", return_value={"recommendation": "REVIEW"}):
            result = ocr_pipeline.extract_text("poor-image.jpg")
        report = check_fields(result, package_sides_complete=True, ocr_quality_sufficient=False)
        self.assertEqual(report["overall_result"], "REVIEW")

    def test_clamped_boxes_stay_inside_image_bounds(self):
        words = ocr_pipeline._clamp_words(
            [{"text": "edge", "conf": 80.0, "left": 95, "top": 96, "width": 20, "height": 20}],
            100,
            100,
        )
        self.assertEqual(words[0]["left"], 95)
        self.assertEqual(words[0]["top"], 96)
        self.assertEqual(words[0]["width"], 5)
        self.assertEqual(words[0]["height"], 4)


if __name__ == "__main__":
    unittest.main()
