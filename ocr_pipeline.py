"""
ocr_pipeline.py
----------------
Wraps Tesseract OCR to extract:
  1. Full text of the label (for regex field-matching)
  2. Per-word bounding boxes + confidence (for the readability/font-size
     heuristic and for highlighting detected regions on the image)

Team notes:
- Swap `pytesseract` for `easyocr` later if you find it handles skewed /
  low-quality photos better - the rest of the pipeline (rules_config,
  compliance_checker) doesn't care which OCR engine produced the text,
  as long as you keep returning the same dict shape from extract_text().
"""

from PIL import Image
import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'





def preprocess_image(image_path: str) -> np.ndarray:
    """Basic preprocessing to help OCR accuracy on real product photos:
    grayscale + adaptive threshold. Tune this as you test on real photos."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(
            f"OpenCV could not read image at '{image_path}'. "
            "The file may be missing, corrupted, or in an unsupported format."
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold helps with uneven lighting on real label photos
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return thresh


def extract_text(image_path: str, use_preprocessing: bool = True) -> dict:
    """
    Returns:
        {
            "full_text": str,
            "words": [
                {"text": str, "conf": float, "left": int, "top": int,
                 "width": int, "height": int},
                ...
            ]
        }
    """
    if use_preprocessing:
        processed = preprocess_image(image_path)
        pil_img = Image.fromarray(processed)
    else:
        pil_img = Image.open(image_path)

    full_text = pytesseract.image_to_string(pil_img)

    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < 0:  # -1 means no confidence value (non-text region)
            continue
        words.append(
            {
                "text": text,
                "conf": conf,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            }
        )

    return {"full_text": full_text, "words": words}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_pipeline.py <image_path>")
        sys.exit(1)

    result = extract_text(sys.argv[1])
    print("---- FULL TEXT ----")
    print(result["full_text"])
    print(f"---- {len(result['words'])} WORDS DETECTED ----")
    for w in result["words"][:20]:
        print(w)
