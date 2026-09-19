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

from typing import Any

from PIL import Image
import pytesseract
import cv2
import numpy as np

from quality_gate import assess_image_quality

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Keep this pass deliberately bounded: three overlapping regions recover small
# label print without turning one upload into dozens of OCR invocations.
TILE_ROWS = 3
TILE_COLUMNS = 1
TILE_OVERLAP = 0.18
TILE_SCALE = 2.0
# Package labels contain dense but irregular blocks rather than one paragraph;
# sparse-text segmentation keeps small adjacent declarations separate.
TILE_OCR_CONFIG = "--psm 11"

# Paddle is intentionally optional.  The production environment currently
# runs Python 3.14, for which the supported Paddle wheels are unavailable.
# Keeping this import and model construction lazy lets a supported deployment
# use Paddle as the primary engine without making app startup or Tesseract
# fallback depend on it.
_PADDLE_MODEL: Any | None = None
_PADDLE_INITIALIZED = False





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


def enhance_image(image_path: str, scale: float = 2.0) -> tuple[np.ndarray, float]:
    """Create a non-binary OCR variant while retaining the source scale."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(
            f"OpenCV could not read image at '{image_path}'. "
            "The file may be missing, corrupted, or in an unsupported format."
        )

    enlarged = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray), scale


def _ocr_words(
    image: Image.Image,
    scale: float = 1.0,
    offset_left: int = 0,
    offset_top: int = 0,
    config: str | None = None,
) -> list[dict]:
    """Extract word data and map any resized-image boxes to source coordinates."""
    options = {"output_type": pytesseract.Output.DICT}
    if config:
        options["config"] = config
    data = pytesseract.image_to_data(image, **options)
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue

        conf = float(data["conf"][i])
        if conf < 0:
            continue

        words.append(
            {
                "text": text,
                "conf": conf,
                "left": offset_left + round(data["left"][i] / scale),
                "top": offset_top + round(data["top"][i] / scale),
                "width": max(1, round(data["width"][i] / scale)),
                "height": max(1, round(data["height"][i] / scale)),
            }
        )
    return words


def _overlap_ratio(first: dict, second: dict) -> float:
    """Return intersection over the smaller box area for OCR de-duplication."""
    left = max(first["left"], second["left"])
    top = max(first["top"], second["top"])
    right = min(first["left"] + first["width"], second["left"] + second["width"])
    bottom = min(first["top"] + first["height"], second["top"] + second["height"])
    intersection = max(0, right - left) * max(0, bottom - top)
    smallest_area = min(
        first["width"] * first["height"], second["width"] * second["height"]
    )
    return intersection / smallest_area if smallest_area else 0.0


def _text_detail(text: str) -> int:
    """Prefer complete mixed tokens over shorter overlapping OCR fragments."""
    alphanumeric = "".join(character for character in text if character.isalnum())
    has_letters = any(character.isalpha() for character in alphanumeric)
    has_digits = any(character.isdigit() for character in alphanumeric)
    return len(alphanumeric) + int(has_letters and has_digits)


def _merge_words(*word_sets: list[dict]) -> list[dict]:
    """Keep the most confident reading when both passes identify one region."""
    merged = []
    for word_set in word_sets:
        for candidate in word_set:
            normalized = candidate["text"].casefold()
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if _overlap_ratio(candidate, existing) >= 0.6
                    or (
                        normalized == existing["text"].casefold()
                        and _overlap_ratio(candidate, existing) >= 0.2
                    )
                ),
                None,
            )
            if duplicate_index is None:
                merged.append(candidate)
            elif (
                candidate["conf"] > merged[duplicate_index]["conf"]
                and _text_detail(candidate["text"])
                >= _text_detail(merged[duplicate_index]["text"])
            ):
                merged[duplicate_index] = candidate
    return sorted(merged, key=lambda word: (word["top"], word["left"]))


def _tile_regions(
    width: int,
    height: int,
    rows: int = TILE_ROWS,
    columns: int = TILE_COLUMNS,
    overlap: float = TILE_OVERLAP,
) -> list[tuple[int, int, int, int]]:
    """Return a small overlapping grid of source-coordinate tile regions."""
    regions = []
    base_width = max(1, int(np.ceil(width / columns)))
    base_height = max(1, int(np.ceil(height / rows)))
    overlap_x = int(round(base_width * overlap))
    overlap_y = int(round(base_height * overlap))
    for row in range(rows):
        for column in range(columns):
            cell_left = column * base_width
            cell_top = row * base_height
            cell_right = min(width, (column + 1) * base_width)
            cell_bottom = min(height, (row + 1) * base_height)
            left = max(0, cell_left - overlap_x)
            top = max(0, cell_top - overlap_y)
            right = min(width, cell_right + overlap_x)
            bottom = min(height, cell_bottom + overlap_y)
            if right > left and bottom > top:
                regions.append((left, top, right - left, bottom - top))
    return regions


def _clamp_words(words: list[dict], width: int, height: int) -> list[dict]:
    """Keep all returned boxes within the original uploaded image."""
    clamped = []
    for word in words:
        left = max(0, min(int(word["left"]), width - 1))
        top = max(0, min(int(word["top"]), height - 1))
        right = min(width, max(left + 1, int(word["left"] + word["width"])))
        bottom = min(height, max(top + 1, int(word["top"] + word["height"])))
        if right <= left or bottom <= top:
            continue
        clamped.append({**word, "left": left, "top": top, "width": right - left, "height": bottom - top})
    return clamped


def _tiled_enhanced_words(image_path: str) -> list[dict]:
    """OCR a bounded overlapping grid after non-destructive local enhancement."""
    source = cv2.imread(image_path)
    if source is None:
        raise ValueError(f"OpenCV could not read image at '{image_path}'.")

    height, width = source.shape[:2]
    tiled_words = []
    for left, top, tile_width, tile_height in _tile_regions(width, height):
        tile = source[top : top + tile_height, left : left + tile_width]
        enlarged = cv2.resize(tile, None, fx=TILE_SCALE, fy=TILE_SCALE, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_tile = clahe.apply(gray)
        tiled_words.extend(
            _ocr_words(
                Image.fromarray(enhanced_tile),
                scale=TILE_SCALE,
                offset_left=left,
                offset_top=top,
                config=TILE_OCR_CONFIG,
            )
        )
    return tiled_words


def _words_to_text(words: list[dict]) -> str:
    """Rebuild readable lines from source-coordinate word detections."""
    lines = []
    for word in words:
        word_center = word["top"] + word["height"] / 2
        if lines and abs(word_center - lines[-1]["center"]) <= max(8, word["height"]):
            line = lines[-1]
            line["words"].append(word)
            line["center"] = sum(
                item["top"] + item["height"] / 2 for item in line["words"]
            ) / len(line["words"])
        else:
            lines.append({"center": word_center, "words": [word]})

    return "\n".join(
        " ".join(item["text"] for item in sorted(line["words"], key=lambda item: item["left"]))
        for line in lines
    )


def _get_paddle_model() -> Any | None:
    """Create one optional PaddleOCR model on first eligible OCR request.

    Importing PaddleOCR or constructing its model is deliberately deferred:
    quality review must finish first, and the API must remain usable when the
    optional dependency is absent.  A process restart is required to retry a
    failed optional initialisation, which avoids repeated imports/download
    attempts for every uploaded image.
    """
    global _PADDLE_INITIALIZED, _PADDLE_MODEL
    if _PADDLE_INITIALIZED:
        return _PADDLE_MODEL

    _PADDLE_INITIALIZED = True
    try:
        from paddleocr import PaddleOCR

        _PADDLE_MODEL = PaddleOCR(lang="en")
    except Exception:
        # Paddle is an enhancement, not an availability requirement.  The
        # caller will use the existing Tesseract pipeline instead.
        _PADDLE_MODEL = None
    return _PADDLE_MODEL


def _paddle_word(item: Any, image_width: int, image_height: int) -> dict | None:
    """Convert PaddleOCR's polygon/result pair to the public word contract."""
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    polygon, recognition = item[0], item[1]
    if not isinstance(polygon, (list, tuple)) or not isinstance(recognition, (list, tuple)):
        return None
    if len(recognition) < 2 or not isinstance(recognition[0], str):
        return None
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon if len(point) >= 2]
        if not points:
            return None
        text = recognition[0].strip()
        if not text:
            return None
        confidence = float(recognition[1])
    except (TypeError, ValueError, IndexError):
        return None

    left = int(min(point[0] for point in points))
    top = int(min(point[1] for point in points))
    right = int(max(point[0] for point in points))
    bottom = int(max(point[1] for point in points))
    # Paddle exposes a 0..1 confidence while the existing contract uses
    # Tesseract's 0..100 scale.
    if confidence <= 1:
        confidence *= 100
    return {
        "text": text,
        "conf": confidence,
        "left": left,
        "top": top,
        "width": max(1, right - left),
        "height": max(1, bottom - top),
    }


def _paddle_result_words(result: Any, image_width: int, image_height: int) -> list[dict]:
    """Extract recognized text lines from Paddle's documented ``ocr`` result."""
    if not isinstance(result, (list, tuple)):
        return []
    # The common PaddleOCR result is a list of pages, each a list of
    # ``[polygon, (text, confidence)]`` records.  Flatten pages only; never
    # merge them with a second engine's output.
    pages = result
    if pages and isinstance(pages[0], (list, tuple)) and pages[0] and _paddle_word(
        pages[0], image_width, image_height
    ):
        pages = [pages]

    words = []
    for page in pages:
        if not isinstance(page, (list, tuple)):
            continue
        for item in page:
            word = _paddle_word(item, image_width, image_height)
            if word is not None:
                words.append(word)
    return _clamp_words(words, image_width, image_height)


def _paddle_words(image_path: str) -> list[dict]:
    """Run the primary provider, returning no words when it is unavailable."""
    model = _get_paddle_model()
    if model is None:
        return []
    try:
        with Image.open(image_path) as source:
            width, height = source.size
        result = model.ocr(image_path, cls=False)
        return _paddle_result_words(result, width, height)
    except Exception:
        return []


def _tesseract_words(image_path: str) -> list[dict]:
    """Run the established Tesseract flow as a single-provider fallback."""
    original_image = Image.open(image_path)
    original_words = _ocr_words(original_image)

    enhanced_image, scale = enhance_image(image_path)
    enhanced_words = _ocr_words(Image.fromarray(enhanced_image), scale=scale)
    tiled_words = _tiled_enhanced_words(image_path)

    return _clamp_words(
        _merge_words(original_words, enhanced_words, tiled_words),
        *original_image.size,
    )


def extract_text(image_path: str, use_preprocessing: bool = False) -> dict:
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
    # Assess the original uploaded file before any OCR-specific image opening
    # or transformation. Review is advisory: no text is inferred as absent.
    quality = assess_image_quality(image_path)
    if quality["recommendation"] != "PROCEED":
        return {"full_text": "", "words": []}

    # Paddle is primary when it is available and produces usable evidence.
    # Tesseract is called only when Paddle is unavailable, fails, or returns
    # no words; blending provider output would duplicate text and corrupt the
    # source-coordinate evidence contract.
    words = _paddle_words(image_path)
    if not words:
        words = _tesseract_words(image_path)
    return {"full_text": _words_to_text(words), "words": words}


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
