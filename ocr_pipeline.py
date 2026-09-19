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
import os
import re
import time
from threading import Lock

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
_PADDLE_ERROR: str | None = None
_PADDLE_LOCK = Lock()
MAX_IMAGE_SIDE = 2600
SMALL_TEXT_HEIGHT = 18
MAX_PADDLE_PASSES = 3





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
                    if (
                        normalized == existing["text"].casefold()
                        and _overlap_ratio(candidate, existing) >= 0.2
                    ) or (
                        _overlap_ratio(candidate, existing) >= 0.7
                        and _text_similarity(normalized, existing["text"].casefold()) >= 0.75
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


def _text_similarity(first: str, second: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, first, second).ratio()


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
    global _PADDLE_INITIALIZED, _PADDLE_MODEL, _PADDLE_ERROR
    with _PADDLE_LOCK:
        if _PADDLE_INITIALIZED:
            return _PADDLE_MODEL
        _PADDLE_INITIALIZED = True
        try:
            # PaddlePaddle 3.3.0 on Windows has a oneDNN conversion failure
            # with this model.  The supported plain CPU runner was verified
            # against real images before selecting it here.
            os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
            from paddleocr import PaddleOCR

            _PADDLE_MODEL = PaddleOCR(
                device="cpu",
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            _PADDLE_ERROR = None
        except Exception as exc:
            _PADDLE_MODEL = None
            _PADDLE_ERROR = f"Paddle initialization failed: {type(exc).__name__}: {exc}"
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
    """Extract recognized text lines from PaddleOCR 3.x or legacy output."""
    if not isinstance(result, (list, tuple)):
        return []
    if result and hasattr(result[0], "json"):
        words = []
        for page in result:
            payload = page.json.get("res", page.json)
            for text, score, polygon in zip(
                payload.get("rec_texts", ()),
                payload.get("rec_scores", ()),
                payload.get("rec_polys", ()),
            ):
                points = [(float(point[0]), float(point[1])) for point in polygon]
                if not points or not str(text).strip():
                    continue
                left, top = int(min(x for x, _ in points)), int(min(y for _, y in points))
                right, bottom = int(max(x for x, _ in points)), int(max(y for _, y in points))
                parts = list(re.finditer(r"\S+", str(text)))
                for part in parts:
                    # Paddle returns line polygons.  Allocate a proportional
                    # rectangle for each token while retaining the source line.
                    token_left = left + round((right - left) * part.start() / len(text))
                    token_right = left + round((right - left) * part.end() / len(text))
                    words.append({
                        "text": part.group(), "conf": float(score) * 100,
                        "left": token_left, "top": top,
                        "width": max(1, token_right - token_left),
                        "height": max(1, bottom - top),
                        "provider": "paddle", "original_text": str(text),
                    })
        return _clamp_words(words, image_width, image_height)
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
    with Image.open(image_path) as source:
        width, height = source.size
    result = model.predict(image_path)
    return _paddle_result_words(result, width, height)


def _paddle_variant_words(image: np.ndarray, scale: float, offset: tuple[int, int] = (0, 0)) -> list[dict]:
    """Run a bounded enhanced pass and map each box to original pixels."""
    model = _get_paddle_model()
    if model is None:
        return []
    height, width = image.shape[:2]
    words = _paddle_result_words(model.predict(image), width, height)
    mapped = []
    for word in words:
        mapped.append({**word,
            "left": offset[0] + round(word["left"] / scale),
            "top": offset[1] + round(word["top"] / scale),
            "width": max(1, round(word["width"] / scale)),
            "height": max(1, round(word["height"] / scale)),
        })
    return mapped


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


def quality_allows_ocr(quality: dict) -> bool:
    """Resolution-only warnings may yield evidence, but never a legal PASS.

    The quality decision remains REVIEW downstream. An otherwise readable
    narrow panel should not lose all visible text simply because one image
    dimension is below the advisory resolution threshold.
    """
    if quality.get("recommendation") == "PROCEED":
        return True
    checks = quality.get("checks") or {}
    return bool(checks and all(
        checks.get(name, {}).get("status") == "GOOD"
        for name in ("blur", "brightness", "glare", "readability")
    ) and checks.get("resolution", {}).get("status") in {"WARNING", "POOR"})


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
    started = time.monotonic()
    metadata = {"providers_used": [], "elapsed_seconds": 0.0,
                "word_count": 0, "variants": [], "errors": [], "fallback_reasons": []}
    quality = assess_image_quality(image_path)
    if not quality_allows_ocr(quality):
        metadata["fallback_reasons"].append("Image quality requires review before OCR")
        metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return {"full_text": "", "words": [], "metadata": metadata}
    if quality["recommendation"] != "PROCEED":
        metadata["fallback_reasons"].append("Resolution-only quality review; OCR is advisory")

    # Paddle is primary when it is available and produces usable evidence.
    # Tesseract is called only when Paddle is unavailable, fails, or returns
    # no words; blending provider output would duplicate text and corrupt the
    # source-coordinate evidence contract.
    words = []
    try:
        words = _paddle_words(image_path)
        if words:
            metadata["providers_used"].append("paddle")
            metadata["variants"].append("original")
        elif _PADDLE_ERROR:
            metadata["errors"].append(_PADDLE_ERROR)
    except Exception as exc:
        metadata["errors"].append(f"Paddle inference failed: {type(exc).__name__}: {exc}")

    if words and os.path.exists(image_path):
        try:
            source = cv2.imread(image_path)
            if source is not None:
                height, width = source.shape[:2]
                median_height = sorted(word["height"] for word in words)[len(words) // 2]
                # Small print gets one scaled and contrast-enhanced pass.
                if (median_height < SMALL_TEXT_HEIGHT or len(words) < 25) and max(width, height) < MAX_IMAGE_SIDE:
                    scale = min(2.0, MAX_IMAGE_SIDE / max(width, height))
                    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
                    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                    enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    recovered = _paddle_variant_words(enhanced, scale)
                    words = _merge_words(words, _clamp_words(recovered, width, height))
                    metadata["variants"].append("clahe_upscaled")
                # One additional pass on a large image's lower panel recovers
                # dense declarations without a costly unbounded tile grid.
                if height >= 1500 and len(metadata["variants"]) < MAX_PADDLE_PASSES:
                    top = height // 2
                    tile = source[top:height]
                    recovered = _paddle_variant_words(tile, 1.0, (0, top))
                    words = _merge_words(words, _clamp_words(recovered, width, height))
                    metadata["variants"].append("lower_panel")
        except Exception as exc:
            metadata["errors"].append(f"Paddle enhancement failed: {type(exc).__name__}: {exc}")

    # Recover weak or missing declarations with the established secondary
    # provider.  Duplicate regions are resolved by geometry and confidence;
    # distinct text at different coordinates remains available to reviewers.
    lower_text = " ".join(word["text"] for word in words).casefold()
    needs_recovery = (not words or len(words) < 25 or
                      not any(marker in lower_text for marker in ("mrp", "retail price", "net weight", "net wt", "net qty")))
    if needs_recovery:
        metadata["fallback_reasons"].append("Paddle unavailable or declaration coverage weak")
        try:
            tesseract_words = [
                {**word, "provider": "tesseract"}
                for word in _tesseract_words(image_path)
                if float(word.get("conf", 0)) >= 55 and
                any(character.isalnum() for character in word.get("text", ""))
            ]
            if tesseract_words:
                metadata["providers_used"].append("tesseract")
                metadata["variants"].append("tesseract_original_enhanced_tiles")
                words = _merge_words(words, tesseract_words)
        except Exception as exc:
            metadata["errors"].append(f"Tesseract failed: {type(exc).__name__}: {exc}")
    metadata["word_count"] = len(words)
    metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return {"full_text": _words_to_text(words), "words": words, "metadata": metadata}


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
