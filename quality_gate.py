"""Advisory image-quality checks for product-label photographs.

This module is independent of OCR and compliance assessment. A non-GOOD result
only asks for human review; it never represents a violation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np


GOOD = "GOOD"
WARNING = "WARNING"
POOR = "POOR"
REVIEW = "REVIEW"

ImageInput = Union[str, Path, np.ndarray]


def _check(status: str, score: float, value: dict[str, Any], explanation: str) -> dict[str, Any]:
    return {"status": status, "score": round(float(np.clip(score, 0, 100)), 1),
            "value": value, "explanation": explanation}


def _grayscale(image: ImageInput) -> np.ndarray:
    """Read an image path or convert an in-memory BGR/gray image."""
    if isinstance(image, (str, Path)):
        source = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"OpenCV could not read image at '{image}'.")
    elif isinstance(image, np.ndarray):
        source = image
    else:
        raise TypeError("image must be a file path or a numpy array")
    if source.size == 0:
        raise ValueError("image must not be empty")
    if source.ndim == 2:
        return source.astype(np.uint8, copy=False)
    if source.ndim == 3 and source.shape[2] in (3, 4):
        conversion = cv2.COLOR_BGRA2GRAY if source.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(source.astype(np.uint8, copy=False), conversion)
    raise ValueError("image must be grayscale, BGR, or BGRA")


def _resolution(height: int, width: int) -> dict[str, Any]:
    pixels = height * width
    value = {"width": width, "height": height, "megapixels": round(pixels / 1_000_000, 2)}
    if min(width, height) < 400 or pixels < 300_000:
        return _check(POOR, 20, value, "Resolution is too low to reliably read a label.")
    if min(width, height) < 800 or pixels < 1_000_000:
        return _check(WARNING, 65, value, "Resolution may be inadequate for small label text.")
    return _check(GOOD, 100, value, "Resolution is suitable for label reading.")


def _blur(gray: np.ndarray) -> dict[str, Any]:
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    value = {"laplacian_variance": round(variance, 1)}
    if variance < 35:
        return _check(POOR, variance * 40 / 35, value, "Image is very blurry and text is unlikely to be reliable.")
    if variance < 80:
        return _check(WARNING, 55 + (variance - 35) * 25 / 45, value, "Image is somewhat blurry; inspect label text manually.")
    return _check(GOOD, min(100, 80 + variance / 20), value, "Image sharpness is suitable for label text.")


def _brightness(gray: np.ndarray) -> dict[str, Any]:
    mean = float(gray.mean())
    value = {"mean_intensity": round(mean, 1)}
    if mean < 50:
        return _check(POOR, mean * 40 / 50, value, "Image is too dark to read reliably.")
    if mean < 75:
        return _check(WARNING, 55 + (mean - 50), value, "Image is dark; some text may be obscured.")
    if mean > 235:
        return _check(POOR, max(0, 40 - (mean - 235) * 5), value, "Image is severely overexposed.")
    if mean > 215:
        return _check(WARNING, 80 - (mean - 215), value, "Image is bright; inspect pale text carefully.")
    return _check(GOOD, 100, value, "Brightness is suitable for label reading.")


def _glare(gray: np.ndarray) -> dict[str, Any]:
    near_white_fraction = float(np.mean(gray >= 250))
    value = {"near_white_fraction": round(near_white_fraction, 4)}
    if near_white_fraction > 0.12:
        return _check(POOR, 40 - near_white_fraction * 100, value, "Glare or overexposure obscures part of the image.")
    if near_white_fraction > 0.03:
        return _check(WARNING, 80 - near_white_fraction * 200, value, "Glare may obscure label details.")
    return _check(GOOD, 100, value, "No significant glare or overexposure detected.")


def _readability(gray: np.ndarray, checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    contrast = float(gray.std())
    visual_score = float(np.mean([checks[name]["score"] for name in ("resolution", "blur", "brightness", "glare")]))
    score = min(100, contrast * 2 + visual_score * 0.45)
    value = {"contrast_stddev": round(contrast, 1)}
    if contrast < 12 or visual_score < 50:
        return _check(POOR, score, value, "The label is unlikely to be reliably readable.")
    if contrast < 20 or visual_score < 75:
        return _check(WARNING, score, value, "Label readability is uncertain and needs human review.")
    return _check(GOOD, score, value, "The image is likely readable for label extraction.")


def assess_image_quality(image: ImageInput) -> dict[str, Any]:
    """Return a structured, non-compliance image-quality assessment.

    Any WARNING or POOR result has a REVIEW recommendation. This module does
    not produce violations or call the OCR pipeline.
    """
    gray = _grayscale(image)
    height, width = gray.shape
    checks = {
        "resolution": _resolution(height, width),
        "blur": _blur(gray),
        "brightness": _brightness(gray),
        "glare": _glare(gray),
    }
    checks["readability"] = _readability(gray, checks)

    statuses = [check["status"] for check in checks.values()]
    quality_status = POOR if POOR in statuses else WARNING if WARNING in statuses else GOOD
    quality_score = round(float(np.mean([check["score"] for check in checks.values()])), 1)
    flagged = [name for name, check in checks.items() if check["status"] != GOOD]

    if quality_status == GOOD:
        explanation = "Image quality is good and the label is likely readable."
        recommendation = "PROCEED"
    else:
        explanation = "Image quality needs review: " + ", ".join(flagged) + "."
        recommendation = REVIEW

    return {"quality_status": quality_status, "quality_score": quality_score,
            "checks": checks, "explanation": explanation,
            "recommendation": recommendation}
