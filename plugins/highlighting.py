"""Rule-specific visual evidence overlays for structured field decisions."""

import os
from PIL import Image, ImageDraw, ImageFont

ANNOTATED_DIR = "annotated"
os.makedirs(ANNOTATED_DIR, exist_ok=True)
STATUS_COLORS = {"PASS": (15, 110, 86), "REVIEW": (180, 83, 9), "VIOLATION": (153, 60, 27)}
DECLARATION_NAMES = {
    "manufacturer_or_packer": "Manufacturer / packer", "generic_product_name": "Product name",
    "net_quantity": "Net quantity", "mrp": "MRP", "manufacture_or_packing_date": "Manufacture / packing date",
    "consumer_care": "Consumer care",
}


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def _valid_bbox(bbox: dict | None, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Return a clipped source-coordinate box, or None when no evidence exists."""
    if not isinstance(bbox, dict):
        return None
    try:
        left, top, width, height = (int(bbox[key]) for key in ("left", "top", "width", "height"))
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    image_width, image_height = image_size
    right, bottom = left + width, top + height
    if right <= 0 or bottom <= 0 or left >= image_width or top >= image_height:
        return None
    return max(0, left), max(0, top), min(image_width - 1, right), min(image_height - 1, bottom)


def _label_text(decision: dict) -> str:
    name = DECLARATION_NAMES.get(decision.get("field_key"), decision.get("field_key", "Declaration"))
    value = decision.get("extracted_value")
    return f"{name} - {decision.get('status', 'REVIEW')}" + (f" - {value}" if value else "")


def _fit_label(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    while text and draw.textbbox((0, 0), text + suffix, font=font)[2] > max_width:
        text = text[:-1]
    return (text + suffix) if text else suffix


def _label_position(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, label: str,
                    box: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int, str]:
    """Keep labels inside the source image, above a box when possible."""
    image_width, image_height = image_size
    label = _fit_label(draw, label, font, max(1, image_width - 8))
    text_box = draw.textbbox((0, 0), label, font=font)
    label_width, label_height = text_box[2] - text_box[0] + 6, text_box[3] - text_box[1] + 6
    left, top, _, bottom = box
    x = min(max(0, left), max(0, image_width - label_width))
    y = top - label_height - 2 if top >= label_height + 2 else bottom + 2
    y = min(max(0, y), max(0, image_height - label_height))
    return x, y, label_width, label_height, label


def _overlaps(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]


def _avoid_label_collisions(position: tuple[int, int, int, int, str], occupied: list[tuple[int, int, int, int]],
                            image_size: tuple[int, int]) -> tuple[int, int, int, int, str]:
    """Move a label vertically when a nearby field would otherwise obscure it."""
    x, y, width, height, label = position
    image_width, image_height = image_size
    candidates = [y]
    for offset in range(height + 2, image_height, height + 2):
        candidates.extend((y + offset, y - offset))
    for candidate_y in candidates:
        candidate_y = min(max(0, candidate_y), max(0, image_height - height))
        candidate = (x, candidate_y, x + width, candidate_y + height)
        if not any(_overlaps(candidate, prior) for prior in occupied):
            return x, candidate_y, width, height, label
    return x, y, width, height, label


def _draw_decisions(image: Image.Image, decisions: list[dict]) -> list[dict]:
    """Draw only rule-relevant field evidence and return metadata for tests."""
    draw, font, overlays, occupied_labels = ImageDraw.Draw(image), _font(), [], []
    for decision in decisions:
        box = _valid_bbox(decision.get("bbox"), image.size)
        if box is None:
            continue
        status = decision.get("status", "REVIEW")
        color = STATUS_COLORS.get(status, STATUS_COLORS["REVIEW"])
        position = _label_position(draw, font, _label_text(decision), box, image.size)
        x, y, width, height, label = _avoid_label_collisions(position, occupied_labels, image.size)
        draw.rectangle(box, outline=color, width=3)
        draw.rectangle((x, y, x + width, y + height), fill=color)
        draw.text((x + 3, y + 3), label, fill=(255, 255, 255), font=font)
        overlays.append({"field_key": decision.get("field_key"), "status": status, "box": box,
                         "label_box": (x, y, x + width, y + height), "color": color})
        occupied_labels.append((x, y, x + width, y + height))
    return overlays


def _legacy_decisions(report: dict, ocr_words: list[dict]) -> list[dict]:
    """Adapt historical reports that predate structured field decisions."""
    decisions = []
    for field_key, field in report.get("fields", {}).items():
        match = field.get("matched_text")
        matched_words = [word for word in ocr_words if match and word["text"].casefold() in str(match).casefold()]
        bbox = None
        if matched_words:
            left, top = min(w["left"] for w in matched_words), min(w["top"] for w in matched_words)
            right, bottom = max(w["left"] + w["width"] for w in matched_words), max(w["top"] + w["height"] for w in matched_words)
            bbox = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        decisions.append({"field_key": field_key, "status": "PASS" if field.get("present") else "VIOLATION",
                          "extracted_value": match, "bbox": bbox})
    return decisions


def _draw_legend(image: Image.Image) -> None:
    draw, font = ImageDraw.Draw(image), _font()
    x, y = 6, 6
    for status in ("PASS", "REVIEW", "VIOLATION"):
        draw.rectangle((x, y, x + 12, y + 12), fill=STATUS_COLORS[status])
        draw.text((x + 17, y - 1), status, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0), font=font)
        y += 17


def annotate_image(image_path: str, ocr_words: list, report: dict, item_id: str) -> str:
    """Render decision evidence on the unchanged-resolution source image."""
    image = Image.open(image_path).convert("RGB")
    decisions = list(report.get("field_decisions", {}).values()) or _legacy_decisions(report, ocr_words)
    _draw_decisions(image, decisions)
    _draw_legend(image)
    output_path = os.path.join(ANNOTATED_DIR, f"{item_id}_annotated.png")
    image.save(output_path)
    return output_path
