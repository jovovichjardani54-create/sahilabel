"""A reusable, source-preserving record for a product photographed from multiple sides."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class CoverageStatus(str, Enum):
    """Whether the required package views have been collected."""

    INCOMPLETE = "INCOMPLETE"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True, slots=True)
class OCRResult:
    """OCR evidence from one image without altering its coordinate system.

    ``bounding_boxes`` deliberately accepts the OCR provider's native coordinate
    objects. The record keeps those objects unchanged, so callers can retain
    pixel coordinates, polygons, or provider-specific metadata.
    """

    text: str
    bounding_boxes: tuple[Any, ...] = ()

    def __init__(self, text: str, bounding_boxes: Iterable[Any] = ()) -> None:
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "bounding_boxes", tuple(bounding_boxes))


@dataclass(frozen=True, slots=True)
class ProductImage:
    """One labelled product view and the OCR evidence extracted from it."""

    label: str
    source: Any
    ocr_results: tuple[OCRResult, ...] = ()

    def __init__(
        self,
        label: str,
        source: Any,
        ocr_results: Iterable[OCRResult] = (),
    ) -> None:
        normalized_label = label.strip().lower()
        if not normalized_label:
            raise ValueError("Image labels must not be empty.")
        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "ocr_results", tuple(ocr_results))

    @property
    def ocr_text(self) -> str:
        """All non-empty OCR fragments for this image, in extraction order."""
        return "\n".join(result.text for result in self.ocr_results if result.text)


@dataclass(slots=True)
class ProductRecord:
    """A product/session's ordered collection of labelled package images.

    Coverage is complete only after all three required package views (front,
    back, and side) are represented. Extra labels are allowed for future views
    such as top, bottom, or an ingredient-panel close-up.
    """

    product_id: str
    session_id: str | None = None
    images: list[ProductImage] = field(default_factory=list)

    REQUIRED_PACKAGE_LABELS = frozenset({"front", "back", "side"})

    def __post_init__(self) -> None:
        if not self.product_id:
            raise ValueError("product_id must not be empty.")
        initial_images = tuple(self.images)
        self.images = []
        for image in initial_images:
            self.add_image(image.label, image.source, image.ocr_results)

    @property
    def image_list(self) -> tuple[ProductImage, ...]:
        """The labelled images in their capture order."""
        return tuple(self.images)

    def add_image(
        self,
        label: str,
        source: Any,
        ocr_results: Iterable[OCRResult] = (),
    ) -> ProductImage:
        """Store a new labelled view, rejecting duplicate labels per record."""
        image = ProductImage(label, source, ocr_results)
        if any(existing.label == image.label for existing in self.images):
            raise ValueError(f"Duplicate image label: {image.label!r}")
        self.images.append(image)
        return image

    def image_for(self, label: str) -> ProductImage | None:
        """Return a labelled image, if this record contains it."""
        normalized_label = label.strip().lower()
        return next((image for image in self.images if image.label == normalized_label), None)

    @property
    def front_image(self) -> ProductImage | None:
        return self.image_for("front")

    @property
    def back_image(self) -> ProductImage | None:
        return self.image_for("back")

    @property
    def side_image(self) -> ProductImage | None:
        return self.image_for("side")

    @property
    def coverage_status(self) -> CoverageStatus:
        labels = {image.label for image in self.images}
        if self.REQUIRED_PACKAGE_LABELS.issubset(labels):
            return CoverageStatus.COMPLETE
        return CoverageStatus.INCOMPLETE

    @property
    def merged_ocr_text(self) -> str:
        """OCR evidence from every image, ordered by image then result."""
        return "\n".join(image.ocr_text for image in self.images if image.ocr_text)
