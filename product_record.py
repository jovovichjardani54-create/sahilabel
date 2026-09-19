"""A reusable, source-preserving record for a product photographed from multiple sides."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import os
from typing import Any, Iterable


class CoverageStatus(str, Enum):
    """Whether the required package views have been collected."""

    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class OCRResult:
    """OCR evidence from one image without altering its coordinate system.

    ``bounding_boxes`` deliberately accepts the OCR provider's native coordinate
    objects. The record keeps those objects unchanged, so callers can retain
    pixel coordinates, polygons, or provider-specific metadata.
    """

    text: str
    bounding_boxes: tuple[Any, ...] = ()
    metadata: Any = None

    def __init__(
        self,
        text: str,
        bounding_boxes: Iterable[Any] = (),
        metadata: Any = None,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError("OCR text must be a string.")
        object.__setattr__(self, "text", text)
        try:
            object.__setattr__(self, "bounding_boxes", tuple(bounding_boxes))
        except TypeError as exc:
            raise TypeError("OCR bounding_boxes must be iterable.") from exc
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class ProductImage:
    """One labelled product view and the OCR evidence extracted from it."""

    label: str
    source: Any
    source_identity: str
    ocr_results: tuple[OCRResult, ...] = ()
    metadata: Any = None

    def __init__(
        self,
        label: str,
        source: Any,
        ocr_results: Iterable[OCRResult] = (),
        metadata: Any = None,
    ) -> None:
        normalized_label = _normalize_label(label)
        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_identity", _source_identity(source))
        try:
            results = tuple(ocr_results)
        except TypeError as exc:
            raise TypeError("ocr_results must be iterable.") from exc
        if not all(isinstance(result, OCRResult) for result in results):
            raise TypeError("Each OCR result must be an OCRResult instance.")
        object.__setattr__(self, "ocr_results", results)
        object.__setattr__(self, "metadata", metadata)

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

    product_id: str | None = None
    session_id: str | None = None
    images: list[ProductImage] = field(default_factory=list)

    SUPPORTED_IMAGE_LABELS = frozenset({"front", "back", "side"})
    REQUIRED_PACKAGE_LABELS = SUPPORTED_IMAGE_LABELS

    def __post_init__(self) -> None:
        if not _has_value(self.product_id) and not _has_value(self.session_id):
            raise ValueError("A product_id or session_id is required.")
        if self.product_id is not None and not _has_value(self.product_id):
            raise ValueError("product_id must not be empty.")
        if self.session_id is not None and not _has_value(self.session_id):
            raise ValueError("session_id must not be empty.")
        initial_images = tuple(self.images)
        self.images = []
        for image in initial_images:
            if not isinstance(image, ProductImage):
                raise TypeError("images must contain ProductImage instances.")
            self.add_image(image.label, image.source, image.ocr_results, image.metadata)

    @property
    def image_list(self) -> tuple[ProductImage, ...]:
        """The labelled images in their capture order."""
        return tuple(self.images)

    def add_image(
        self,
        label: str,
        source: Any,
        ocr_results: Iterable[OCRResult] = (),
        metadata: Any = None,
    ) -> ProductImage:
        """Store a new labelled view, rejecting duplicate labels per record."""
        image = ProductImage(label, source, ocr_results, metadata)
        if any(existing.label == image.label for existing in self.images):
            raise ValueError(f"Duplicate image label: {image.label!r}")
        if any(existing.source_identity == image.source_identity for existing in self.images):
            raise ValueError("Duplicate image source.")
        self.images.append(image)
        return image

    def image_for(self, label: str) -> ProductImage | None:
        """Return a labelled image, if this record contains it."""
        normalized_label = _normalize_label(label)
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


def _normalize_label(label: str) -> str:
    if not isinstance(label, str):
        raise TypeError("Image labels must be strings.")
    normalized_label = label.strip().lower()
    if normalized_label not in ProductRecord.SUPPORTED_IMAGE_LABELS:
        supported = ", ".join(sorted(ProductRecord.SUPPORTED_IMAGE_LABELS))
        raise ValueError(f"Image label must be one of: {supported}.")
    return normalized_label


def _source_identity(source: Any) -> str:
    """Create a stable identity without changing the stored source object."""
    if isinstance(source, bytes):
        if not source:
            raise ValueError("Image source must not be empty.")
        payload = b"bytes:\0" + source
    elif isinstance(source, (str, os.PathLike)):
        source_value = os.fspath(source)
        if not isinstance(source_value, str) or not source_value.strip():
            raise ValueError("Image source must not be empty.")
        payload = f"source:\0{source_value}".encode("utf-8")
    else:
        raise TypeError("Image source must be a path, URL, or bytes.")
    return hashlib.sha256(payload).hexdigest()


def _has_value(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())
