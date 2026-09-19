"""Source-preserving aggregation and compliance evaluation for product images.

This module intentionally combines declaration *meaning*, never image
coordinate systems.  Every declaration candidate retains the OCR words and
bounding box from exactly one ``ProductImage``/``OCRResult`` source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from compliance_checker import check_extracted_fields
from field_extractor import FOUND, NOT_FOUND, UNCERTAIN, extract_fields
from product_record import CoverageStatus, OCRResult, ProductImage, ProductRecord


_FIELD_KEYS = (
    "manufacturer_or_packer",
    "generic_product_name",
    "net_quantity",
    "mrp",
    "manufacture_or_packing_date",
    "consumer_care",
)
_STATUS_RANK = {NOT_FOUND: 0, UNCERTAIN: 1, FOUND: 2}
_LABEL_TIE_BREAK = {"front": 2, "back": 1, "side": 0}
_WORD_KEYS = frozenset({"text", "conf", "left", "top", "width", "height"})


@dataclass(frozen=True, slots=True)
class DeclarationEvidence:
    """One extracted declaration and its unmodified, single-image evidence."""

    field_key: str
    value: str | None
    status: str
    confidence: float
    evidence_text: str
    bounding_box: Mapping[str, Any] | None
    reason: str
    source_label: str
    source_identity: str
    source: Any
    ocr_words: tuple[Mapping[str, Any], ...]
    image_metadata: Any
    ocr_metadata: Any
    context_score: int

    def extracted_field(self) -> dict[str, Any]:
        """Return the existing compliance checker's field-record shape."""
        return {
            "value": self.value,
            "status": self.status,
            "confidence": self.confidence,
            "evidence_text": self.evidence_text,
            "bbox": self.bounding_box,
            "reason": self.reason,
        }

    def provenance(self) -> dict[str, Any]:
        """Return source traceability without copying or combining coordinates."""
        return {
            "source_label": self.source_label,
            "source_identity": self.source_identity,
            "source": self.source,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "ocr_words": self.ocr_words,
            "image_metadata": self.image_metadata,
            "ocr_metadata": self.ocr_metadata,
        }


@dataclass(frozen=True, slots=True)
class AggregatedProductEvidence:
    """Selected declarations plus every competing source-specific candidate.

    The selection policy is deterministic: FOUND beats UNCERTAIN beats
    NOT_FOUND; then higher OCR confidence wins; then a candidate with more
    declaration-specific/context words wins; ties use a fixed label order and
    finally source identity.  Capture order is never a ranking criterion.
    """

    coverage_status: CoverageStatus
    declarations: Mapping[str, DeclarationEvidence]
    candidates: Mapping[str, tuple[DeclarationEvidence, ...]]

    @property
    def package_sides_complete(self) -> bool:
        return self.coverage_status is CoverageStatus.COMPLETE

    def extracted_fields(self) -> dict[str, dict[str, Any]]:
        return {key: evidence.extracted_field() for key, evidence in self.declarations.items()}


@dataclass(frozen=True, slots=True)
class ProductComplianceEvaluation:
    """The aggregation contract Jainil can pass to the compliance layer.

    ``evidence`` contains selected declarations and all alternatives with
    source-local OCR words and coordinates. ``report`` is the existing
    compliance-checker response evaluated with the record's actual coverage.
    """

    evidence: AggregatedProductEvidence
    report: Mapping[str, Any]


def aggregate_product_evidence(product_record: ProductRecord) -> AggregatedProductEvidence:
    """Aggregate per-image declarations while keeping all evidence source-local.

    Args:
        product_record: A ``ProductRecord`` whose OCR results contain either
            word dictionaries in ``bounding_boxes`` (the current upload flow)
            or in ``OCRResult.metadata['words']``.

    Returns:
        ``AggregatedProductEvidence`` with actual coverage, the selected
        evidence per declaration, and every competing candidate.

    Raises:
        TypeError: If the input is not a ``ProductRecord``.
        ValueError: If OCR word dictionaries are malformed.
    """
    if not isinstance(product_record, ProductRecord):
        raise TypeError("product_record must be a ProductRecord instance.")

    candidates = {field_key: [] for field_key in _FIELD_KEYS}
    for image in product_record.image_list:
        for ocr_result in image.ocr_results or (None,):
            payload, words, metadata = _ocr_payload(ocr_result)
            for field_key, extracted in extract_fields(payload).items():
                candidates[field_key].append(
                    _evidence_from(image, words, metadata, field_key, extracted)
                )

    selected = {
        field_key: _select_evidence(field_candidates)
        for field_key, field_candidates in candidates.items()
    }
    return AggregatedProductEvidence(
        coverage_status=product_record.coverage_status,
        declarations=selected,
        candidates={key: tuple(values) for key, values in candidates.items()},
    )


def evaluate_product_record(
    product_record: ProductRecord,
    ocr_quality_sufficient: bool | None = None,
) -> ProductComplianceEvaluation:
    """Evaluate all product-image evidence using the existing compliance rules.

    This is the public integration function. Pass a ``ProductRecord`` and it
    returns selected source-traceable evidence plus the existing compliance
    report. Coverage is derived only from the record's front/back/side images.
    """
    aggregated = aggregate_product_evidence(product_record)
    # Readability observations examine each OCR source once.  Candidates are
    # field-specific views of those sources and would otherwise duplicate words.
    all_words = [
        word
        for image in product_record.image_list
        for ocr_result in image.ocr_results
        for word in _ocr_payload(ocr_result)[1]
    ]
    quality = (
        _all_sources_have_sufficient_quality(product_record)
        if ocr_quality_sufficient is None
        else ocr_quality_sufficient
    )
    report = check_extracted_fields(
        aggregated.extracted_fields(),
        package_sides_complete=aggregated.package_sides_complete,
        ocr_quality_sufficient=quality,
        source_evidence={key: value.provenance() for key, value in aggregated.declarations.items()},
        readability_words=all_words,
    )
    return ProductComplianceEvaluation(aggregated, report)


def _ocr_payload(ocr_result: OCRResult | None) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...], Any]:
    if ocr_result is None:
        return {"full_text": "", "words": []}, (), None
    if not isinstance(ocr_result, OCRResult):
        raise TypeError("Product image OCR results must be OCRResult instances.")
    metadata_words = (
        ocr_result.metadata.get("words")
        if isinstance(ocr_result.metadata, Mapping) else None
    )
    raw_words = metadata_words if metadata_words is not None else ocr_result.bounding_boxes
    words = _validate_words(raw_words)
    return {"full_text": ocr_result.text, "words": list(words)}, words, ocr_result.metadata


def _validate_words(raw_words: Any) -> tuple[Mapping[str, Any], ...]:
    if not raw_words:
        return ()
    if not all(isinstance(word, Mapping) for word in raw_words):
        # OCRResult also supports provider-native coordinate tuples. They cannot
        # be parsed by this extractor, but remain stored on the source image.
        return ()
    words = tuple(raw_words)
    for word in words:
        missing = _WORD_KEYS.difference(word)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"OCR word is missing required fields: {names}.")
    return words


def _evidence_from(
    image: ProductImage,
    words: tuple[Mapping[str, Any], ...],
    metadata: Any,
    field_key: str,
    extracted: Mapping[str, Any],
) -> DeclarationEvidence:
    evidence_text = extracted["evidence_text"]
    return DeclarationEvidence(
        field_key=field_key,
        value=extracted["value"],
        status=extracted["status"],
        confidence=float(extracted["confidence"]),
        evidence_text=evidence_text,
        bounding_box=extracted["bbox"],
        reason=extracted["reason"],
        source_label=image.label,
        source_identity=image.source_identity,
        source=image.source,
        ocr_words=words,
        image_metadata=image.metadata,
        ocr_metadata=metadata,
        context_score=len(evidence_text.split()),
    )


def _select_evidence(candidates: list[DeclarationEvidence]) -> DeclarationEvidence:
    if not candidates:
        raise ValueError("A ProductRecord must provide at least one image.")
    return max(
        candidates,
        key=lambda candidate: (
            _STATUS_RANK[candidate.status],
            candidate.confidence,
            candidate.context_score,
            _LABEL_TIE_BREAK[candidate.source_label],
            candidate.source_identity,
        ),
    )


def _all_sources_have_sufficient_quality(product_record: ProductRecord) -> bool:
    source_qualities = []
    for image in product_record.image_list:
        for result in image.ocr_results or (None,):
            _, words, _ = _ocr_payload(result)
            confidences = [float(word["conf"]) for word in words if float(word["conf"]) >= 0]
            source_qualities.append(bool(confidences) and sum(confidences) / len(confidences) >= 60.0)
    return bool(source_qualities) and all(source_qualities)
