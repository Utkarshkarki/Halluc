"""Preprocessing pipeline for RAGTruth data.

Handles:
- Merging response records with source-info records
- Normalising hallucination labels and spans
- Validating individual samples and the full dataset
- Computing aggregate dataset statistics
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from src.data.schema import (
    DatasetStats,
    HallucinationLabel,
    HallucinationSpan,
    NormalizedSample,
    RawResponseRecord,
    RawSourceRecord,
    ValidationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def _classify_label(labels: list[dict[str, Any]]) -> HallucinationLabel:
    """Derive a response-level hallucination label from span-level annotations.

    Rules (following RAGTruth taxonomy):
        - No labels at all                            -> SUPPORTED
        - Any label_type containing 'Conflict'        -> CONTRADICTED
        - Only labels containing 'Baseless' (no Conflict) -> INSUFFICIENT_EVIDENCE

    If a label_type doesn't match either keyword, it defaults to
    INSUFFICIENT_EVIDENCE (conservative).
    """
    if not labels:
        return HallucinationLabel.SUPPORTED

    has_conflict = any(
        "conflict" in (lbl.get("label_type", "") or "").lower()
        for lbl in labels
    )
    if has_conflict:
        return HallucinationLabel.CONTRADICTED

    return HallucinationLabel.INSUFFICIENT_EVIDENCE


def _parse_spans(labels: list[dict[str, Any]]) -> list[HallucinationSpan]:
    """Convert raw label dicts into HallucinationSpan objects."""
    spans: list[HallucinationSpan] = []
    for lbl in labels:
        try:
            span = HallucinationSpan(
                start=int(lbl.get("start", 0)),
                end=int(lbl.get("end", 0)),
                text=str(lbl.get("text", "")),
                label_type=str(lbl.get("label_type", "unknown")),
                due_to_null=bool(lbl.get("due_to_null", False)),
                implicit_true=bool(lbl.get("implicit_true", False)),
                meta=str(lbl.get("meta", "")),
            )
            spans.append(span)
        except Exception as e:
            logger.warning("Could not parse span label %r: %s", lbl, e)
    return spans


def _serialize_source_info(source_info: str | dict[str, Any]) -> str:
    """Convert source_info to a string representation."""
    if isinstance(source_info, str):
        return source_info
    return json.dumps(source_info, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_responses_and_sources(
    responses: list[RawResponseRecord],
    sources: list[RawSourceRecord],
) -> list[NormalizedSample]:
    """Join response records with source-info records on source_id.

    Produces a list of NormalizedSample ready for downstream processing.
    Responses whose source_id has no matching source record are still included
    with best-effort defaults and a logged warning.

    Args:
        responses: Parsed response records from RAGTruth.
        sources: Parsed source-info records from RAGTruth.

    Returns:
        List of normalised samples.
    """
    source_map: dict[str, RawSourceRecord] = {s.source_id: s for s in sources}
    samples: list[NormalizedSample] = []
    missing_sources = 0

    for resp in responses:
        src = source_map.get(resp.source_id)

        if src is not None:
            task_type = src.task_type
            question_or_prompt = src.prompt
            reference_context = _serialize_source_info(src.source_info)
            source_name = src.source
        else:
            missing_sources += 1
            task_type = "unknown"
            question_or_prompt = ""
            reference_context = ""
            source_name = "unknown"

        hallucination_label = _classify_label(resp.labels)
        hallucinated_spans = _parse_spans(resp.labels)

        # Collect original label types for metadata
        original_label_types = [
            lbl.get("label_type", "unknown") for lbl in resp.labels
        ]

        sample = NormalizedSample(
            sample_id=resp.id,
            source_id=resp.source_id,
            task_type=task_type,
            question_or_prompt=question_or_prompt,
            reference_context=reference_context,
            generated_response=resp.response,
            hallucination_label=hallucination_label,
            hallucinated_spans=hallucinated_spans,
            split=resp.split,
            metadata={
                "model": resp.model,
                "temperature": resp.temperature,
                "quality": resp.quality,
                "source": source_name,
                "task_type": task_type,
                "original_label_types": original_label_types,
            },
        )
        samples.append(sample)

    if missing_sources > 0:
        logger.warning(
            "%d responses had no matching source_id in source_info",
            missing_sources,
        )
    logger.info("Merged %d responses into normalised samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_sample(sample: NormalizedSample) -> ValidationResult:
    """Validate a single normalised sample.

    Checks:
    - Non-empty sample_id
    - Non-empty generated_response
    - Non-empty reference_context
    - Span offset correctness (text at [start:end] matches span.text)
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not sample.sample_id:
        errors.append("Missing sample_id")

    if not sample.generated_response:
        errors.append("Empty generated_response")

    if not sample.reference_context:
        warnings.append("Empty reference_context (source info may be missing)")

    if not sample.question_or_prompt:
        warnings.append("Empty question_or_prompt")

    # Validate span offsets
    for i, span in enumerate(sample.hallucinated_spans):
        if span.start < 0 or span.end < 0:
            errors.append(f"Span {i}: negative offset (start={span.start}, end={span.end})")
            continue

        if span.start >= span.end:
            errors.append(f"Span {i}: start ({span.start}) >= end ({span.end})")
            continue

        if span.end > len(sample.generated_response):
            errors.append(
                f"Span {i}: end offset {span.end} exceeds response length "
                f"{len(sample.generated_response)}"
            )
            continue

        actual_text = sample.generated_response[span.start : span.end]
        if actual_text != span.text:
            errors.append(
                f"Span {i}: text mismatch at [{span.start}:{span.end}]. "
                f"Expected {span.text!r}, got {actual_text!r}"
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_dataset(
    samples: list[NormalizedSample],
) -> tuple[list[NormalizedSample], list[tuple[NormalizedSample, ValidationResult]]]:
    """Validate all samples and partition into valid / invalid.

    Also checks for:
    - Duplicate sample_ids
    - Empty dataset

    Returns:
        (valid_samples, list_of_(invalid_sample, validation_result))
    """
    if not samples:
        logger.warning("validate_dataset called with empty sample list")
        return [], []

    valid: list[NormalizedSample] = []
    invalid: list[tuple[NormalizedSample, ValidationResult]] = []

    # Duplicate ID detection
    id_counts: Counter[str] = Counter(s.sample_id for s in samples)
    duplicates = {sid for sid, count in id_counts.items() if count > 1}
    if duplicates:
        logger.warning(
            "%d duplicate sample_ids found: %s",
            len(duplicates),
            list(duplicates)[:10],
        )

    for sample in samples:
        result = validate_sample(sample)

        # Add duplicate warning
        if sample.sample_id in duplicates:
            result.warnings.append(
                f"Duplicate sample_id: {sample.sample_id}"
            )

        if result.is_valid:
            valid.append(sample)
        else:
            invalid.append((sample, result))

    logger.info(
        "Validation complete: %d valid, %d invalid out of %d total",
        len(valid),
        len(invalid),
        len(samples),
    )
    return valid, invalid


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(
    samples: list[NormalizedSample],
    invalid_count: int = 0,
) -> DatasetStats:
    """Compute aggregate statistics over normalised samples.

    All numbers are derived from the actual data—nothing is fabricated.

    Args:
        samples: List of normalised samples.
        invalid_count: Number of records that failed validation.

    Returns:
        DatasetStats with all fields populated from real data.
    """
    if not samples:
        return DatasetStats(invalid_records=invalid_count)

    hallucinated = sum(1 for s in samples if s.is_hallucinated())
    all_spans: list[HallucinationSpan] = []
    label_type_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    quality_counter: Counter[str] = Counter()
    total_response_len = 0

    for s in samples:
        all_spans.extend(s.hallucinated_spans)
        for span in s.hallucinated_spans:
            label_type_counter[span.label_type] += 1
        task_counter[s.task_type] += 1
        model_counter[s.metadata.get("model", "unknown")] += 1
        split_counter[s.split] += 1
        quality_counter[s.metadata.get("quality", "unknown")] += 1
        total_response_len += len(s.generated_response)

    return DatasetStats(
        total_responses=len(samples),
        hallucinated_responses=hallucinated,
        non_hallucinated_responses=len(samples) - hallucinated,
        annotated_spans=len(all_spans),
        task_distribution=dict(task_counter),
        model_distribution=dict(model_counter),
        split_distribution=dict(split_counter),
        quality_distribution=dict(quality_counter),
        avg_response_length=total_response_len / len(samples) if samples else 0.0,
        invalid_records=invalid_count,
        label_type_distribution=dict(label_type_counter),
    )
