"""Normalized data schema for hallucination detection.

This module defines Pydantic v2 models for:
- Raw RAGTruth records (mirroring the original dataset format)
- Normalized internal representations
- Validation and statistics structures
- Detector/evaluator contracts

RAGTruth Field Mapping (Original -> Internal):
    response.jsonl:
        id          -> sample_id
        source_id   -> source_id
        model       -> metadata["model"]
        temperature -> metadata["temperature"]
        labels      -> hallucinated_spans + hallucination_label
        split       -> split
        quality     -> metadata["quality"]
        response    -> generated_response

    source_info.jsonl:
        source_id   -> source_id (join key)
        task_type   -> task_type
        source      -> metadata["source"]
        source_info -> reference_context
        prompt      -> question_or_prompt
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Hallucination label taxonomy
# ---------------------------------------------------------------------------

class HallucinationLabel(str, Enum):
    """Claim-level hallucination classification.

    SUPPORTED:             Claim is fully supported by the reference context.
    CONTRADICTED:          Claim directly conflicts with the reference context.
    INSUFFICIENT_EVIDENCE: Claim is not supported but doesn't directly conflict.
    """

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------
# Span-level annotation
# ---------------------------------------------------------------------------

class HallucinationSpan(BaseModel):
    """A single annotated hallucination span within a generated response.

    Attributes:
        start: Start character offset in the response text.
        end: End character offset in the response text.
        text: The hallucinated text substring.
        label_type: RAGTruth label category (e.g. 'Evident Conflict',
                    'Evident Baseless Info', 'Subtle Conflict', 'Subtle Baseless Info').
        due_to_null: Whether the hallucination was caused by null values in source data.
        implicit_true: Whether the span is factually correct but not in the context.
        meta: Annotator comment / notes.
    """

    start: int
    end: int
    text: str
    label_type: str
    due_to_null: bool = False
    implicit_true: bool = False
    meta: str = ""


# ---------------------------------------------------------------------------
# Raw record models (mirror RAGTruth format exactly)
# ---------------------------------------------------------------------------

class RawResponseRecord(BaseModel):
    """Mirrors a single row from RAGTruth response.jsonl exactly.

    Preserves original field names and types for traceability.
    """

    id: str
    source_id: str
    model: str
    temperature: float
    labels: list[dict[str, Any]]
    split: str
    quality: str
    response: str


class RawSourceRecord(BaseModel):
    """Mirrors a single row from RAGTruth source_info.jsonl exactly.

    The source_info field is a string for Summary tasks and a dict for QA/Data2txt.
    """

    source_id: str
    task_type: str
    source: str
    source_info: str | dict[str, Any]
    prompt: str


# ---------------------------------------------------------------------------
# Normalized internal representation
# ---------------------------------------------------------------------------

class NormalizedSample(BaseModel):
    """Internal normalized representation of a hallucination evaluation sample.

    Every processed sample can be traced back to its raw source via sample_id
    (original response id) and source_id.

    The hallucination_label is derived from RAGTruth's span-level labels:
        - No labels                      -> SUPPORTED
        - Any label containing 'Conflict' -> CONTRADICTED
        - Only 'Baseless' labels         -> INSUFFICIENT_EVIDENCE
    """

    sample_id: str = Field(description="Original response ID from RAGTruth")
    source_id: str = Field(description="Source information ID for join/grouping")
    task_type: str = Field(description="QA, Summary, or Data2txt")
    question_or_prompt: str = Field(description="The prompt sent to the LLM")
    reference_context: str = Field(description="Source context (serialized to str)")
    generated_response: str = Field(description="LLM-generated response text")
    hallucination_label: HallucinationLabel
    hallucinated_spans: list[HallucinationSpan] = Field(default_factory=list)
    split: str = Field(description="Official split: 'train' or 'test'")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields: model, temperature, quality, source, etc.",
    )

    def is_hallucinated(self) -> bool:
        """Return True if the sample contains any hallucination."""
        return self.hallucination_label != HallucinationLabel.SUPPORTED


# ---------------------------------------------------------------------------
# Validation structures
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Result of validating a single sample or dataset."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

class DatasetStats(BaseModel):
    """Aggregate statistics over a processed dataset.

    All values MUST be computed from real loaded data—never fabricated.
    """

    total_responses: int = 0
    hallucinated_responses: int = 0
    non_hallucinated_responses: int = 0
    annotated_spans: int = 0
    task_distribution: dict[str, int] = Field(default_factory=dict)
    model_distribution: dict[str, int] = Field(default_factory=dict)
    split_distribution: dict[str, int] = Field(default_factory=dict)
    quality_distribution: dict[str, int] = Field(default_factory=dict)
    avg_response_length: float = 0.0
    invalid_records: int = 0
    label_type_distribution: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detector contract (for future phases)
# ---------------------------------------------------------------------------

class DetectorPrediction(BaseModel):
    """Prediction output from a hallucination detector."""

    sample_id: str
    predicted_label: HallucinationLabel
    confidence: float = 0.0
    predicted_spans: list[HallucinationSpan] = Field(default_factory=list)
    detector_name: str = ""


class BaseDetector(ABC):
    """Abstract base class for hallucination detectors.

    Future detectors (NLI-based, LLM-as-judge, etc.) should subclass this
    to plug into the evaluation framework.

    Example usage (future phase)::

        class NLIDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "nli-deberta-v3"

            def predict(self, sample: NormalizedSample) -> DetectorPrediction:
                # ... run NLI inference ...
                return DetectorPrediction(...)

        detector = NLIDetector()
        predictions = detector.predict_batch(test_samples)
        result = evaluator.evaluate(predictions, test_samples)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name."""
        ...

    @abstractmethod
    def predict(self, sample: NormalizedSample) -> DetectorPrediction:
        """Predict hallucination label for a single sample."""
        ...

    def predict_batch(
        self, samples: list[NormalizedSample]
    ) -> list[DetectorPrediction]:
        """Predict for a batch of samples. Override for efficiency."""
        return [self.predict(s) for s in samples]


# ---------------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Container for evaluation metrics produced by the Evaluator."""

    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion_mat: NDArray[np.int64]
    auroc_score: float | None = None
    auprc_score: float | None = None
    sample_count: int = 0
    class_distribution: dict[str, int] = dc_field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Samples: {self.sample_count}",
            f"Accuracy: {self.accuracy:.4f}",
        ]
        for cls_name, m in self.per_class.items():
            lines.append(
                f"  {cls_name}: P={m.get('precision', 0):.4f} "
                f"R={m.get('recall', 0):.4f} F1={m.get('f1', 0):.4f}"
            )
        if self.auroc_score is not None:
            lines.append(f"AUROC: {self.auroc_score:.4f}")
        if self.auprc_score is not None:
            lines.append(f"AUPRC: {self.auprc_score:.4f}")
        return "\n".join(lines)
