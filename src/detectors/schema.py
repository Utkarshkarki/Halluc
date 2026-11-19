"""Schemas for detector inputs, outputs, and configuration.

Defines the output contract for NLI and future detectors.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from src.data.schema import HallucinationLabel


class NLIProbabilities(BaseModel):
    """Normalized probability distribution over the 3-way hallucination taxonomy.

    Probabilities must sum to 1.0 within numerical precision.
    """

    supported: float = Field(ge=0.0, le=1.0, description="P(Entailment -> Supported)")
    insufficient_evidence: float = Field(
        ge=0.0, le=1.0, description="P(Neutral -> Insufficient Evidence)"
    )
    contradicted: float = Field(
        ge=0.0, le=1.0, description="P(Contradiction -> Contradicted)"
    )


class NLIOutput(BaseModel):
    """Standardized output structure for an evaluated premise-hypothesis pair.

    Adheres to the project output contract:
    - label: Predicted HallucinationLabel (SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE)
    - confidence: Confidence score for the winning label in [0.0, 1.0]
    - probabilities: Complete 3-way probability distribution
    - model: Identifier of the detector model used
    - truncated: Boolean flag indicating if premise/hypothesis exceeded context limits
    """

    label: HallucinationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: NLIProbabilities
    model: str
    truncated: bool = False
    premise_tokens: int | None = None
    hypothesis_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_contract_dict(self) -> dict[str, Any]:
        """Serialize exactly as specified in the Phase 2 output contract."""
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "probabilities": {
                "supported": round(self.probabilities.supported, 4),
                "insufficient_evidence": round(self.probabilities.insufficient_evidence, 4),
                "contradicted": round(self.probabilities.contradicted, 4),
            },
            "model": self.model,
            "truncated": self.truncated,
        }


class NLIConfig(BaseModel):
    """Configuration for NLI detector loading and inference."""

    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    revision: str | None = None
    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"
    fp16: bool = True
    long_context_strategy: str = Field(
        default="chunk_evidence",
        description="Strategy for long contexts: 'chunk_evidence' or 'truncate'",
    )
    seed: int = 42
