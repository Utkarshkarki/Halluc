"""Schemas for atomic claim extraction and classification.

Defines Pydantic v2 models for:
- Claim categories (FACTUAL, OPINION, INSTRUCTION, QUESTION, NON_VERIFIABLE)
- Individual atomic claims with span offsets and provenance
- Extraction responses with auditable excluded statements and execution metadata
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Categorization of decomposed statements.

    Only FACTUAL claims should proceed to factual verification.
    Excluded non-factual categories remain recorded for full auditability.
    """

    FACTUAL = "FACTUAL"
    OPINION = "OPINION"
    INSTRUCTION = "INSTRUCTION"
    QUESTION = "QUESTION"
    NON_VERIFIABLE = "NON_VERIFIABLE"


class ExtractedClaim(BaseModel):
    """An atomic, independently verifiable proposition.

    Attributes:
        claim_id: Unique identifier for the claim within the response (e.g. 'C1', 'C2').
        text: Self-contained proposition without unresolved pronouns.
        claim_type: Classification tag (FACTUAL, OPINION, etc.).
        source_start: Starting character offset in the source response text (None if reconstructed).
        source_end: Ending character offset in the source response text (None if reconstructed).
        context_tags: Optional tags (entities, dates, quantities) preserved from original statement.
        confidence: Confidence score of the extraction/classification in [0.0, 1.0].
    """

    claim_id: str = Field(description="Unique identifier within response, e.g. 'C1'")
    text: str = Field(description="Self-contained atomic proposition")
    claim_type: ClaimType = Field(
        default=ClaimType.FACTUAL,
        description="Category: FACTUAL, OPINION, INSTRUCTION, QUESTION, NON_VERIFIABLE",
    )
    source_start: int | None = Field(
        default=None,
        description="Start character offset in source text. None if reconstructed.",
    )
    source_end: int | None = Field(
        default=None,
        description="End character offset in source text. None if reconstructed.",
    )
    context_tags: list[str] = Field(
        default_factory=list,
        description="Preserved attributes: dates, quantities, named entities",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def is_verifiable(self) -> bool:
        """Return True only if claim is factual and eligible for verification."""
        return self.claim_type == ClaimType.FACTUAL


class ClaimExtractionPayload(BaseModel):
    """Schema returned by structured LLM provider outputs."""

    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description="List of extracted claims with classifications and offsets",
    )


class ClaimExtractionResult(BaseModel):
    """Complete provenance-tracked result of extracting claims from a response."""

    claims: list[ExtractedClaim] = Field(default_factory=list)
    raw_response: str = Field(description="Original response text decomposed")
    sample_id: str | None = Field(default=None, description="Dataset sample ID if available")
    extractor_model: str = Field(description="LLM provider and model used for extraction")
    prompt_version: str = Field(description="Version of the extraction prompt")
    latency_ms: float = Field(default=0.0, description="Extraction duration in milliseconds")
    cached: bool = Field(default=False, description="Whether the result was loaded from cache")
    total_claims: int = 0
    factual_claims_count: int = 0
    excluded_claims_count: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_factual_claims(self) -> list[ExtractedClaim]:
        """Return only factual claims eligible for verification."""
        return [c for c in self.claims if c.is_verifiable()]

    def get_excluded_claims(self) -> list[ExtractedClaim]:
        """Return non-factual claims (opinions, instructions, questions, etc.)."""
        return [c for c in self.claims if not c.is_verifiable()]
