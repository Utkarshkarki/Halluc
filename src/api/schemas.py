"""Pydantic v2 schemas for FastAPI productization (Phase 6).

Defines clean request and response contracts for:
- /api/v1/health
- /api/v1/ready
- /api/v1/models
- /api/v1/analyze
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_RESPONSE_LENGTH = 100000
MAX_QUESTION_LENGTH = 10000
MAX_CONTEXT_LENGTH = 200000
MAX_CONTEXT_ITEMS = 100


# ---------------------------------------------------------------------------
# Error Response Schema
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Structured error schema returned on client or server failure."""

    error: str = Field(description="Error category or type")
    detail: Any = Field(description="Error details or validation message")
    request_id: str | None = Field(default=None, description="Tracking request identifier")
    status_code: int = Field(description="HTTP status code")


# ---------------------------------------------------------------------------
# Health & Status Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: Literal["ok", "degraded", "error"] = "ok"
    timestamp: str = Field(description="ISO 8601 UTC timestamp")
    version: str = Field(default="0.1.0", description="API version")


class ReadyResponse(BaseModel):
    """Readiness probe response checking subsystem availability."""

    ready: bool = Field(description="True if all critical inference subsystems are loaded")
    components: dict[str, bool] = Field(
        description="Individual subsystem health: nli, embeddings, faiss, claim_extractor"
    )
    device: str = Field(description="Active execution device (cuda or cpu)")


class ModelsMetadataResponse(BaseModel):
    """Metadata regarding active models and threshold configuration."""

    schema_version: str = "v1.0.0"
    nli_model: str = Field(description="HuggingFace model identifier for NLI sequence classifier")
    embedding_model: str = Field(description="SentenceTransformer embedding model name")
    embedding_dimension: int = Field(description="Vector embedding dimension")
    claim_extractor_provider: str = Field(description="Active claim decomposition provider name")
    retrieval_top_k: int = Field(description="Default top-K retrieved evidence chunks per claim")
    support_threshold: float = Field(description="Minimum entailment probability to assign SUPPORTED")
    contradiction_threshold: float = Field(description="Minimum contradiction probability to assign CONTRADICTED")
    risk_thresholds: dict[str, float] = Field(
        description="Threshold cutoffs for categorising hallucination risk level"
    )
    device: str = Field(description="Compute device")


# ---------------------------------------------------------------------------
# Analysis Request & Response Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request payload for claim-level hallucination detection."""

    response: str = Field(
        ...,
        min_length=1,
        max_length=MAX_RESPONSE_LENGTH,
        description="The LLM-generated response text to decompose and verify for hallucinations",
    )
    reference_context: str | list[str] = Field(
        ...,
        description="Trusted reference corpus or passages against which the response is evaluated",
    )
    question: str | None = Field(
        default=None,
        max_length=MAX_QUESTION_LENGTH,
        description="Optional prompt or question that elicited the response",
    )
    source_id: str | None = Field(
        default=None,
        description="Optional source context identifier for scoping retrieval and audit provenance",
    )
    sample_id: str | None = Field(
        default=None,
        description="Optional client-provided tracking identifier",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for number of evidence passages to retrieve per claim",
    )

    @field_validator("response", mode="before")
    @classmethod
    def _validate_response_non_empty(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("Response text cannot be empty or whitespace only.")
        if len(s) > MAX_RESPONSE_LENGTH:
            raise ValueError(f"Response text exceeds maximum length of {MAX_RESPONSE_LENGTH} characters.")
        return s

    @field_validator("reference_context", mode="before")
    @classmethod
    def _validate_reference_context_non_empty(cls, v: Any) -> str | list[str]:
        if isinstance(v, list):
            if len(v) > MAX_CONTEXT_ITEMS:
                raise ValueError(
                    f"Reference context list exceeds maximum allowed item count of {MAX_CONTEXT_ITEMS}."
                )
            filtered = [str(item).strip() for item in v if str(item).strip()]
            if not filtered:
                raise ValueError("Reference context list cannot be empty or contain only empty items.")
            total_len = sum(len(item) for item in filtered)
            if total_len > MAX_CONTEXT_LENGTH:
                raise ValueError(
                    f"Reference context total length ({total_len}) exceeds maximum allowed of {MAX_CONTEXT_LENGTH} characters."
                )
            return filtered
        s = str(v).strip()
        if not s:
            raise ValueError("Reference context cannot be empty or whitespace only.")
        if len(s) > MAX_CONTEXT_LENGTH:
            raise ValueError(
                f"Reference context length ({len(s)}) exceeds maximum allowed of {MAX_CONTEXT_LENGTH} characters."
            )
        return s


class EvidencePassageOut(BaseModel):
    """Ranked evidence passage considered during verification of a claim."""

    chunk_id: str
    source_id: str
    text: str
    similarity_score: float = Field(description="Cosine similarity between claim and evidence passage")
    rank: int = Field(ge=1, description="1-based retrieval ranking")
    nli_verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    nli_confidence: float = Field(ge=0.0, le=1.0)
    nli_probabilities: dict[str, float] = Field(
        description="Complete 3-way distribution: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE"
    )


class VerifiedClaimOut(BaseModel):
    """An atomic factual claim with evidence and aggregated verdict."""

    claim_id: str
    claim_text: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidencePassageOut] = Field(default_factory=list)


class ExcludedClaimOut(BaseModel):
    """Non-factual statement excluded from verification (opinions, instructions, questions)."""

    claim_id: str
    text: str
    type: str = Field(description="Category: OPINION, INSTRUCTION, QUESTION, NON_VERIFIABLE")


class AnalyzeResponse(BaseModel):
    """Complete claim-level hallucination detection result."""

    schema_version: str = "v1.0.0"
    sample_id: str
    hallucination_risk: float = Field(ge=0.0, le=1.0, description="Contradicted claims / total factual claims")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    support_rate: float = Field(ge=0.0, le=1.0, description="Supported claims / total factual claims")
    evidence_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description="Claims with decisive evidence (supported + contradicted) / total factual claims",
    )
    abstention_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Insufficient evidence claims / total factual claims",
    )
    total_factual_claims: int
    total_claims: int
    verified_claims: list[VerifiedClaimOut] = Field(default_factory=list)
    excluded_claims: list[ExcludedClaimOut] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
