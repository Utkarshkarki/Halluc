from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal

from pydantic import BaseModel, Field

# Verdict for each claim after aggregation
class ClaimVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

# Per‑evidence passage information with NLI scores
class EvidencePassage(BaseModel):
    chunk_id: str = Field(description="Identifier of the evidence chunk")
    source_id: str = Field(description="Source document identifier")
    text: str = Field(description="Evidence passage text")
    similarity_score: float = Field(description="Cosine similarity of claim‑evidence embedding")
    rank: int = Field(ge=1, description="1‑based rank in top‑K list")
    nli_probabilities: dict[str, float] = Field(
        description="Probabilities for supported, insufficient_evidence, contradicted"
    )
    nli_label: ClaimVerdict = Field(description="NLI verdict for this passage")
    nli_confidence: float = Field(description="Confidence of the NLI label")

# Result for a single claim after aggregation
class VerifiedClaim(BaseModel):
    claim_id: str = Field(description="Identifier of the claim within the response")
    claim_text: str = Field(description="Atomic factual claim text")
    verdict: ClaimVerdict = Field(description="Final verdict after evidence aggregation")
    confidence: float = Field(description="Aggregated confidence score for the verdict")
    evidence: List[EvidencePassage] = Field(default_factory=list, description="Evidence passages considered")

# Overall pipeline result for a sample
class PipelineResult(BaseModel):
    schema_version: str = "v1.0.0"
    sample_id: str = Field(description="Dataset sample identifier")
    hallucination_risk: float = Field(description="Risk score in [0,1]")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Categorised risk level")
    support_rate: float = Field(default=0.0, description="Supported claims / total factual claims")
    evidence_coverage: float = Field(description="Claims with decisive evidence (supported + contradicted) / total factual claims")
    abstention_rate: float = Field(description="Insufficient evidence claims / total factual claims")
    total_factual_claims: int = Field(description="Number of factual claims extracted")
    total_claims: int = Field(description="Total claims (including non‑factual)")
    verified_claims: List[VerifiedClaim] = Field(default_factory=list)
    excluded_claims: List[dict[str, Any]] = Field(default_factory=list, description="Non‑factual claims for audit")
    latency_ms: dict[str, float] = Field(default_factory=dict, description="Component latency breakdown")
    metadata: dict[str, Any] = Field(default_factory=dict)
