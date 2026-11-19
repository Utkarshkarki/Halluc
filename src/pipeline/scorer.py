# src/pipeline/scorer.py
"""Response scoring utilities for Phase 5.

The :class:`ResponseScorer` aggregates the per‑claim ``VerifiedClaim``
objects produced by the detector and computes a final hallucination risk
score together with auxiliary metrics such as evidence coverage and
abstention rate.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

from .schema import ClaimVerdict, VerifiedClaim


class ResponseScore(BaseModel):
    """Aggregated metrics for a single LLM response.

    Attributes
    ----------
    total_factual_claims: int
        Number of extracted factual claims.
    supported: int
        Number of factual claims with verdict ``SUPPORTED``.
    contradicted: int
        Number of factual claims with verdict ``CONTRADICTED``.
    insufficient: int
        Number of factual claims with verdict ``INSUFFICIENT_EVIDENCE``.
    evidence_coverage: float
        Ratio of decisive claims (supported + contradicted) over total factual claims.
    abstention_rate: float
        Ratio of insufficient‑evidence claims over total factual claims.
    hallucination_risk: float
        Ratio of contradicted claims over total factual claims.
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
        Categorical risk derived from ``hallucination_risk``.
    """

    total_factual_claims: int = Field(..., description="Number of factual claims extracted")
    supported: int = Field(..., description="Supported factual claims")
    contradicted: int = Field(..., description="Contradicted factual claims")
    insufficient: int = Field(..., description="Claims with insufficient evidence")
    support_rate: float = Field(default=0.0, description="Supported / total factual claims")
    evidence_coverage: float = Field(..., description="Decisive claims (supported + contradicted) / total factual claims")
    abstention_rate: float = Field(..., description="Insufficient / total factual claims")
    hallucination_risk: float = Field(..., description="Contradicted / total factual claims")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Categorised risk level")

    @field_validator("support_rate", "evidence_coverage", "abstention_rate", "hallucination_risk", mode="before")
    @classmethod
    def _bound_zero_one(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class ResponseScorer:
    """Utility that turns a list of :class:`VerifiedClaim` objects into a
    :class:`ResponseScore`.
    """

    def __init__(self, low_threshold: float = 0.20, medium_threshold: float = 0.50):
        if not (0.0 <= low_threshold < medium_threshold <= 1.0):
            raise ValueError("Invalid risk thresholds: must satisfy 0 <= low < medium <= 1")
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold

    def _categorise_risk(self, risk: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
        if risk <= self.low_threshold:
            return "LOW"
        if risk <= self.medium_threshold:
            return "MEDIUM"
        return "HIGH"

    def score(self, claims: List[VerifiedClaim]) -> ResponseScore:
        total = len(claims)
        if total == 0:
            return ResponseScore(
                total_factual_claims=0,
                supported=0,
                contradicted=0,
                insufficient=0,
                support_rate=0.0,
                evidence_coverage=0.0,
                abstention_rate=0.0,
                hallucination_risk=0.0,
                risk_level="LOW",
            )
        supported = sum(1 for c in claims if c.verdict == ClaimVerdict.SUPPORTED)
        contradicted = sum(1 for c in claims if c.verdict == ClaimVerdict.CONTRADICTED)
        insufficient = sum(1 for c in claims if c.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE)
        support_rate = supported / total
        evidence_coverage = (supported + contradicted) / total
        abstention_rate = insufficient / total
        hallucination_risk = contradicted / total
        risk_level = self._categorise_risk(hallucination_risk)
        return ResponseScore(
            total_factual_claims=total,
            supported=supported,
            contradicted=contradicted,
            insufficient=insufficient,
            support_rate=support_rate,
            evidence_coverage=evidence_coverage,
            abstention_rate=abstention_rate,
            hallucination_risk=hallucination_risk,
            risk_level=risk_level,
        )

__all__ = ["ResponseScorer", "ResponseScore"]
