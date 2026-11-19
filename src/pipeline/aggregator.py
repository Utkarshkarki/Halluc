# src/pipeline/aggregator.py
"""Evidence aggregation module.
Aggregates NLI results over top‑K evidence passages for a single claim
into a final verdict and confidence score.
"""

from __future__ import annotations

from typing import List

from .schema import ClaimVerdict, EvidencePassage


class EvidenceAggregator:
    """Aggregate evidence passages for a claim.

    The aggregation follows a *contradiction‑first* policy:
    * If any passage has label ``CONTRADICTED`` with probability
      >= ``contradiction_threshold`` → verdict ``CONTRADICTED``.
    * Else if any passage has label ``SUPPORTED`` with probability
      >= ``support_threshold`` → verdict ``SUPPORTED``.
    * Otherwise → ``INSUFFICIENT_EVIDENCE``.

    The returned confidence is the maximum probability of the label
    that triggered the decision.
    """

    def __init__(self, contradiction_threshold: float = 0.40, support_threshold: float = 0.50):
        self.contradiction_threshold = contradiction_threshold
        self.support_threshold = support_threshold

    def aggregate(self, passages: List[EvidencePassage]) -> tuple[ClaimVerdict, float]:
        """Aggregate a list of ``EvidencePassage`` objects.

        Returns:
            (verdict, confidence)
        """
        # Track best probabilities for each label
        best_contra = 0.0
        best_support = 0.0
        best_insuff = 0.0
        for p in passages:
            probs = p.nli_probabilities
            # Ensure keys exist; fall back to 0.0
            contra = probs.get("CONTRADICTED", 0.0)
            support = probs.get("SUPPORTED", 0.0)
            insuff = probs.get("INSUFFICIENT_EVIDENCE", 0.0)
            if contra > best_contra:
                best_contra = contra
            if support > best_support:
                best_support = support
            if insuff > best_insuff:
                best_insuff = insuff

        # Contradiction‑first logic
        if best_contra >= self.contradiction_threshold:
            return ClaimVerdict.CONTRADICTED, best_contra
        if best_support >= self.support_threshold:
            return ClaimVerdict.SUPPORTED, best_support
        # Insufficient evidence fallback
        return ClaimVerdict.INSUFFICIENT_EVIDENCE, best_insuff
