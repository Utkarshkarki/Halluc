# src/pipeline/detector.py
"""Orchestrator for the end‑to‑end hallucination detection pipeline (Phase 5).

The :class:`HallucinationDetector` glues together the existing components
(AtomicClaimExtractor, EvidenceRetriever, NLIDetector, EvidenceAggregator)
with the new ``ResponseScorer`` (and optional ``CalibrationModule``).

Bug-fixes applied during Phase 5 audit:
  - extract_claims() returns ClaimExtractionResult (Pydantic model), not dict.
    Use .claims / .get_factual_claims() instead of .get("claims", []).
  - retrieve_for_claim() returns ClaimEvidenceResult, not List[EvidencePassage].
    Convert evidence passages from retrieval schema to pipeline EvidencePassage.
  - NLI results are NLIOutput Pydantic objects, not plain dicts.
    Use attribute access (result.label, result.probabilities) instead of dict access.
  - NLIProbabilities uses lowercase keys (supported/contradicted/insufficient_evidence)
    but EvidenceAggregator expects uppercase (SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE).
    Map correctly when building the nli_probabilities dict.
"""

from __future__ import annotations

import time
from typing import List, Dict, Any

from .schema import (
    PipelineResult,
    VerifiedClaim,
    ClaimVerdict,
    EvidencePassage,
)
from .aggregator import EvidenceAggregator
from .scorer import ResponseScorer
from .calibration import CalibrationModule

# Existing components
from src.extraction.extractor import AtomicClaimExtractor
from src.extraction.schema import ClaimExtractionResult
from src.retrieval.retriever import EvidenceRetriever
from src.detectors.nli_detector import NLIDetector
from src.detectors.schema import NLIOutput


def _nli_probs_to_pipeline_dict(nli_output: NLIOutput) -> dict[str, float]:
    """Convert NLIOutput.probabilities (lowercase keys) → aggregator-compatible dict.

    EvidenceAggregator.aggregate() reads keys:
        'CONTRADICTED', 'SUPPORTED', 'INSUFFICIENT_EVIDENCE'
    NLIProbabilities stores:
        .contradicted, .supported, .insufficient_evidence
    """
    probs = nli_output.probabilities
    return {
        "SUPPORTED": probs.supported,
        "CONTRADICTED": probs.contradicted,
        "INSUFFICIENT_EVIDENCE": probs.insufficient_evidence,
    }


class HallucinationDetector:
    """Main pipeline class for Phase 5.

    Parameters
    ----------
    extractor: AtomicClaimExtractor
    retriever: EvidenceRetriever
    nli_detector: NLIDetector
    aggregator: EvidenceAggregator
    scorer: ResponseScorer
    calibration: CalibrationModule | None, optional
        If provided and enabled, calibrates NLI logits before aggregation.
    """

    def __init__(
        self,
        extractor: AtomicClaimExtractor,
        retriever: EvidenceRetriever,
        nli_detector: NLIDetector,
        aggregator: EvidenceAggregator,
        scorer: ResponseScorer,
        calibration: CalibrationModule | None = None,
    ) -> None:
        self.extractor = extractor
        self.retriever = retriever
        self.nli = nli_detector
        self.aggregator = aggregator
        self.scorer = scorer
        self.calibration = calibration

    # ------------------------------------------------------------------
    def _process_claim(self, claim_text: str, source_id: str | None = None) -> VerifiedClaim:
        """Retrieve evidence, run NLI, optionally calibrate, and aggregate."""
        # retrieve_for_claim() → ClaimEvidenceResult with .evidence (list[RetrievedPassage])
        evidence_result = self.retriever.retrieve_for_claim(claim_text, source_id=source_id)
        raw_passages = evidence_result.evidence  # list[RetrievedPassage]

        if not raw_passages:
            return VerifiedClaim(
                claim_id="",
                claim_text=claim_text,
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                evidence=[],
            )

        # Run NLI on each passage – batch mode
        pairs = [(p.text, claim_text) for p in raw_passages]
        # predict_pairs() returns List[NLIOutput] (Pydantic objects, not dicts)
        nli_results: List[NLIOutput] = self.nli.predict_pairs(pairs)

        # Build EvidencePassage list (pipeline schema) merging retrieval + NLI
        passages: List[EvidencePassage] = []
        for rp, nli_out in zip(raw_passages, nli_results):
            # Map NLIOutput.label (HallucinationLabel) → ClaimVerdict
            # Both enums share the same string values: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE
            try:
                nli_verdict = ClaimVerdict(nli_out.label.value)
            except ValueError:
                nli_verdict = ClaimVerdict.INSUFFICIENT_EVIDENCE

            prob_dict = _nli_probs_to_pipeline_dict(nli_out)

            passages.append(
                EvidencePassage(
                    chunk_id=rp.chunk_id,
                    source_id=rp.source_id,
                    text=rp.text,
                    similarity_score=rp.similarity_score,
                    rank=rp.rank,
                    nli_probabilities=prob_dict,
                    nli_label=nli_verdict,
                    nli_confidence=nli_out.confidence,
                )
            )

        # Aggregation – contradiction-first logic
        verdict, confidence = self.aggregator.aggregate(passages)
        return VerifiedClaim(
            claim_id="",
            claim_text=claim_text,
            verdict=verdict,
            confidence=confidence,
            evidence=passages,
        )

    # ------------------------------------------------------------------
    def process(self, sample: Dict[str, Any]) -> PipelineResult:
        """Process a single sample and return a ``PipelineResult``.

        Expected ``sample`` keys:
        - ``sample_id`` (str)
        - ``question`` (str)
        - ``response`` (str)
        - ``reference_context`` (list[str] | str) – trusted corpus for retrieval
        - ``source_id`` (str | None, optional) – used to scope retrieval
        """
        start_time = time.perf_counter()
        latency: Dict[str, float] = {}

        sample_id = sample.get("sample_id", "unknown")
        response_text = sample.get("response", "")
        source_id = sample.get("source_id", None)

        # ------------------------------------------------------------------
        # 1. Claim extraction – returns ClaimExtractionResult (Pydantic model)
        t0 = time.perf_counter()
        extraction_result: ClaimExtractionResult = self.extractor.extract_claims(
            response_text, sample_id=sample_id
        )
        latency["extraction"] = (time.perf_counter() - t0) * 1000.0

        # Use typed accessor methods, not .get()
        factual_claims = extraction_result.get_factual_claims()   # List[ExtractedClaim]
        total_claims = extraction_result.total_claims

        # ------------------------------------------------------------------
        # 2. Per‑claim processing
        verified: List[VerifiedClaim] = []
        t_claims_start = time.perf_counter()
        for claim in factual_claims:
            verified.append(self._process_claim(claim.text, source_id=source_id))
        latency["claim_processing"] = (time.perf_counter() - t_claims_start) * 1000.0

        # ------------------------------------------------------------------
        # 3. Scoring
        t_score = time.perf_counter()
        response_score = self.scorer.score(verified)
        latency["scoring"] = (time.perf_counter() - t_score) * 1000.0
        latency["total"] = (time.perf_counter() - start_time) * 1000.0

        # ------------------------------------------------------------------
        excluded = [
            {"claim_id": c.claim_id, "text": c.text, "type": c.claim_type.value}
            for c in extraction_result.get_excluded_claims()
        ]

        result = PipelineResult(
            sample_id=sample_id,
            hallucination_risk=response_score.hallucination_risk,
            risk_level=response_score.risk_level,
            support_rate=response_score.support_rate,
            evidence_coverage=response_score.evidence_coverage,
            abstention_rate=response_score.abstention_rate,
            total_factual_claims=response_score.total_factual_claims,
            total_claims=total_claims,
            verified_claims=verified,
            excluded_claims=excluded,
            latency_ms=latency,
            metadata={},
        )
        return result

    def process_batch(self, samples: List[Dict[str, Any]]) -> List[PipelineResult]:
        """Convenience wrapper for batch processing."""
        return [self.process(s) for s in samples]


__all__ = ["HallucinationDetector"]
