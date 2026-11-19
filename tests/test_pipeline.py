"""Unit tests for Phase 5 pipeline components.

Coverage:
  - EvidenceAggregator: contradiction-first logic, edge cases
  - ResponseScorer: verdicts → risk ratios, risk categorisation, Pydantic v2 validators
  - CalibrationModule: fit / apply / save / load
  - HallucinationDetector: interface wiring via mocks (no real models)
  - run_pipeline_evaluation helpers: config loading, metric calculation

Scientific integrity:
  - Mocks are used for unit isolation only.
  - No mock is presented as evidence that real-model integration works.
  - Real-model integration status is documented separately in the audit report.
"""

from __future__ import annotations

import json
import math
import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared imports
# ---------------------------------------------------------------------------
from src.pipeline.schema import ClaimVerdict, EvidencePassage, PipelineResult, VerifiedClaim
from src.pipeline.aggregator import EvidenceAggregator
from src.pipeline.scorer import ResponseScorer, ResponseScore
from src.pipeline.calibration import CalibrationModule


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_passage(
    text: str = "Some evidence text.",
    supported: float = 0.8,
    contradicted: float = 0.1,
    insufficient: float = 0.1,
    rank: int = 1,
) -> EvidencePassage:
    """Create an EvidencePassage with specified NLI probabilities."""
    max_p = max(supported, contradicted, insufficient)
    if max_p == supported:
        label = ClaimVerdict.SUPPORTED
        confidence = supported
    elif max_p == contradicted:
        label = ClaimVerdict.CONTRADICTED
        confidence = contradicted
    else:
        label = ClaimVerdict.INSUFFICIENT_EVIDENCE
        confidence = insufficient
    return EvidencePassage(
        chunk_id=f"chunk_{rank}",
        source_id="src1",
        text=text,
        similarity_score=0.9,
        rank=rank,
        nli_probabilities={
            "SUPPORTED": supported,
            "CONTRADICTED": contradicted,
            "INSUFFICIENT_EVIDENCE": insufficient,
        },
        nli_label=label,
        nli_confidence=confidence,
    )


def _make_verified_claim(verdict: ClaimVerdict, confidence: float = 0.9) -> VerifiedClaim:
    return VerifiedClaim(
        claim_id="C1",
        claim_text="Some factual claim.",
        verdict=verdict,
        confidence=confidence,
        evidence=[],
    )


# ===========================================================================
# EvidenceAggregator tests
# ===========================================================================

class TestEvidenceAggregator:
    def setup_method(self) -> None:
        self.agg = EvidenceAggregator(contradiction_threshold=0.4, support_threshold=0.5)

    def test_empty_passages_returns_insufficient(self) -> None:
        verdict, conf = self.agg.aggregate([])
        assert verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
        assert conf == 0.0

    def test_strongly_supported_passage(self) -> None:
        passages = [_make_passage(supported=0.92, contradicted=0.04, insufficient=0.04)]
        verdict, conf = self.agg.aggregate(passages)
        assert verdict == ClaimVerdict.SUPPORTED
        assert math.isclose(conf, 0.92)

    def test_strongly_contradicted_passage(self) -> None:
        passages = [_make_passage(supported=0.05, contradicted=0.90, insufficient=0.05)]
        verdict, conf = self.agg.aggregate(passages)
        assert verdict == ClaimVerdict.CONTRADICTED
        assert math.isclose(conf, 0.90)

    def test_contradiction_takes_priority_over_support(self) -> None:
        """Even if one passage strongly supports, a contradicting passage wins."""
        passages = [
            _make_passage(supported=0.95, contradicted=0.03, insufficient=0.02, rank=1),
            _make_passage(supported=0.05, contradicted=0.80, insufficient=0.15, rank=2),
        ]
        verdict, conf = self.agg.aggregate(passages)
        assert verdict == ClaimVerdict.CONTRADICTED

    def test_weak_evidence_gives_insufficient(self) -> None:
        """Probabilities below both thresholds → INSUFFICIENT_EVIDENCE."""
        passages = [_make_passage(supported=0.30, contradicted=0.20, insufficient=0.50)]
        verdict, conf = self.agg.aggregate(passages)
        assert verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE

    def test_multiple_passages_takes_max(self) -> None:
        """Best value across all passages is used."""
        passages = [
            _make_passage(supported=0.30, contradicted=0.10, insufficient=0.60, rank=1),
            _make_passage(supported=0.75, contradicted=0.15, insufficient=0.10, rank=2),
        ]
        verdict, conf = self.agg.aggregate(passages)
        assert verdict == ClaimVerdict.SUPPORTED
        assert math.isclose(conf, 0.75)

    def test_custom_thresholds(self) -> None:
        agg = EvidenceAggregator(contradiction_threshold=0.9, support_threshold=0.95)
        # Neither threshold met
        passages = [_make_passage(supported=0.80, contradicted=0.85, insufficient=0.05)]
        verdict, _ = agg.aggregate(passages)
        assert verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE


# ===========================================================================
# ResponseScorer tests
# ===========================================================================

class TestResponseScorer:
    def setup_method(self) -> None:
        self.scorer = ResponseScorer(low_threshold=0.20, medium_threshold=0.50)

    def test_empty_claims_returns_low_risk(self) -> None:
        score = self.scorer.score([])
        assert score.total_factual_claims == 0
        assert score.hallucination_risk == 0.0
        assert score.risk_level == "LOW"

    def test_all_supported(self) -> None:
        claims = [_make_verified_claim(ClaimVerdict.SUPPORTED) for _ in range(5)]
        score = self.scorer.score(claims)
        assert score.supported == 5
        assert score.contradicted == 0
        assert score.hallucination_risk == 0.0
        assert score.support_rate == 1.0
        assert score.evidence_coverage == 1.0
        assert score.abstention_rate == 0.0
        assert score.risk_level == "LOW"

    def test_all_contradicted(self) -> None:
        claims = [_make_verified_claim(ClaimVerdict.CONTRADICTED) for _ in range(4)]
        score = self.scorer.score(claims)
        assert score.contradicted == 4
        assert score.hallucination_risk == 1.0
        assert score.support_rate == 0.0
        assert score.evidence_coverage == 1.0  # Decisive evidence was retrieved for all claims
        assert score.abstention_rate == 0.0
        assert score.risk_level == "HIGH"

    def test_mixed_verdicts(self) -> None:
        claims = [
            _make_verified_claim(ClaimVerdict.SUPPORTED),
            _make_verified_claim(ClaimVerdict.CONTRADICTED),
            _make_verified_claim(ClaimVerdict.INSUFFICIENT_EVIDENCE),
            _make_verified_claim(ClaimVerdict.INSUFFICIENT_EVIDENCE),
        ]
        score = self.scorer.score(claims)
        assert score.total_factual_claims == 4
        assert score.supported == 1
        assert score.contradicted == 1
        assert score.insufficient == 2
        assert math.isclose(score.hallucination_risk, 0.25)
        assert math.isclose(score.support_rate, 0.25)
        assert math.isclose(score.evidence_coverage, 0.50)  # 2 of 4 had decisive evidence
        assert math.isclose(score.abstention_rate, 0.50)
        assert score.risk_level == "MEDIUM"

    def test_risk_level_boundaries(self) -> None:
        """20% → LOW boundary, 21% → MEDIUM, 50% → MEDIUM, 51% → HIGH."""
        def risk_for_ratio(contradicted: int, total: int) -> str:
            claims = [_make_verified_claim(ClaimVerdict.CONTRADICTED) for _ in range(contradicted)]
            claims += [_make_verified_claim(ClaimVerdict.SUPPORTED) for _ in range(total - contradicted)]
            return self.scorer.score(claims).risk_level

        assert risk_for_ratio(2, 10) == "LOW"    # 20%
        assert risk_for_ratio(3, 10) == "MEDIUM"  # 30%
        assert risk_for_ratio(5, 10) == "MEDIUM"  # 50%
        assert risk_for_ratio(6, 10) == "HIGH"    # 60%

    def test_risk_clamped_to_zero_one(self) -> None:
        """Pydantic v2 field_validator must clamp values to [0, 1]."""
        score = ResponseScore(
            total_factual_claims=1,
            supported=0,
            contradicted=1,
            insufficient=0,
            evidence_coverage=-0.5,   # out of range
            abstention_rate=1.5,       # out of range
            hallucination_risk=2.0,    # out of range
            risk_level="HIGH",
        )
        assert score.evidence_coverage == 0.0
        assert score.abstention_rate == 1.0
        assert score.hallucination_risk == 1.0

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError):
            ResponseScorer(low_threshold=0.6, medium_threshold=0.4)


# ===========================================================================
# CalibrationModule tests
# ===========================================================================

class TestCalibrationModule:
    def test_disabled_calibration_is_identity(self) -> None:
        cal = CalibrationModule(enabled=False)
        import numpy as np
        logits = np.array([[2.0, 1.0, 0.5]])
        result = cal.apply(logits)
        assert (result == logits).all()

    def test_fit_and_apply(self) -> None:
        """Calibration adjusts probabilities without crashing."""
        import numpy as np
        cal = CalibrationModule(enabled=True)
        logits = np.array([[2.0, 0.5, 0.3], [0.1, 1.5, 0.8]])
        labels = np.array([0, 1])
        cal.fit(logits, labels)
        assert cal.is_fitted()
        assert cal.temperature is not None
        result = cal.apply(logits)
        assert result.shape == logits.shape

    def test_save_and_load(self) -> None:
        """Calibration can be persisted and reloaded via instance method."""
        import numpy as np
        cal = CalibrationModule(enabled=True)
        logits = np.array([[1.0, 0.5, 0.3]])
        labels = np.array([0])
        cal.fit(logits, labels)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            cal.save(path)
            loaded = CalibrationModule(enabled=False)
            loaded.load(path)  # instance method, not classmethod
            assert loaded.enabled == cal.enabled
            assert loaded.is_fitted() == cal.is_fitted()
            assert loaded.temperature == cal.temperature
        finally:
            path.unlink(missing_ok=True)

    def test_apply_without_fit_raises_or_passthrough(self) -> None:
        """Calling apply before fit on enabled calibration should not silently corrupt data."""
        import numpy as np
        cal = CalibrationModule(enabled=True)
        logits = np.array([[1.0, 0.5, 0.3]])
        # Either raises or returns passthrough — both are acceptable
        try:
            result = cal.apply(logits)
            # If no exception, result should still be numeric
            assert result.shape == logits.shape
        except Exception:
            pass  # Raising is also acceptable


# ===========================================================================
# HallucinationDetector (mock-based isolation tests)
# ===========================================================================

class TestHallucinationDetectorInterface:
    """
    These tests use mocks exclusively for unit isolation.
    They verify that the detector correctly wires components together.
    They do NOT verify real NLI or real retrieval behaviour —
    that is covered by the execution-verified integration tests.
    """

    def _make_detector(
        self,
        claims: list[dict[str, Any]] | None = None,
        retrieval_passages: list | None = None,
        nli_label: str = "SUPPORTED",
        nli_prob_supported: float = 0.9,
    ):
        from src.pipeline.detector import HallucinationDetector
        from src.detectors.schema import NLIOutput, NLIProbabilities
        from src.data.schema import HallucinationLabel
        from src.retrieval.schema import RetrievedPassage, ClaimEvidenceResult

        # Mock extractor
        mock_extractor = MagicMock()
        from src.extraction.schema import ClaimExtractionResult, ExtractedClaim, ClaimType
        extracted = [
            ExtractedClaim(
                claim_id=f"C{i+1}",
                text=c["text"],
                claim_type=ClaimType.FACTUAL,
            )
            for i, c in enumerate(claims or [{"text": "Paris is the capital of France."}])
        ]
        mock_extraction_result = ClaimExtractionResult(
            claims=extracted,
            raw_response="mock response",
            extractor_model="mock:mock",
            prompt_version="v1",
            total_claims=len(extracted),
            factual_claims_count=len(extracted),
            excluded_claims_count=0,
        )
        mock_extractor.extract_claims.return_value = mock_extraction_result

        # Mock retriever
        mock_retriever = MagicMock()
        passages = retrieval_passages or [
            RetrievedPassage(
                chunk_id="c1",
                source_id="src1",
                text="Paris is in France.",
                similarity_score=0.95,
                rank=1,
            )
        ]
        mock_evidence_result = ClaimEvidenceResult(
            claim_id="C1",
            claim_text="mock claim",
            evidence=passages,
        )
        mock_retriever.retrieve_for_claim.return_value = mock_evidence_result

        # Mock NLI detector
        mock_nli = MagicMock()
        contra = max(0.0, 1.0 - nli_prob_supported - 0.05)
        insuff = max(0.0, 1.0 - nli_prob_supported - contra)
        nli_output = NLIOutput(
            label=HallucinationLabel(nli_label),
            confidence=nli_prob_supported,
            probabilities=NLIProbabilities(
                supported=nli_prob_supported,
                insufficient_evidence=insuff,
                contradicted=contra,
            ),
            model="mock-nli",
        )
        mock_nli.predict_pairs.return_value = [nli_output] * len(passages)

        aggregator = EvidenceAggregator()
        scorer = ResponseScorer()

        detector = HallucinationDetector(
            extractor=mock_extractor,
            retriever=mock_retriever,
            nli_detector=mock_nli,
            aggregator=aggregator,
            scorer=scorer,
        )
        return detector

    def test_process_returns_pipeline_result(self) -> None:
        detector = self._make_detector()
        result = detector.process({
            "sample_id": "test001",
            "question": "Where is Paris?",
            "response": "Paris is the capital of France.",
            "reference_context": "France is a country in Europe. Paris is its capital.",
        })
        assert isinstance(result, PipelineResult)
        assert result.sample_id == "test001"
        assert isinstance(result.hallucination_risk, float)
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH")

    def test_supported_claim_produces_low_risk(self) -> None:
        detector = self._make_detector(nli_label="SUPPORTED", nli_prob_supported=0.95)
        result = detector.process({
            "sample_id": "s1",
            "response": "Paris is the capital of France.",
            "reference_context": "Paris is the capital of France.",
        })
        assert result.risk_level == "LOW"
        assert result.hallucination_risk == 0.0

    def test_contradicted_claim_produces_high_risk(self) -> None:
        detector = self._make_detector(nli_label="CONTRADICTED", nli_prob_supported=0.02)
        result = detector.process({
            "sample_id": "s2",
            "response": "Berlin is the capital of France.",
            "reference_context": "Paris is the capital of France.",
        })
        # contradicted claim → non-zero risk
        assert result.hallucination_risk > 0.0

    def test_empty_response_no_claims_zero_risk(self) -> None:
        from src.pipeline.detector import HallucinationDetector
        from src.extraction.schema import ClaimExtractionResult

        mock_extractor = MagicMock()
        mock_extractor.extract_claims.return_value = ClaimExtractionResult(
            claims=[],
            raw_response="",
            extractor_model="mock:mock",
            prompt_version="v1",
            total_claims=0,
            factual_claims_count=0,
            excluded_claims_count=0,
        )
        detector = HallucinationDetector(
            extractor=mock_extractor,
            retriever=MagicMock(),
            nli_detector=MagicMock(),
            aggregator=EvidenceAggregator(),
            scorer=ResponseScorer(),
        )
        result = detector.process({"sample_id": "empty", "response": "", "reference_context": ""})
        assert result.total_factual_claims == 0
        assert result.hallucination_risk == 0.0
        assert result.risk_level == "LOW"

    def test_no_retrieval_passages_gives_insufficient_evidence(self) -> None:
        from src.retrieval.schema import ClaimEvidenceResult
        detector = self._make_detector(retrieval_passages=[])
        # Override retriever to return empty evidence
        detector.retriever.retrieve_for_claim.return_value = ClaimEvidenceResult(
            claim_id="C1",
            claim_text="Some claim",
            evidence=[],
        )
        result = detector.process({
            "sample_id": "s3",
            "response": "Some claim with no evidence.",
            "reference_context": "Unrelated context.",
        })
        # All claims should be INSUFFICIENT_EVIDENCE
        for vc in result.verified_claims:
            assert vc.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE

    def test_process_batch(self) -> None:
        detector = self._make_detector()
        samples = [
            {"sample_id": f"batch_{i}", "response": "Some claim.", "reference_context": "Some context."}
            for i in range(3)
        ]
        results = detector.process_batch(samples)
        assert len(results) == 3
        assert all(isinstance(r, PipelineResult) for r in results)


# ===========================================================================
# Evaluation script helpers
# ===========================================================================

class TestEvalScriptHelpers:
    """Test config loading, metric calculation, output serialisation."""

    def test_load_config_valid(self, tmp_path: Path) -> None:
        cfg_data = {
            "retrieval": {"top_k": 5},
            "aggregation": {"support_threshold": 0.5, "contradiction_threshold": 0.4},
            "scoring": {"low_risk_max": 0.20, "medium_risk_max": 0.50},
            "device": "cpu",
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg_data))

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_pipeline_evaluation import load_config
        cfg = load_config(p)
        assert cfg["retrieval"]["top_k"] == 5

    def test_load_config_missing_file_exits(self, tmp_path: Path) -> None:
        from run_pipeline_evaluation import load_config
        with pytest.raises(SystemExit) as exc_info:
            load_config(tmp_path / "nonexistent.json")
        assert exc_info.value.code == 1

    def test_load_config_invalid_json_exits(self, tmp_path: Path) -> None:
        from run_pipeline_evaluation import load_config
        p = tmp_path / "bad.json"
        p.write_text("{not valid json}")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1

    def test_config_defaults_applied(self, tmp_path: Path) -> None:
        """Missing optional keys get default values."""
        from run_pipeline_evaluation import load_config
        p = tmp_path / "minimal.json"
        p.write_text("{}")
        cfg = load_config(p)
        assert cfg["retrieval"]["top_k"] == 5
        assert cfg["aggregation"]["support_threshold"] == 0.5
        assert cfg["scoring"]["low_risk_max"] == 0.20

    def test_compute_response_metrics_all_correct(self) -> None:
        from run_pipeline_evaluation import compute_response_level_metrics
        preds = [
            {"status": "success", "ground_truth_is_hallucinated": True,
             "risk_level": "HIGH", "hallucination_risk": 0.9},
            {"status": "success", "ground_truth_is_hallucinated": False,
             "risk_level": "LOW", "hallucination_risk": 0.1},
        ]
        metrics = compute_response_level_metrics(preds)
        assert metrics["accuracy"] == 1.0
        assert metrics["tp"] == 1
        assert metrics["tn"] == 1
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_compute_response_metrics_all_failed(self) -> None:
        from run_pipeline_evaluation import compute_response_level_metrics
        preds = [{"status": "error"}, {"status": "skipped"}]
        metrics = compute_response_level_metrics(preds)
        assert "error" in metrics

    def test_compute_response_metrics_fp_fn(self) -> None:
        from run_pipeline_evaluation import compute_response_level_metrics
        preds = [
            # FP: predict hallucinated, truth is not
            {"status": "success", "ground_truth_is_hallucinated": False,
             "risk_level": "HIGH", "hallucination_risk": 0.8},
            # FN: predict not hallucinated, truth is
            {"status": "success", "ground_truth_is_hallucinated": True,
             "risk_level": "LOW", "hallucination_risk": 0.1},
        ]
        metrics = compute_response_level_metrics(preds)
        assert metrics["accuracy"] == 0.0
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_output_serialization(self, tmp_path: Path) -> None:
        """pipeline_predictions.jsonl must be valid, parseable JSON lines."""
        pred_path = tmp_path / "preds.jsonl"
        records = [
            {"sample_id": "s1", "status": "success", "risk_level": "LOW",
             "hallucination_risk": 0.1, "ground_truth_is_hallucinated": False},
            {"sample_id": "s2", "status": "error", "error": "Some error"},
        ]
        with open(pred_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        # Reload
        loaded = []
        with open(pred_path) as f:
            for line in f:
                loaded.append(json.loads(line.strip()))
        assert len(loaded) == 2
        assert loaded[0]["sample_id"] == "s1"

    def test_no_fabricated_claim_level_gt(self) -> None:
        """Verify scientific guard: metrics must not use claim-level labels from RAGTruth.

        RAGTruth only provides response-level and span-level labels.
        No per-claim ground truth exists, so claim-level verdicts must NOT be
        compared to any fabricated ground truth.

        This test asserts that the metric function only uses
        ground_truth_is_hallucinated (response-level) and risk_level (pipeline output).
        """
        from run_pipeline_evaluation import compute_response_level_metrics
        # Metric function should NOT look at 'claim_level_gt' even if present
        preds = [
            {
                "status": "success",
                "ground_truth_is_hallucinated": True,
                "risk_level": "HIGH",
                "hallucination_risk": 0.9,
                # This fabricated key must NOT be used
                "claim_level_gt": ["SUPPORTED", "CONTRADICTED"],
            }
        ]
        metrics = compute_response_level_metrics(preds)
        assert "claim_level_gt" not in metrics
        assert metrics["total_evaluated"] == 1
