"""Unit tests for atomic claim extraction, classification, caching, and retry behavior."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.data.schema import NormalizedSample
from src.extraction.cache import ClaimExtractionCache
from src.extraction.extractor import AtomicClaimExtractor
from src.extraction.providers.anthropic_provider import AnthropicProvider
from src.extraction.providers.mock_provider import MockLLMProvider
from src.extraction.providers.rule_based_provider import RuleBasedExtractorProvider
from src.extraction.schema import (
    ClaimExtractionPayload,
    ClaimExtractionResult,
    ClaimType,
    ExtractedClaim,
)


def test_extracted_claim_schema_validation():
    # Valid claim
    c = ExtractedClaim(
        claim_id="C1",
        text="Albert Einstein was born in Germany.",
        claim_type=ClaimType.FACTUAL,
        source_start=0,
        source_end=35,
    )
    assert c.is_verifiable()
    assert c.claim_type == ClaimType.FACTUAL

    # Non-factual claim
    c_op = ExtractedClaim(
        claim_id="C2",
        text="This is an awesome concept.",
        claim_type=ClaimType.OPINION,
    )
    assert not c_op.is_verifiable()

    # Payload validation
    payload = ClaimExtractionPayload(claims=[c, c_op])
    assert len(payload.claims) == 2


def test_compound_claim_decomposition_contract():
    provider = RuleBasedExtractorProvider()
    extractor = AtomicClaimExtractor(provider=provider)

    input_text = "Albert Einstein was born in Germany in 1879 and received the 1921 Nobel Prize for relativity."
    res = extractor.extract_claims(input_text, sample_id="einstein_01")

    assert res.total_claims >= 3
    assert res.factual_claims_count >= 3
    assert res.excluded_claims_count == 0

    claim_texts = [c.text for c in res.claims]
    # Check that individual propositions are decoupled
    assert any("born in Germany" in t for t in claim_texts)
    assert any("1879" in t for t in claim_texts)
    assert any("Nobel Prize" in t for t in claim_texts)


def test_empty_and_whitespace_response():
    provider = RuleBasedExtractorProvider()
    extractor = AtomicClaimExtractor(provider=provider)

    # Empty string
    res_empty = extractor.extract_claims("")
    assert res_empty.total_claims == 0
    assert res_empty.claims == []
    assert res_empty.latency_ms == 0.0

    # Whitespace string
    res_ws = extractor.extract_claims("    \n\t  ")
    assert res_ws.total_claims == 0


def test_non_factual_and_opinion_responses():
    provider = RuleBasedExtractorProvider()
    extractor = AtomicClaimExtractor(provider=provider)

    opinion_text = "In my opinion, this is the most terrible movie ever made. Please avoid it! Hope this helps."
    res = extractor.extract_claims(opinion_text)

    assert res.total_claims >= 2
    types = [c.claim_type for c in res.claims]
    assert ClaimType.OPINION in types
    assert (ClaimType.INSTRUCTION in types) or (ClaimType.NON_VERIFIABLE in types)
    assert res.excluded_claims_count > 0


def test_malformed_provider_output_bounded_retry():
    # Mock provider that fails twice, then succeeds on attempt 3
    valid_payload = {
        "claims": [
            {
                "claim_id": "C1",
                "text": "A valid claim statement.",
                "claim_type": "FACTUAL",
                "source_start": None,
                "source_end": None,
            }
        ]
    }
    mock_prov = MockLLMProvider(canned_response=valid_payload, fail_times=2)
    extractor = AtomicClaimExtractor(provider=mock_prov, max_retries=2)

    res = extractor.extract_claims("Some text to extract.")
    # Succeeded on attempt 3 (calls_count == 3)
    assert mock_prov.calls_count == 3
    assert res.total_claims == 1
    assert len(res.validation_errors) == 2  # Recorded the 2 failed attempts


def test_retry_limit_exhaustion():
    # Mock provider that always fails
    mock_prov = MockLLMProvider(canned_response={}, fail_times=10)
    extractor = AtomicClaimExtractor(provider=mock_prov, max_retries=2)

    # Should not raise uncaught exception or loop infinitely
    res = extractor.extract_claims("Some text.")
    assert mock_prov.calls_count == 3  # Initial attempt + 2 retries
    assert res.total_claims == 0
    assert len(res.validation_errors) == 3


def test_cache_behavior(tmp_path: Path):
    cache = ClaimExtractionCache(tmp_path)
    mock_prov = MockLLMProvider(
        canned_response={
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Cached fact statement.",
                    "claim_type": "FACTUAL",
                }
            ]
        }
    )
    extractor = AtomicClaimExtractor(provider=mock_prov, cache=cache)

    text = "Query text to be cached."
    # First call: cache miss
    res1 = extractor.extract_claims(text)
    assert not res1.cached
    assert mock_prov.calls_count == 1

    # Second call: cache hit
    res2 = extractor.extract_claims(text)
    assert res2.cached
    assert res2.total_claims == 1
    assert res2.claims[0].text == "Cached fact statement."
    assert mock_prov.calls_count == 1  # Provider was NOT called again!

    # Different text: cache miss
    res3 = extractor.extract_claims("Different text.")
    assert not res3.cached
    assert mock_prov.calls_count == 2


def test_provenance_tracking():
    provider = RuleBasedExtractorProvider()
    extractor = AtomicClaimExtractor(provider=provider)

    sample = NormalizedSample(
        sample_id="sample_test_99",
        source_id="src_42",
        task_type="QA",
        question_or_prompt="When was Newton born?",
        reference_context="Context passage",
        generated_response="Isaac Newton was born in England in 1642.",
        hallucination_label="SUPPORTED",
        split="test",
    )
    res = extractor.extract_from_sample(sample)
    assert res.sample_id == "sample_test_99"
    assert res.raw_response == "Isaac Newton was born in England in 1642."
    assert "rule_based" in res.extractor_model
    assert res.prompt_version == "v1.0.0"


def test_secret_safety_in_anthropic_provider(monkeypatch):
    # Ensure api key is never logged or exposed in string representation
    fake_key = "sk-ant-api03-SECRET_DO_NOT_EXPOSE"
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_key)

    provider = AnthropicProvider(model="claude-3-5-sonnet")
    repr_str = repr(provider)
    assert fake_key not in repr_str
    assert fake_key not in str(provider)
