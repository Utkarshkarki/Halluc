"""Unit tests for preprocessing, schema validation, and span verification."""

from __future__ import annotations

import pytest

from src.data.preprocessing import (
    _classify_label,
    compute_stats,
    merge_responses_and_sources,
    validate_dataset,
    validate_sample,
)
from src.data.schema import (
    HallucinationLabel,
    HallucinationSpan,
    NormalizedSample,
    RawResponseRecord,
    RawSourceRecord,
)


def test_classify_label_taxonomy():
    # Empty -> SUPPORTED
    assert _classify_label([]) == HallucinationLabel.SUPPORTED

    # Conflict -> CONTRADICTED
    conflict_labels = [{"label_type": "Evident Conflict", "text": "bad", "start": 0, "end": 3}]
    assert _classify_label(conflict_labels) == HallucinationLabel.CONTRADICTED

    # Subtle Conflict -> CONTRADICTED
    subtle_conflict = [{"label_type": "Subtle Conflict", "text": "bad", "start": 0, "end": 3}]
    assert _classify_label(subtle_conflict) == HallucinationLabel.CONTRADICTED

    # Baseless Info only -> INSUFFICIENT_EVIDENCE
    baseless_labels = [{"label_type": "Evident Baseless Info", "text": "new", "start": 0, "end": 3}]
    assert _classify_label(baseless_labels) == HallucinationLabel.INSUFFICIENT_EVIDENCE

    # Mixed Conflict and Baseless -> CONTRADICTED
    mixed = [
        {"label_type": "Evident Baseless Info", "text": "new", "start": 0, "end": 3},
        {"label_type": "Evident Conflict", "text": "bad", "start": 4, "end": 7},
    ]
    assert _classify_label(mixed) == HallucinationLabel.CONTRADICTED


def test_merge_responses_and_sources():
    responses = [
        RawResponseRecord(
            id="resp_1",
            source_id="src_1",
            model="mistral-7B",
            temperature=0.5,
            labels=[{"start": 0, "end": 5, "text": "Hello", "label_type": "Evident Conflict"}],
            split="train",
            quality="good",
            response="Hello world!",
        ),
        RawResponseRecord(
            id="resp_2",
            source_id="src_missing",
            model="gpt-4",
            temperature=0.0,
            labels=[],
            split="test",
            quality="good",
            response="All true information.",
        ),
    ]

    sources = [
        RawSourceRecord(
            source_id="src_1",
            task_type="Summary",
            source="CNN/DM",
            source_info="Original article text here.",
            prompt="Summarize this text:",
        )
    ]

    samples = merge_responses_and_sources(responses, sources)
    assert len(samples) == 2

    # Verify first sample joined properly
    s1 = samples[0]
    assert s1.sample_id == "resp_1"
    assert s1.source_id == "src_1"
    assert s1.task_type == "Summary"
    assert s1.hallucination_label == HallucinationLabel.CONTRADICTED
    assert len(s1.hallucinated_spans) == 1
    assert s1.hallucinated_spans[0].text == "Hello"
    assert s1.metadata["model"] == "mistral-7B"

    # Verify missing source handling
    s2 = samples[1]
    assert s2.sample_id == "resp_2"
    assert s2.task_type == "unknown"
    assert s2.hallucination_label == HallucinationLabel.SUPPORTED


def test_span_validation():
    # Valid span
    valid_sample = NormalizedSample(
        sample_id="v1",
        source_id="s1",
        task_type="QA",
        question_or_prompt="Question?",
        reference_context="Context",
        generated_response="The capital of France is Madrid.",
        hallucination_label=HallucinationLabel.CONTRADICTED,
        hallucinated_spans=[
            HallucinationSpan(
                start=25,
                end=31,
                text="Madrid",
                label_type="Evident Conflict",
            )
        ],
        split="train",
    )
    res_valid = validate_sample(valid_sample)
    assert res_valid.is_valid
    assert len(res_valid.errors) == 0

    # Span text mismatch
    mismatch_sample = NormalizedSample(
        sample_id="v2",
        source_id="s1",
        task_type="QA",
        question_or_prompt="Question?",
        reference_context="Context",
        generated_response="The capital of France is Madrid.",
        hallucination_label=HallucinationLabel.CONTRADICTED,
        hallucinated_spans=[
            HallucinationSpan(
                start=25,
                end=31,
                text="Paris",  # Mismatch! Response has 'Madrid'
                label_type="Evident Conflict",
            )
        ],
        split="train",
    )
    res_mismatch = validate_sample(mismatch_sample)
    assert not res_mismatch.is_valid
    assert any("text mismatch" in e for e in res_mismatch.errors)

    # Out of bounds span
    oob_sample = NormalizedSample(
        sample_id="v3",
        source_id="s1",
        task_type="QA",
        question_or_prompt="Question?",
        reference_context="Context",
        generated_response="Short.",
        hallucination_label=HallucinationLabel.CONTRADICTED,
        hallucinated_spans=[
            HallucinationSpan(
                start=10,
                end=50,
                text="nonexistent",
                label_type="Evident Baseless Info",
            )
        ],
        split="train",
    )
    res_oob = validate_sample(oob_sample)
    assert not res_oob.is_valid
    assert any("exceeds response length" in e for e in res_oob.errors)


def test_validate_dataset_and_stats():
    valid_sample = NormalizedSample(
        sample_id="s_1",
        source_id="src_1",
        task_type="QA",
        question_or_prompt="Q",
        reference_context="Ctx",
        generated_response="Answer here.",
        hallucination_label=HallucinationLabel.SUPPORTED,
        split="train",
    )
    invalid_sample = NormalizedSample(
        sample_id="",  # Missing ID
        source_id="src_1",
        task_type="QA",
        question_or_prompt="Q",
        reference_context="Ctx",
        generated_response="",  # Empty response
        hallucination_label=HallucinationLabel.SUPPORTED,
        split="train",
    )

    valid_list, invalid_list = validate_dataset([valid_sample, invalid_sample])
    assert len(valid_list) == 1
    assert len(invalid_list) == 1
    assert invalid_list[0][0].sample_id == ""

    # Test compute stats
    stats = compute_stats(valid_list, invalid_count=len(invalid_list))
    assert stats.total_responses == 1
    assert stats.hallucinated_responses == 0
    assert stats.non_hallucinated_responses == 1
    assert stats.invalid_records == 1
    assert stats.task_distribution == {"QA": 1}
