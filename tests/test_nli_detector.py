"""Unit and integration tests for the NLI detector and dataset adapter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.adapter import DatasetNLIAdapter, NLIPair
from src.data.schema import HallucinationLabel, NormalizedSample
from src.detectors.nli_detector import NLIDetector
from src.detectors.schema import NLIConfig, NLIOutput, NLIProbabilities


class DummyModel(torch.nn.Module):
    """Mock PyTorch sequence classification model for fast unit testing."""

    def __init__(self, logits_matrix: torch.Tensor):
        super().__init__()
        self.logits_matrix = logits_matrix
        self.cursor = 0
        self.config = type(
            "Config",
            (),
            {"id2label": {0: "entailment", 1: "neutral", 2: "contradiction"}},
        )()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        start = self.cursor % len(self.logits_matrix)
        end = start + batch_size
        self.cursor += batch_size

        if end <= len(self.logits_matrix):
            logits = self.logits_matrix[start:end]
        else:
            logits = torch.cat([self.logits_matrix[start:], self.logits_matrix[: end - len(self.logits_matrix)]])
        return type("Output", (), {"logits": logits})()


class DummyTokenizer:
    """Mock HuggingFace tokenizer."""

    def __init__(self, max_len: int = 512):
        self.max_len = max_len

    def encode(self, text: str, text_pair: str | None = None, add_special_tokens: bool = True, truncation: bool = False):
        total = len(text.split()) + (len(text_pair.split()) if text_pair else 0)
        if truncation and total > self.max_len:
            total = self.max_len
        return [1] * total

    def __call__(self, text, text_pair=None, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        if isinstance(text, str):
            text = [text]
        if isinstance(text_pair, str):
            text_pair = [text_pair]

        batch_size = len(text)
        max_seen = 10
        input_ids = torch.ones((batch_size, max_seen), dtype=torch.long)
        attention_mask = torch.ones((batch_size, max_seen), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def mock_detector() -> NLIDetector:
    # 3 samples:
    # 0: highest is entailment (0) -> SUPPORTED
    # 1: highest is contradiction (2) -> CONTRADICTED
    # 2: highest is neutral (1) -> INSUFFICIENT_EVIDENCE
    logits = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.5, 0.5, 6.0],
            [1.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    model = DummyModel(logits)
    tokenizer = DummyTokenizer(max_len=20)
    config = NLIConfig(
        model_name="mock-deberta",
        batch_size=2,
        max_length=20,
        device="cpu",
        fp16=False,
    )
    return NLIDetector(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=torch.device("cpu"),
        entailment_idx=0,
        neutral_idx=1,
        contradiction_idx=2,
    )


def test_label_mapping_and_probabilities(mock_detector: NLIDetector):
    pairs = [
        ("Evidence fact", "Claim matches fact"),
        ("Evidence fact", "Claim contradicts fact"),
        ("Evidence fact", "Claim unmentioned"),
    ]
    outputs = mock_detector.predict_pairs(pairs, batch_size=2)
    assert len(outputs) == 3

    # Sample 0: Entailment -> SUPPORTED
    assert outputs[0].label == HallucinationLabel.SUPPORTED
    assert outputs[0].confidence > 0.9
    assert outputs[0].probabilities.supported > 0.9

    # Sample 1: Contradiction -> CONTRADICTED
    assert outputs[1].label == HallucinationLabel.CONTRADICTED
    assert outputs[1].confidence > 0.9
    assert outputs[1].probabilities.contradicted > 0.9

    # Sample 2: Neutral -> INSUFFICIENT_EVIDENCE
    assert outputs[2].label == HallucinationLabel.INSUFFICIENT_EVIDENCE
    assert outputs[2].confidence > 0.8
    assert outputs[2].probabilities.insufficient_evidence > 0.8

    # Probability normalization check: each must sum to ~1.0
    for out in outputs:
        p = out.probabilities
        total = p.supported + p.insufficient_evidence + p.contradicted
        assert total == pytest.approx(1.0, abs=1e-4)


def test_output_schema_contract(mock_detector: NLIDetector):
    out = mock_detector.predict_pair("Fact", "Statement")
    assert isinstance(out, NLIOutput)
    contract_dict = out.to_contract_dict()

    assert "label" in contract_dict
    assert "confidence" in contract_dict
    assert "probabilities" in contract_dict
    assert "model" in contract_dict
    assert "truncated" in contract_dict

    probs = contract_dict["probabilities"]
    assert "supported" in probs
    assert "insufficient_evidence" in probs
    assert "contradicted" in probs


def test_empty_and_whitespace_input_handling(mock_detector: NLIDetector):
    # Should not raise exception
    out = mock_detector.predict_pair("", "")
    assert isinstance(out, NLIOutput)

    out_ws = mock_detector.predict_pair("   ", "   ")
    assert isinstance(out_ws, NLIOutput)


def test_long_input_truncation_detection(mock_detector: NLIDetector):
    # max_length is 20 words for the mock tokenizer
    short_text = "short sentence"
    long_text = "word " * 30  # 30 words > 20 words

    out_short = mock_detector.predict_pair(short_text, short_text)
    assert not out_short.truncated

    out_long = mock_detector.predict_pair(long_text, short_text)
    assert out_long.truncated


def test_base_detector_interface_compliance(mock_detector: NLIDetector):
    sample = NormalizedSample(
        sample_id="test_sample_1",
        source_id="src_1",
        task_type="QA",
        question_or_prompt="Q?",
        reference_context="Evidence passage",
        generated_response="Response claim",
        hallucination_label=HallucinationLabel.SUPPORTED,
        split="test",
    )
    pred = mock_detector.predict(sample)
    assert pred.sample_id == "test_sample_1"
    assert pred.predicted_label in (
        HallucinationLabel.SUPPORTED,
        HallucinationLabel.CONTRADICTED,
        HallucinationLabel.INSUFFICIENT_EVIDENCE,
    )
    assert 0.0 <= pred.confidence <= 1.0


def test_dataset_adapter():
    sample = NormalizedSample(
        sample_id="s123",
        source_id="src99",
        task_type="Summary",
        question_or_prompt="Summarize article",
        reference_context="Ground truth document text.",
        generated_response="Summary generated by model.",
        hallucination_label=HallucinationLabel.CONTRADICTED,
        split="train",
        metadata={"model": "gpt-4"},
    )
    pair = DatasetNLIAdapter.sample_to_pair(sample)
    assert isinstance(pair, NLIPair)
    assert pair.sample_id == "s123"
    assert pair.premise == "Ground truth document text."
    assert pair.hypothesis == "Summary generated by model."
    assert pair.ground_truth_label == HallucinationLabel.CONTRADICTED
    assert pair.task_type == "Summary"
    assert pair.metadata["model"] == "gpt-4"

    dataset_pairs = DatasetNLIAdapter.adapt_dataset([sample])
    assert len(dataset_pairs) == 1


def test_real_model_smoke_integration():
    """Smoke test exercising the real DeBERTa-v3 model locally."""
    detector = NLIDetector.from_pretrained(
        model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=torch.cuda.is_available(),
    )
    assert detector.name == "nli-MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    out_entail = detector.predict_pair(
        "Paris is the capital of France.",
        "The capital of France is Paris.",
    )
    assert out_entail.label == HallucinationLabel.SUPPORTED
    assert out_entail.confidence > 0.9

    out_contra = detector.predict_pair(
        "Paris is the capital of France.",
        "Madrid is the capital of France.",
    )
    assert out_contra.label == HallucinationLabel.CONTRADICTED
    assert out_contra.confidence > 0.9

