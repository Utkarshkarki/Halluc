"""Unit tests for dataset splitting, group leakage prevention, and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.schema import HallucinationLabel, NormalizedSample
from src.data.splitter import (
    check_leakage,
    create_stratified_split,
    load_splits,
    preserve_official_split,
    save_splits,
)


@pytest.fixture
def mock_grouped_samples() -> list[NormalizedSample]:
    """Create 12 samples across 4 distinct source documents (3 responses per source)."""
    samples = []
    for src_idx in range(4):
        src_id = f"source_{src_idx}"
        for resp_idx in range(3):
            samples.append(
                NormalizedSample(
                    sample_id=f"resp_{src_idx}_{resp_idx}",
                    source_id=src_id,
                    task_type="QA",
                    question_or_prompt=f"Prompt for {src_id}",
                    reference_context=f"Context for {src_id}",
                    generated_response=f"Response {resp_idx} for {src_id}",
                    hallucination_label=HallucinationLabel.SUPPORTED,
                    split="train" if src_idx < 3 else "test",
                )
            )
    return samples


def test_preserve_official_split(mock_grouped_samples):
    splits = preserve_official_split(mock_grouped_samples)
    assert "train" in splits
    assert "test" in splits
    assert len(splits["train"]) == 9
    assert len(splits["test"]) == 3

    leakage = check_leakage(splits, group_key="source_id")
    assert not leakage["has_leakage"]
    assert len(leakage["leaked_groups"]) == 0


def test_group_based_split_prevents_leakage(mock_grouped_samples):
    # Split into train/val with 50/50 ratio
    splits = create_stratified_split(
        mock_grouped_samples,
        ratios={"train": 0.5, "val": 0.5},
        seed=42,
        group_key="source_id",
    )

    # Verify no source_id appears in both splits
    train_sources = {s.source_id for s in splits["train"]}
    val_sources = {s.source_id for s in splits["val"]}
    assert len(train_sources & val_sources) == 0

    leakage = check_leakage(splits, group_key="source_id")
    assert not leakage["has_leakage"]


def test_deterministic_split(mock_grouped_samples):
    # Same seed produces identical splits
    s1 = create_stratified_split(
        mock_grouped_samples,
        ratios={"train": 0.7, "test": 0.3},
        seed=123,
    )
    s2 = create_stratified_split(
        mock_grouped_samples,
        ratios={"train": 0.7, "test": 0.3},
        seed=123,
    )
    assert [x.sample_id for x in s1["train"]] == [x.sample_id for x in s2["train"]]
    assert [x.sample_id for x in s1["test"]] == [x.sample_id for x in s2["test"]]


def test_leakage_detection_catches_overlap(mock_grouped_samples):
    # Intentionally create a leaking split
    leaked_splits = {
        "train": [mock_grouped_samples[0], mock_grouped_samples[1]],
        "test": [mock_grouped_samples[2]],  # shares source_id "source_0" with sample 0 and 1!
    }
    leak_result = check_leakage(leaked_splits, group_key="source_id")
    assert leak_result["has_leakage"]
    assert "source_0" in leak_result["leaked_groups"]


def test_save_and_load_splits(tmp_path: Path, mock_grouped_samples):
    splits = create_stratified_split(
        mock_grouped_samples,
        ratios={"train": 0.5, "test": 0.5},
        seed=42,
    )
    save_splits(splits, tmp_path, seed=42)

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    assert (tmp_path / "split_metadata.json").exists()

    loaded = load_splits(tmp_path)
    assert len(loaded["train"]) == len(splits["train"])
    assert len(loaded["test"]) == len(splits["test"])
    assert {s.sample_id for s in loaded["train"]} == {s.sample_id for s in splits["train"]}
