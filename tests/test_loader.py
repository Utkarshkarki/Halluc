"""Unit tests for dataset loading and raw persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.loader import DatasetAcquisitionError, RAGTruthLoader
from src.data.schema import RawResponseRecord, RawSourceRecord


@pytest.fixture
def sample_raw_response() -> dict:
    return {
        "id": "1001",
        "source_id": "src_42",
        "model": "gpt-4-0613",
        "temperature": 0.7,
        "labels": [
            {
                "start": 10,
                "end": 20,
                "text": "test claim",
                "label_type": "Evident Conflict",
                "due_to_null": False,
                "implicit_true": False,
                "meta": "Annotation note",
            }
        ],
        "split": "train",
        "quality": "good",
        "response": "Here is a test claim that was made.",
    }


@pytest.fixture
def sample_raw_source() -> dict:
    return {
        "source_id": "src_42",
        "task_type": "QA",
        "source": "MARCO",
        "source_info": {
            "question": "What is tested?",
            "passages": "passage 1: This is a verified fact without the claim.",
        },
        "prompt": "Answer the question based on passage.",
    }


def test_loader_from_local_and_save_raw(tmp_path: Path, sample_raw_response, sample_raw_source):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Write sample jsonl files
    with open(raw_dir / "response.jsonl", "w") as f:
        f.write(json.dumps(sample_raw_response) + "\n")

    with open(raw_dir / "source_info.jsonl", "w") as f:
        f.write(json.dumps(sample_raw_source) + "\n")

    loader = RAGTruthLoader.from_local(raw_dir)
    assert len(loader.get_responses()) == 1
    assert len(loader.get_sources()) == 1

    resp = loader.get_responses()[0]
    assert isinstance(resp, RawResponseRecord)
    assert resp.id == "1001"
    assert resp.source_id == "src_42"

    src = loader.get_sources()[0]
    assert isinstance(src, RawSourceRecord)
    assert src.source_id == "src_42"
    assert src.task_type == "QA"

    # Test save_raw
    out_dir = tmp_path / "saved_raw"
    loader.save_raw(out_dir)
    assert (out_dir / "response.jsonl").exists()
    assert (out_dir / "source_info.jsonl").exists()
    assert (out_dir / "dataset_info.json").exists()


def test_loader_missing_local_file_raises(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        RAGTruthLoader.from_local(empty_dir)


def test_loader_acquisition_error_reporting(monkeypatch):
    import datasets

    def fake_load_dataset(*args, **kwargs):
        raise ConnectionError("Network unreachable")

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    with pytest.raises(DatasetAcquisitionError) as exc_info:
        RAGTruthLoader.from_huggingface()
    assert "DATASET_ACQUISITION_BLOCKED" in str(exc_info.value)

