"""Unit tests for evidence chunking, FAISS indexing, compatibility, and source isolation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.extraction.schema import ClaimType, ExtractedClaim
from src.retrieval.chunker import DeterministicChunker
from src.retrieval.index import EvidenceIndex, IndexCompatibilityError
from src.retrieval.retriever import EvidenceRetriever
from src.retrieval.schema import ChunkConfig, EvidenceChunk


@pytest.fixture
def test_chunker() -> DeterministicChunker:
    return DeterministicChunker(ChunkConfig(chunk_size=150, chunk_overlap=30))


@pytest.fixture
def sample_text() -> str:
    return (
        "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire. "
        "In 1880, his family moved to Munich, where his father and his uncle founded a company. "
        "He later moved to Switzerland and renounced his German citizenship."
    )


def test_deterministic_chunking(test_chunker: DeterministicChunker, sample_text: str):
    chunks1 = test_chunker.chunk_document(source_id="src_42", text=sample_text)
    chunks2 = test_chunker.chunk_document(source_id="src_42", text=sample_text)

    # Identical inputs must yield identical chunks
    assert len(chunks1) > 1
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.char_start == c2.char_start
        assert c1.char_end == c2.char_end
        assert c1.source_id == "src_42"


def test_chunking_metadata_preservation(test_chunker: DeterministicChunker):
    meta = {"task_type": "QA", "benchmark": "MARCO"}
    chunks = test_chunker.chunk_document(
        source_id="src_99",
        text="Sentence one is clear. Sentence two is also very clear.",
        metadata=meta,
    )
    for c in chunks:
        assert c.source_id == "src_99"
        assert c.metadata["task_type"] == "QA"
        assert c.metadata["benchmark"] == "MARCO"


def test_empty_and_short_corpus_handling(test_chunker: DeterministicChunker):
    # Empty
    assert test_chunker.chunk_document(source_id="s1", text="") == []
    assert test_chunker.chunk_document(source_id="s1", text="   ") == []

    # Very short
    short_chunks = test_chunker.chunk_document(source_id="s1", text="Short.")
    assert len(short_chunks) == 1
    assert short_chunks[0].text == "Short."
    assert short_chunks[0].char_start == 0
    assert short_chunks[0].char_end == 6


def test_faiss_index_and_retrieval(test_chunker: DeterministicChunker, sample_text: str):
    chunks = test_chunker.chunk_document(source_id="src_einstein", text=sample_text)
    index = EvidenceIndex(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
    )
    index.add_chunks(chunks)
    assert index.total_chunks == len(chunks)

    retriever = EvidenceRetriever(index=index, default_top_k=2)

    # Query matching birth in Germany
    claim = ExtractedClaim(
        claim_id="C1",
        text="Einstein was born in Germany.",
        claim_type=ClaimType.FACTUAL,
    )
    res = retriever.retrieve_for_claim(claim, top_k=2)

    assert res.claim_id == "C1"
    assert len(res.evidence) == 2
    assert res.evidence[0].rank == 1
    assert res.evidence[1].rank == 2
    # The top retrieved chunk should contain birth or Germany
    top_text = res.evidence[0].text.lower()
    assert "born" in top_text or "german" in top_text


def test_source_isolation_prevents_leakage(test_chunker: DeterministicChunker):
    doc_a = "Isaac Newton discovered gravity in England and formulated laws of motion."
    doc_b = "Marie Curie discovered radium and polonium in Paris and won two Nobel Prizes."

    chunks_a = test_chunker.chunk_document(source_id="doc_newton", text=doc_a)
    chunks_b = test_chunker.chunk_document(source_id="doc_curie", text=doc_b)

    index = EvidenceIndex(model_name="all-MiniLM-L6-v2", device="cpu")
    index.add_chunks(chunks_a + chunks_b)

    retriever = EvidenceRetriever(index=index, default_top_k=3)

    # Query about Newton, but scoped STRICTLY to doc_curie
    res_isolated = retriever.retrieve_for_claim(
        claim="Newton discovered gravity",
        source_id="doc_curie",  # Isolated to Curie's document
        top_k=3,
    )
    # Must only contain chunks from doc_curie, NEVER doc_newton
    for ev in res_isolated.evidence:
        assert ev.source_id == "doc_curie"
        assert "Newton" not in ev.text


def test_index_save_and_load(tmp_path: Path, test_chunker: DeterministicChunker, sample_text: str):
    chunks = test_chunker.chunk_document(source_id="src_save", text=sample_text)
    index = EvidenceIndex(model_name="all-MiniLM-L6-v2", device="cpu")
    index.add_chunks(chunks)

    save_dir = tmp_path / "test_idx"
    index.save(save_dir)

    assert (save_dir / "index.faiss").exists()
    assert (save_dir / "chunks_metadata.json").exists()
    assert (save_dir / "index_metadata.json").exists()

    # Load back
    loaded_index = EvidenceIndex.load(save_dir, device="cpu")
    assert loaded_index.total_chunks == index.total_chunks
    assert loaded_index.model_name == index.model_name
    assert loaded_index.embedding_dim == index.embedding_dim

    # Test search on loaded index
    results = loaded_index.search("Germany", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id.startswith("src_save")


def test_model_index_incompatibility_detection(tmp_path: Path, test_chunker: DeterministicChunker):
    chunks = test_chunker.chunk_document(source_id="s1", text="Sample text for index.")
    index = EvidenceIndex(model_name="all-MiniLM-L6-v2", device="cpu")
    index.add_chunks(chunks)

    save_dir = tmp_path / "idx_compat"
    index.save(save_dir)

    # Attempting to load with a DIFFERENT model name must raise IndexCompatibilityError
    with pytest.raises(IndexCompatibilityError) as exc:
        EvidenceIndex.load(save_dir, model_name="BAAI/bge-small-en-v1.5", device="cpu")
    assert "Model incompatibility detected" in str(exc.value)


def test_empty_query_and_empty_index():
    index = EvidenceIndex(model_name="all-MiniLM-L6-v2", device="cpu")
    retriever = EvidenceRetriever(index=index)

    # Empty query on empty index
    res1 = retriever.retrieve_for_claim("")
    assert res1.evidence == []

    # Non-empty query on empty index
    res2 = retriever.retrieve_for_claim("Test claim")
    assert res2.evidence == []


def test_no_label_leakage_in_chunks(test_chunker: DeterministicChunker):
    # Ensure chunker only extracts content from raw context, never injecting labels
    sample_raw_context = "The patient tested positive for COVID-19 and was quarantined for 14 days."
    chunks = test_chunker.chunk_document(source_id="med_01", text=sample_raw_context)

    for c in chunks:
        # Verify no label strings leaked
        assert "SUPPORTED" not in c.text
        assert "CONTRADICTED" not in c.text
        assert "INSUFFICIENT_EVIDENCE" not in c.text
        assert "hallucination" not in c.text
