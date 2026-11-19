"""Schemas for evidence chunking, indexing, and retrieval.

Defines data models for:
- Chunking configuration
- Evidence chunks with provenance and span offsets
- Retrieved evidence passages with similarity scores and ranks
- Claim-level retrieval results adhering to the Phase 4 output contract
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ChunkConfig(BaseModel):
    """Configuration for deterministic evidence chunking."""

    chunk_size: int = Field(
        default=300,
        description="Target maximum chunk length in characters (~50-60 words/tokens).",
    )
    chunk_overlap: int = Field(
        default=60,
        description="Character overlap between consecutive chunks to preserve boundary context.",
    )
    method: str = Field(
        default="sentence_boundary_sliding_window",
        description="Chunking strategy name.",
    )


class EvidenceChunk(BaseModel):
    """An individual indexed evidence passage preserving full provenance."""

    chunk_id: str = Field(description="Unique identifier, e.g. '{source_id}_c{idx}'")
    source_id: str = Field(description="Identifier of the origin source document")
    text: str = Field(description="Raw text content of the chunk")
    char_start: int | None = Field(
        default=None,
        description="Start character offset within the source context.",
    )
    char_end: int | None = Field(
        default=None,
        description="End character offset within the source context.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedPassage(BaseModel):
    """A retrieved evidence passage matching the Phase 4 output specification."""

    chunk_id: str = Field(description="ID of the matching chunk")
    source_id: str = Field(description="ID of the source document")
    text: str = Field(description="Evidence text")
    similarity_score: float = Field(
        description="Cosine similarity score in [-1.0, 1.0], higher indicates higher relevance."
    )
    rank: int = Field(ge=1, description="1-based rank among top-K results")


class ClaimEvidenceResult(BaseModel):
    """Structured evidence retrieval result for a single claim."""

    claim_id: str
    claim_text: str
    evidence: list[RetrievedPassage] = Field(default_factory=list)
    top_k: int = 3
    retrieval_latency_ms: float = 0.0

    def to_contract_dict(self) -> dict[str, Any]:
        """Serialize according to the Phase 4 output contract."""
        return {
            "claim_id": self.claim_id,
            "evidence": [
                {
                    "chunk_id": ep.chunk_id,
                    "source_id": ep.source_id,
                    "text": ep.text,
                    "similarity_score": round(ep.similarity_score, 4),
                    "rank": ep.rank,
                }
                for ep in self.evidence
            ],
        }


class IndexMetadata(BaseModel):
    """Provenance and compatibility metadata for persisted FAISS index."""

    model_name: str
    embedding_dim: int
    normalize_embeddings: bool = True
    total_chunks: int = 0
    total_sources: int = 0
    chunk_config: ChunkConfig
    created_at: str
