"""FAISS vector evidence index with strict compatibility validation and source isolation.

Uses SentenceTransformers for local dense embeddings (e.g. BAAI/bge-small-en-v1.5)
and FAISS IndexFlatIP for exact cosine similarity search over normalized vectors.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.schema import (
    ChunkConfig,
    EvidenceChunk,
    IndexMetadata,
    RetrievedPassage,
)

logger = logging.getLogger(__name__)


class IndexCompatibilityError(Exception):
    """Raised when an index is loaded with an incompatible embedding model or dimension."""


class EvidenceIndex:
    """Vector index for evidence passages with persistence and metadata tracking."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        normalize_embeddings: bool = True,
        chunk_config: ChunkConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.chunk_config = chunk_config or ChunkConfig()

        # Resolve 'auto' to a concrete device string; SentenceTransformer
        # does not accept the string "auto" — it must be "cpu" or "cuda".
        import torch as _torch
        if device == "auto":
            resolved_device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        self._device = resolved_device

        # Initialize embedding model locally
        self._model = SentenceTransformer(self.model_name, device=self._device)
        if hasattr(self._model, "get_embedding_dimension"):
            self.embedding_dim = int(self._model.get_embedding_dimension())
        else:
            self.embedding_dim = int(self._model.get_sentence_embedding_dimension())

        # FAISS IndexFlatIP computes inner product (equivalent to cosine similarity on normalized vectors)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.embedding_dim)
        self._chunks: list[EvidenceChunk] = []
        self._source_to_chunk_indices: dict[str, list[int]] = {}

        logger.info(
            "EvidenceIndex initialized with model=%s (dim=%d, device=%s)",
            self.model_name,
            self.embedding_dim,
            self._device,
        )

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def total_sources(self) -> int:
        return len(self._source_to_chunk_indices)

    # ----- Indexing -------------------------------------------------------

    def add_chunks(
        self,
        chunks: Sequence[EvidenceChunk],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> None:
        """Embed and add evidence chunks to the FAISS index."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        ).astype(np.float32)

        start_idx = len(self._chunks)
        self._index.add(embeddings)

        for i, c in enumerate(chunks):
            idx = start_idx + i
            self._chunks.append(c)
            self._source_to_chunk_indices.setdefault(c.source_id, []).append(idx)

        logger.info(
            "Indexed %d new chunks (total in index: %d, sources: %d)",
            len(chunks),
            self.total_chunks,
            self.total_sources,
        )

    # ----- Search & Retrieval ---------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        source_id_filter: str | None = None,
    ) -> list[RetrievedPassage]:
        """Retrieve top-K most similar evidence passages for a query.

        Args:
            query: Claim or question text to match.
            top_k: Maximum number of passages to return.
            source_id_filter: If specified, restricts candidate pool strictly to chunks
                              originating from this source_id (enforcing evaluation isolation).

        Returns:
            List of RetrievedPassage ranked by similarity score.
        """
        q_text = query.strip()
        if not q_text or self.total_chunks == 0:
            return []

        # Encode query
        q_emb = self._model.encode(
            [q_text],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Case A: Scoped retrieval for a specific source_id (prevents evaluation leakage)
        if source_id_filter is not None:
            candidate_indices = self._source_to_chunk_indices.get(source_id_filter, [])
            if not candidate_indices:
                logger.debug("No indexed chunks found for source_id: %s", source_id_filter)
                return []

            # Extract vectors for candidate indices
            cand_vectors = np.vstack([
                self._index.reconstruct(int(idx)) for idx in candidate_indices
            ])
            # Inner product scores
            scores = np.dot(cand_vectors, q_emb[0])
            sorted_order = np.argsort(-scores)[:top_k]

            results: list[RetrievedPassage] = []
            for rank, local_idx in enumerate(sorted_order, start=1):
                global_idx = candidate_indices[local_idx]
                chunk = self._chunks[global_idx]
                results.append(
                    RetrievedPassage(
                        chunk_id=chunk.chunk_id,
                        source_id=chunk.source_id,
                        text=chunk.text,
                        similarity_score=float(scores[local_idx]),
                        rank=rank,
                    )
                )
            return results

        # Case B: Global retrieval across full index
        k_val = min(top_k, self.total_chunks)
        distances, indices = self._index.search(q_emb, k_val)

        results = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    text=chunk.text,
                    similarity_score=float(score),
                    rank=rank,
                )
            )
        return results

    # ----- Persistence & Compatibility ------------------------------------

    def save(self, directory: Path | str) -> None:
        """Persist FAISS index, chunk metadata, and compatibility info to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 1. Save FAISS binary index
        faiss_path = dir_path / "index.faiss"
        faiss.write_index(self._index, str(faiss_path))

        # 2. Save chunks metadata
        chunks_path = dir_path / "chunks_metadata.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self._chunks], f, indent=2)

        # 3. Save index metadata and model compatibility contract
        meta = IndexMetadata(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            normalize_embeddings=self.normalize_embeddings,
            total_chunks=self.total_chunks,
            total_sources=self.total_sources,
            chunk_config=self.chunk_config,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        meta_path = dir_path / "index_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(meta.model_dump_json(indent=2))

        logger.info("Saved EvidenceIndex to %s (%d chunks)", dir_path, self.total_chunks)

    @classmethod
    def load(
        cls,
        directory: Path | str,
        model_name: str | None = None,
        device: str = "auto",
    ) -> EvidenceIndex:
        """Load persisted index and strictly validate embedding model compatibility."""
        dir_path = Path(directory)
        meta_path = dir_path / "index_metadata.json"
        faiss_path = dir_path / "index.faiss"
        chunks_path = dir_path / "chunks_metadata.json"

        if not meta_path.exists() or not faiss_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Missing required index files in {dir_path}")

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = IndexMetadata.model_validate_json(f.read())

        target_model = model_name or meta.model_name

        # Compatibility Check: reject mismatched models
        if target_model != meta.model_name:
            raise IndexCompatibilityError(
                f"Model incompatibility detected! The index was created with model '{meta.model_name}' "
                f"(dim={meta.embedding_dim}), but load requested model '{target_model}'. "
                "Embeddings from different models cannot be queried in the same space."
            )

        # Create instance
        instance = cls(
            model_name=meta.model_name,
            device=device,
            normalize_embeddings=meta.normalize_embeddings,
            chunk_config=meta.chunk_config,
        )

        if instance.embedding_dim != meta.embedding_dim:
            raise IndexCompatibilityError(
                f"Dimension mismatch! Loaded model embedding dimension is {instance.embedding_dim}, "
                f"but persisted index dimension is {meta.embedding_dim}."
            )

        # Load FAISS index
        instance._index = faiss.read_index(str(faiss_path))

        # Load chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        instance._chunks = [EvidenceChunk.model_validate(rc) for rc in raw_chunks]
        instance._source_to_chunk_indices = {}
        for i, c in enumerate(instance._chunks):
            instance._source_to_chunk_indices.setdefault(c.source_id, []).append(i)

        logger.info(
            "Loaded EvidenceIndex from %s (%d chunks, %d sources)",
            dir_path,
            instance.total_chunks,
            instance.total_sources,
        )
        return instance
