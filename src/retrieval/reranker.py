"""Extensible reranker interfaces for evidence passages.

Allows plugging in cross-encoders or custom scoring without altering
the primary vector retrieval pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.retrieval.schema import RetrievedPassage


class BaseReranker(ABC):
    """Abstract interface for passage reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        passages: Sequence[RetrievedPassage],
    ) -> list[RetrievedPassage]:
        """Rerank a list of retrieved passages relative to query.

        Args:
            query: The claim or question text.
            passages: Candidate passages from initial vector retrieval.

        Returns:
            Reranked list of RetrievedPassage with updated scores and ranks.
        """
        ...


class IdentityReranker(BaseReranker):
    """Default pass-through reranker preserving FAISS vector ranking."""

    def rerank(
        self,
        query: str,
        passages: Sequence[RetrievedPassage],
    ) -> list[RetrievedPassage]:
        # Preserves initial ranking
        return list(passages)
