"""High-level evidence retriever orchestrating indexing, search, and reranking.

Connects:
    Claim -> Embedding -> FAISS Search (scoped to sample's source context) -> Reranker -> Top-K Evidence
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from src.extraction.schema import ExtractedClaim
from src.retrieval.index import EvidenceIndex
from src.retrieval.reranker import BaseReranker, IdentityReranker
from src.retrieval.schema import ClaimEvidenceResult, RetrievedPassage

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Orchestrates candidate evidence retrieval for factual claims.

    Important Principles:
    1. Similarity != Entailment: A passage with high cosine similarity (e.g. 0.85)
       indicates topical relevance to the claim, NOT that the claim is true.
       Entailment or contradiction is strictly evaluated downstream in Phase 5.
    2. Source Scoping / No Leakage: For evaluation datasets like RAGTruth, queries
       should be scoped to the source document assigned to that sample (`source_id`),
       preventing information leakage across evaluation items.
    3. Annotation Isolation: Only raw trusted context text is indexed. Under no
       circumstances are target labels, quality tags, or hallucination spans indexed.
    """

    def __init__(
        self,
        index: EvidenceIndex,
        reranker: BaseReranker | None = None,
        default_top_k: int = 3,
    ) -> None:
        self.index = index
        self.reranker = reranker or IdentityReranker()
        self.default_top_k = default_top_k

    def retrieve_for_claim(
        self,
        claim: ExtractedClaim | str,
        claim_id: str = "C1",
        source_id: str | None = None,
        top_k: int | None = None,
    ) -> ClaimEvidenceResult:
        """Retrieve top-K evidence passages for a single claim.

        Args:
            claim: ExtractedClaim object or raw claim text string.
            claim_id: Identifier for the claim (defaults to claim.claim_id if ExtractedClaim).
            source_id: Optional source document ID to scope retrieval to.
            top_k: Number of passages to return (defaults to self.default_top_k).

        Returns:
            ClaimEvidenceResult containing ranked RetrievedPassage items.
        """
        k = top_k or self.default_top_k
        if isinstance(claim, ExtractedClaim):
            c_text = claim.text
            c_id = claim.claim_id
        else:
            c_text = str(claim)
            c_id = claim_id

        start_time = time.perf_counter()

        # Step 1: Initial FAISS vector retrieval
        raw_passages = self.index.search(
            query=c_text,
            top_k=k,
            source_id_filter=source_id,
        )

        # Step 2: Optional reranking
        final_passages = self.reranker.rerank(query=c_text, passages=raw_passages)

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return ClaimEvidenceResult(
            claim_id=c_id,
            claim_text=c_text,
            evidence=final_passages,
            top_k=k,
            retrieval_latency_ms=round(duration_ms, 2),
        )

    def retrieve_for_claims(
        self,
        claims: Sequence[ExtractedClaim],
        source_id: str | None = None,
        top_k: int | None = None,
    ) -> list[ClaimEvidenceResult]:
        """Retrieve top-K evidence for a collection of claims."""
        return [
            self.retrieve_for_claim(c, source_id=source_id, top_k=top_k)
            for c in claims
        ]
