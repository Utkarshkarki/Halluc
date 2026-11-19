"""Service layer for Phase 6 FastAPI backend.

Wraps the frozen Phase 1-5 ML core pipeline without duplicating detection logic:
- NLIDetector (DeBERTa-v3)
- EvidenceIndex & EvidenceRetriever (BAAI/bge-small-en-v1.5)
- DeterministicChunker
- AtomicClaimExtractor (RuleBasedExtractorProvider default or live LLM if key set)
- EvidenceAggregator
- ResponseScorer
- HallucinationDetector
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from src.detectors.nli_detector import NLIDetector
from src.extraction.extractor import AtomicClaimExtractor
from src.extraction.providers.rule_based_provider import RuleBasedExtractorProvider
from src.pipeline.aggregator import EvidenceAggregator
from src.pipeline.detector import HallucinationDetector
from src.pipeline.schema import PipelineResult
from src.pipeline.scorer import ResponseScorer
from src.retrieval.chunker import DeterministicChunker
from src.retrieval.index import EvidenceIndex
from src.retrieval.retriever import EvidenceRetriever
from src.retrieval.schema import ChunkConfig, EvidenceChunk

from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    EvidencePassageOut,
    ExcludedClaimOut,
    HealthResponse,
    ModelsMetadataResponse,
    ReadyResponse,
    VerifiedClaimOut,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/pipeline_config.json")


class PipelineService:
    """Singleton service wrapping the frozen HallucinationDetector pipeline."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self._load_config()

        # Config extraction
        retrieval_cfg = self.config.get("retrieval", {})
        agg_cfg = self.config.get("aggregation", {})
        scoring_cfg = self.config.get("scoring", {})

        self.top_k = retrieval_cfg.get("top_k", 5)
        self.support_threshold = agg_cfg.get("support_threshold", 0.50)
        self.contradiction_threshold = agg_cfg.get("contradiction_threshold", 0.40)
        self.low_risk_max = scoring_cfg.get("low_risk_max", 0.20)
        self.medium_risk_max = scoring_cfg.get("medium_risk_max", 0.50)
        self.device = self.config.get("device", "auto")

        # Core components
        self.chunk_config = ChunkConfig(chunk_size=300, chunk_overlap=60)
        self.chunker = DeterministicChunker(self.chunk_config)
        self.index: EvidenceIndex | None = None
        self.retriever: EvidenceRetriever | None = None
        self.nli: NLIDetector | None = None
        self.extractor: AtomicClaimExtractor | None = None
        self.aggregator: EvidenceAggregator | None = None
        self.scorer: ResponseScorer | None = None
        self.detector: HallucinationDetector | None = None

        self._lock = threading.Lock()
        self._is_ready = False

    def _load_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not read %s, using defaults: %s", self.config_path, e)
        return {
            "retrieval": {"top_k": 5},
            "aggregation": {"support_threshold": 0.5, "contradiction_threshold": 0.4},
            "scoring": {"low_risk_max": 0.20, "medium_risk_max": 0.50},
            "device": "auto",
        }

    def warmup(self) -> None:
        """Initialize all ML core models and pipeline components."""
        if self._is_ready:
            return

        logger.info("Initializing PipelineService subsystems...")
        t0 = time.perf_counter()

        # 1. NLI Sequence Classifier
        self.nli = NLIDetector.from_pretrained(
            model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            device=self.device,
        )

        # 2. Embedding Index & Retriever
        self.index = EvidenceIndex(
            model_name="BAAI/bge-small-en-v1.5",
            device=self.device,
            chunk_config=self.chunk_config,
        )
        self.retriever = EvidenceRetriever(
            index=self.index,
            default_top_k=self.top_k,
        )

        # 3. Aggregator & Scorer
        self.aggregator = EvidenceAggregator(
            contradiction_threshold=self.contradiction_threshold,
            support_threshold=self.support_threshold,
        )
        self.scorer = ResponseScorer(
            low_threshold=self.low_risk_max,
            medium_threshold=self.medium_risk_max,
        )

        # 4. Atomic Claim Extractor (Offline deterministic RuleBased by default, LLM if configured)
        provider = RuleBasedExtractorProvider()
        self.extractor = AtomicClaimExtractor(provider=provider)

        # 5. Core Pipeline Orchestrator
        self.detector = HallucinationDetector(
            extractor=self.extractor,
            retriever=self.retriever,
            nli_detector=self.nli,
            aggregator=self.aggregator,
            scorer=self.scorer,
        )

        self._is_ready = True
        logger.info("PipelineService ready in %.2fs on %s", time.perf_counter() - t0, self.device)

    def get_health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            version="0.1.0",
        )

    def get_ready(self) -> ReadyResponse:
        device_str = str(self.nli.device) if self.nli else "uninitialized"
        return ReadyResponse(
            ready=self._is_ready,
            components={
                "nli": self.nli is not None,
                "embeddings": self.index is not None,
                "faiss": self.index is not None and self.index._index is not None,
                "claim_extractor": self.extractor is not None,
            },
            device=device_str,
        )

    def get_models_metadata(self) -> ModelsMetadataResponse:
        emb_dim = self.index.embedding_dim if self.index else 384
        device_str = str(self.nli.device) if self.nli else self.device
        provider_name = self.extractor.provider.name if self.extractor else "rule_based"

        return ModelsMetadataResponse(
            schema_version="v1.0.0",
            nli_model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dimension=emb_dim,
            claim_extractor_provider=provider_name,
            retrieval_top_k=self.top_k,
            support_threshold=self.support_threshold,
            contradiction_threshold=self.contradiction_threshold,
            risk_thresholds={
                "low_max": self.low_risk_max,
                "medium_max": self.medium_risk_max,
            },
            device=device_str,
        )

    def _index_request_context(
        self,
        reference_context: str | list[str],
        source_id: str,
    ) -> None:
        """Chunk and index the request-specific reference context with isolated source_id."""
        if self.index is not None and source_id in self.index._source_to_chunk_indices:
            return

        if isinstance(reference_context, str):
            contexts = [reference_context]
        else:
            contexts = reference_context

        chunks: list[EvidenceChunk] = []
        for i, ctx in enumerate(contexts):
            doc_id = f"{source_id}_doc_{i}" if len(contexts) > 1 else source_id
            doc_chunks = self.chunker.chunk_document(
                source_id=source_id,
                text=ctx,
                metadata={"origin": "api_request", "sub_id": doc_id},
            )
            chunks.extend(doc_chunks)

        if self.index is not None and chunks:
            self.index.add_chunks(chunks)

    def analyze(self, request: AnalyzeRequest, request_id: str | None = None) -> AnalyzeResponse:
        """Analyze a response against reference context using the frozen ML core."""
        if not self._is_ready:
            self.warmup()

        with self._lock:
            sample_id = request.sample_id or request_id or f"req_{uuid.uuid4().hex[:12]}"

            # Determine source_id for scoping retrieval
            if request.source_id:
                source_id = request.source_id
            else:
                # Generate deterministic source_id from context text
                ctx_text = (
                    "".join(request.reference_context)
                    if isinstance(request.reference_context, list)
                    else request.reference_context
                )
                h = hashlib.sha256(ctx_text.encode("utf-8")).hexdigest()[:16]
                source_id = f"src_{h}"

            # Dynamically index reference context scoped to source_id
            self._index_request_context(request.reference_context, source_id=source_id)

            # Allow per-request top_k override safely under lock
            original_top_k = self.retriever.default_top_k if self.retriever else self.top_k
            if request.top_k is not None and self.retriever is not None:
                self.retriever.default_top_k = request.top_k

            try:
                # Invoke the frozen HallucinationDetector
                sample_payload = {
                    "sample_id": sample_id,
                    "response": request.response,
                    "source_id": source_id,
                    "question": request.question or "",
                }
                res: PipelineResult = self.detector.process(sample_payload)
            finally:
                if self.retriever is not None:
                    self.retriever.default_top_k = original_top_k

            # Format into clean API response contract
            verified_claims_out: list[VerifiedClaimOut] = []
            for idx, vc in enumerate(res.verified_claims):
                passages_out: list[EvidencePassageOut] = []
                for ev in vc.evidence:
                    passages_out.append(
                        EvidencePassageOut(
                            chunk_id=ev.chunk_id,
                            source_id=ev.source_id,
                            text=ev.text,
                            similarity_score=round(ev.similarity_score, 4),
                            rank=ev.rank,
                            nli_verdict=ev.nli_label.value,
                            nli_confidence=round(ev.nli_confidence, 4),
                            nli_probabilities={
                                k: round(v, 4) for k, v in ev.nli_probabilities.items()
                            },
                        )
                    )
                claim_id = vc.claim_id if vc.claim_id else f"claim_{idx + 1}"
                verified_claims_out.append(
                    VerifiedClaimOut(
                        claim_id=claim_id,
                        claim_text=vc.claim_text,
                        verdict=vc.verdict.value,
                        confidence=round(vc.confidence, 4),
                        evidence=passages_out,
                    )
                )

            excluded_claims_out = [
                ExcludedClaimOut(
                    claim_id=ec.get("claim_id") or f"ex_claim_{i + 1}",
                    text=ec.get("text", ""),
                    type=ec.get("type", "NON_VERIFIABLE"),
                )
                for i, ec in enumerate(res.excluded_claims)
            ]

            metadata = dict(res.metadata)
            metadata.update({
                "source_id": source_id,
                "question": request.question,
            })

            return AnalyzeResponse(
                schema_version=res.schema_version,
                sample_id=res.sample_id,
                hallucination_risk=round(res.hallucination_risk, 4),
                risk_level=res.risk_level,
                support_rate=round(res.support_rate, 4),
                evidence_coverage=round(res.evidence_coverage, 4),
                abstention_rate=round(res.abstention_rate, 4),
                total_factual_claims=res.total_factual_claims,
                total_claims=res.total_claims,
                verified_claims=verified_claims_out,
                excluded_claims=excluded_claims_out,
                latency_ms={k: round(v, 2) for k, v in res.latency_ms.items()},
                metadata=metadata,
            )


# Global service singleton
_service_instance: PipelineService | None = None


def get_pipeline_service() -> PipelineService:
    global _service_instance
    if _service_instance is None:
        _service_instance = PipelineService()
    return _service_instance
