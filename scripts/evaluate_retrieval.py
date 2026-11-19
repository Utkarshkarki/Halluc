#!/usr/bin/env python3
"""Evaluate evidence indexing and Top-K retrieval for factual claims on RAGTruth.

Connects:
    Sample Response -> Phase 3 Claim Extractor -> Factual Claims -> Phase 4 Evidence Retriever -> Top-K Evidence

Evaluates:
- End-to-end integration
- Top-K retrieval latency & throughput
- Source-scoped retrieval to prevent evaluation leakage
- Qualitative inspection cases of claims paired with retrieved evidence passages
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.extractor import AtomicClaimExtractor
from src.extraction.providers.rule_based_provider import RuleBasedExtractorProvider
from src.retrieval.chunker import DeterministicChunker
from src.retrieval.index import EvidenceIndex
from src.retrieval.retriever import EvidenceRetriever
from src.retrieval.schema import ChunkConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_retrieval")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate evidence retrieval for factual claims")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/splits/official/test.jsonl"),
        help="Path to evaluation test split (default: data/splits/official/test.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Directory to save retrieval evaluation report",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=25,
        help="Number of evaluation samples to process (default: 25)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of evidence chunks to retrieve per claim (default: 3)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Chunker and Index
    chunk_config = ChunkConfig(chunk_size=300, chunk_overlap=60)
    chunker = DeterministicChunker(chunk_config)
    index = EvidenceIndex(
        model_name=args.model_name,
        device="cuda" if sys.platform.startswith("linux") else "cpu",
        chunk_config=chunk_config,
    )

    # 2. Load Evaluation Samples
    logger.info("Loading test samples from %s...", args.test_file)
    samples = []
    with open(args.test_file) as f:
        for idx, line in enumerate(f):
            if args.sample_limit and idx >= args.sample_limit:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info("Loaded %d test samples", len(samples))

    # 3. Chunk and Index Source Contexts for these samples (Strict Source Isolation)
    logger.info("Chunking and indexing source contexts for evaluation samples...")
    unique_sources = {}
    for s in samples:
        sid = s["source_id"]
        if sid not in unique_sources:
            unique_sources[sid] = s["reference_context"]

    indexing_start = time.perf_counter()
    all_chunks = []
    for sid, ctx_text in unique_sources.items():
        chunks = chunker.chunk_document(source_id=sid, text=ctx_text)
        all_chunks.extend(chunks)

    index.add_chunks(all_chunks, batch_size=32)
    indexing_time = time.perf_counter() - indexing_start
    logger.info("Indexed %d chunks from %d source contexts in %.2fs", len(all_chunks), len(unique_sources), indexing_time)

    # 4. Initialize Claim Extractor & Evidence Retriever
    claim_extractor = AtomicClaimExtractor(provider=RuleBasedExtractorProvider())
    retriever = EvidenceRetriever(index=index, default_top_k=args.top_k)

    # 5. Execute End-to-End Claim-to-Evidence Pipeline
    total_claims = 0
    total_factual_claims = 0
    retrieval_times_ms = []
    evaluation_records = []

    pipeline_start = time.perf_counter()

    for s in samples:
        sample_id = s["sample_id"]
        source_id = s["source_id"]
        response_text = s["generated_response"]

        # Step A: Extract Atomic Claims
        extract_res = claim_extractor.extract_claims(response_text, sample_id=sample_id)
        factual_claims = extract_res.get_factual_claims()
        total_claims += extract_res.total_claims
        total_factual_claims += len(factual_claims)

        # Step B: Retrieve Top-K Evidence for each factual claim (Scoped to source_id)
        sample_claim_results = []
        for claim in factual_claims:
            c_start = time.perf_counter()
            ret_res = retriever.retrieve_for_claim(
                claim=claim,
                source_id=source_id,  # Strictly source-scoped: prevents evaluation leakage!
                top_k=args.top_k,
            )
            c_latency = (time.perf_counter() - c_start) * 1000.0
            retrieval_times_ms.append(c_latency)
            sample_claim_results.append(ret_res.to_contract_dict())

        evaluation_records.append({
            "sample_id": sample_id,
            "source_id": source_id,
            "task_type": s.get("task_type", "unknown"),
            "model": s.get("metadata", {}).get("model", "unknown"),
            "response": response_text,
            "total_extracted_claims": extract_res.total_claims,
            "factual_claims_count": len(factual_claims),
            "excluded_claims_count": extract_res.excluded_claims_count,
            "claims_evidence": sample_claim_results,
        })

    pipeline_time = time.perf_counter() - pipeline_start
    avg_retrieval_ms = sum(retrieval_times_ms) / len(retrieval_times_ms) if retrieval_times_ms else 0.0

    # 6. Save Full Evaluation Report
    report = {
        "embedding_model": index.model_name,
        "embedding_dimension": index.embedding_dim,
        "device": str(index._device),
        "chunk_configuration": {
            "chunk_size": chunk_config.chunk_size,
            "chunk_overlap": chunk_config.chunk_overlap,
            "method": chunk_config.method,
        },
        "top_k": args.top_k,
        "evaluation_summary": {
            "total_samples": len(samples),
            "unique_source_documents": len(unique_sources),
            "total_chunks_indexed": len(all_chunks),
            "total_claims_extracted": total_claims,
            "factual_claims_evaluated": total_factual_claims,
            "indexing_time_seconds": round(indexing_time, 2),
            "average_retrieval_latency_ms": round(avg_retrieval_ms, 2),
            "pipeline_throughput_claims_per_sec": round(total_factual_claims / pipeline_time, 2) if pipeline_time > 0 else 0.0,
        },
        "inspection_samples": evaluation_records[:5],
    }

    report_path = args.output_dir / "retrieval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 7. Print Inspection Summary
    logger.info("=" * 65)
    logger.info("PHASE 4 EVIDENCE RETRIEVAL REPORT")
    logger.info("=" * 65)
    logger.info("Embedding Model: %s (dim=%d)", index.model_name, index.embedding_dim)
    logger.info("Evaluated Samples: %d | Source Contexts: %d", len(samples), len(unique_sources))
    logger.info("Total Chunks Indexed: %d (Chunk Size: %d, Overlap: %d)", len(all_chunks), chunk_config.chunk_size, chunk_config.chunk_overlap)
    logger.info("Factual Claims Queried: %d | Top-K: %d", total_factual_claims, args.top_k)
    logger.info("Indexing Time: %.2fs | Avg Retrieval Latency: %.2f ms / claim", indexing_time, avg_retrieval_ms)
    logger.info("-" * 65)
    logger.info("QUALITATIVE INSPECTION EXAMPLES:")
    for item in evaluation_records[:2]:
        logger.info("[Sample %s | Source %s | Task: %s]", item["sample_id"], item["source_id"], item["task_type"])
        for ce in item["claims_evidence"][:2]:
            logger.info("  Claim %s: %s", ce["claim_id"], item["response"][:80] + "...")
            for ev in ce["evidence"][:2]:
                logger.info("    -> Rank %d [Score: %.4f | Chunk: %s]: %s", ev["rank"], ev["similarity_score"], ev["chunk_id"], ev["text"][:120] + "...")
    logger.info("=" * 65)
    logger.info("Saved full retrieval report to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
