#!/usr/bin/env python3
"""Build and persist an EvidenceIndex over RAGTruth reference documents.

Chunking and indexing strictly process raw reference texts (source_info.jsonl).
Annotations, labels, and model responses are never indexed to prevent data leakage.
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

from src.retrieval.chunker import DeterministicChunker
from src.retrieval.index import EvidenceIndex
from src.retrieval.schema import ChunkConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_evidence_index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and persist FAISS evidence index")
    parser.add_argument(
        "--source-file",
        type=Path,
        default=Path("data/raw/source_info.jsonl"),
        help="Path to source_info.jsonl (default: data/raw/source_info.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indices/ragtruth_evidence"),
        help="Directory to save FAISS index and metadata (default: data/indices/ragtruth_evidence)",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("configs/retrieval_config.json"),
        help="Path to retrieval configuration JSON (default: configs/retrieval_config.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of source documents to index",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda, cpu, or auto)",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if args.config_file.exists():
        with open(args.config_file) as f:
            cfg = json.load(f)

    model_name = cfg.get("model_name", "BAAI/bge-small-en-v1.5")
    device = args.device or cfg.get("device", "auto")
    chunk_size = cfg.get("chunk_size", 300)
    chunk_overlap = cfg.get("chunk_overlap", 60)

    chunk_config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunker = DeterministicChunker(chunk_config)

    logger.info("Initializing EvidenceIndex with model=%s on %s...", model_name, device)
    index = EvidenceIndex(
        model_name=model_name,
        device=device,
        chunk_config=chunk_config,
    )

    # Load source contexts
    if not args.source_file.exists():
        logger.error("Source file %s not found! Run scripts/download_dataset.py first.", args.source_file)
        return 1

    logger.info("Loading source documents from %s...", args.source_file)
    sources = []
    with open(args.source_file) as f:
        for idx, line in enumerate(f):
            if args.limit and idx >= args.limit:
                break
            line = line.strip()
            if line:
                sources.append(json.loads(line))

    logger.info("Loaded %d source documents for indexing", len(sources))

    # Chunk documents
    all_chunks = []
    chunk_start_time = time.perf_counter()
    for s in sources:
        sid = s["source_id"]
        # Handle source_info string vs dict
        raw_info = s.get("source_info", "")
        text = raw_info if isinstance(raw_info, str) else json.dumps(raw_info, ensure_ascii=False)
        metadata = {
            "task_type": s.get("task_type", "unknown"),
            "source_benchmark": s.get("source", "unknown"),
        }
        chunks = chunker.chunk_document(source_id=sid, text=text, metadata=metadata)
        all_chunks.extend(chunks)

    chunk_time = time.perf_counter() - chunk_start_time
    logger.info("Chunking completed: %d chunks generated in %.2fs", len(all_chunks), chunk_time)

    # Build vector index
    index_start_time = time.perf_counter()
    logger.info("Embedding and indexing %d chunks with batch_size=64...", len(all_chunks))
    index.add_chunks(all_chunks, batch_size=64, show_progress=True)
    index_time = time.perf_counter() - index_start_time

    logger.info("Indexing completed in %.2fs (%.2f chunks/sec)", index_time, len(all_chunks) / index_time if index_time > 0 else 0)

    # Persist index
    logger.info("Saving index to %s...", args.output_dir)
    index.save(args.output_dir)

    # Compute index size
    index_size_bytes = sum(f.stat().st_size for f in args.output_dir.glob("*") if f.is_file())
    index_size_mb = index_size_bytes / (1024 * 1024)

    logger.info("=" * 65)
    logger.info("EVIDENCE INDEX BUILD SUMMARY")
    logger.info("=" * 65)
    logger.info("Embedding Model: %s (dim=%d)", model_name, index.embedding_dim)
    logger.info("Source Documents: %d", len(sources))
    logger.info("Total Chunks: %d (avg %.1f chunks/doc)", len(all_chunks), len(all_chunks) / len(sources) if sources else 0)
    logger.info("Chunk Configuration: size=%d chars, overlap=%d chars", chunk_size, chunk_overlap)
    logger.info("Chunking Time: %.2fs", chunk_time)
    logger.info("Embedding & Indexing Time: %.2fs", index_time)
    logger.info("Persisted Directory: %s", args.output_dir.resolve())
    logger.info("Total Index Size: %.2f MB", index_size_mb)
    logger.info("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
