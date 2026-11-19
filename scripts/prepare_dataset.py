#!/usr/bin/env python3
"""Dataset preparation pipeline.

1. Loads raw RAGTruth data from data/raw/
2. Merges responses with source contexts
3. Validates schema, IDs, responses, and span offsets
4. Saves normalized samples and isolates invalid records
5. Computes and saves true dataset statistics
6. Creates deterministic group-based splits and preserves official split
7. Performs and logs leakage verification
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import RAGTruthLoader
from src.data.preprocessing import (
    compute_stats,
    merge_responses_and_sources,
    validate_dataset,
)
from src.data.splitter import (
    check_leakage,
    create_stratified_split,
    preserve_official_split,
    save_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process raw RAGTruth data into normalized, validated, leak-free splits"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed data output directory (default: data/processed)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Path to splits output directory (default: data/splits)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for splitting (default: 42)",
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    splits_dir: Path = args.splits_dir
    seed: int = args.seed

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw RAGTruth data from %s...", raw_dir)
    try:
        loader = RAGTruthLoader.from_local(raw_dir)
    except FileNotFoundError as e:
        logger.error(
            "Raw data not found! Please run 'python scripts/download_dataset.py' first: %s",
            e,
        )
        return 1

    responses = loader.get_responses()
    sources = loader.get_sources()
    logger.info(
        "Loaded %d raw responses and %d raw sources", len(responses), len(sources)
    )

    # 1. Merge responses with source contexts
    logger.info("Merging responses and sources...")
    samples = merge_responses_and_sources(responses, sources)

    # 2. Validation
    logger.info("Validating dataset and span offsets...")
    valid_samples, invalid_records = validate_dataset(samples)

    logger.info(
        "Validation complete: %d valid samples, %d invalid records",
        len(valid_samples),
        len(invalid_records),
    )

    # 3. Save invalid records separately for investigation
    invalid_file = processed_dir / "invalid_records.jsonl"
    with open(invalid_file, "w") as f:
        for sample, result in invalid_records:
            f.write(
                json.dumps(
                    {
                        "sample": sample.model_dump(),
                        "errors": result.errors,
                        "warnings": result.warnings,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("Saved invalid records to %s", invalid_file)

    # 4. Save normalized valid samples
    normalized_file = processed_dir / "normalized_samples.jsonl"
    with open(normalized_file, "w") as f:
        for s in valid_samples:
            f.write(s.model_dump_json() + "\n")
    logger.info("Saved normalized samples to %s", normalized_file)

    # 5. Compute and save true dataset statistics
    stats = compute_stats(valid_samples, invalid_count=len(invalid_records))
    stats_file = processed_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        f.write(stats.model_dump_json(indent=2))
    logger.info("Saved dataset statistics to %s", stats_file)

    # 6. Splits
    # A) Official RAGTruth split
    official_dir = splits_dir / "official"
    official_splits = preserve_official_split(valid_samples)
    save_splits(official_splits, official_dir, seed=None)
    official_leakage = check_leakage(official_splits, group_key="source_id")

    # B) Deterministic Group-based 70/15/15 split (Train/Calibration/Test)
    group_dir = splits_dir / "group_split"
    group_splits = create_stratified_split(
        valid_samples,
        ratios={"train": 0.70, "calibration": 0.15, "test": 0.15},
        seed=seed,
        group_key="source_id",
    )
    save_splits(group_splits, group_dir, seed=seed)
    group_leakage = check_leakage(group_splits, group_key="source_id")

    # Summary Report
    logger.info("=" * 60)
    logger.info("DATASET PREPARATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total responses loaded: %d", stats.total_responses + stats.invalid_records)
    logger.info("Valid responses: %d", stats.total_responses)
    logger.info("Hallucinated responses: %d", stats.hallucinated_responses)
    logger.info("Non-hallucinated responses: %d", stats.non_hallucinated_responses)
    logger.info("Annotated spans: %d", stats.annotated_spans)
    logger.info("Invalid records: %d", stats.invalid_records)
    logger.info("Avg response length: %.2f chars", stats.avg_response_length)
    logger.info("Task distribution: %s", stats.task_distribution)
    logger.info("Official split counts: %s", official_leakage["split_sample_counts"])
    logger.info(
        "Official split leakage: %s",
        "LEAK DETECTED" if official_leakage["has_leakage"] else "NO LEAKAGE",
    )
    logger.info("Group split counts: %s", group_leakage["split_sample_counts"])
    logger.info(
        "Group split leakage: %s",
        "LEAK DETECTED" if group_leakage["has_leakage"] else "NO LEAKAGE",
    )
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
