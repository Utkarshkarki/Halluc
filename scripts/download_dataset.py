#!/usr/bin/env python3
"""Script to download the real RAGTruth dataset.

Saves raw response and source-info records to data/raw/ without modifications.
Never fabricates mock data if acquisition fails.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import (
    DatasetAcquisitionError,
    RAGTruthLoader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download real RAGTruth dataset into data/raw/"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for raw dataset files (default: data/raw)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "github", "huggingface"],
        default="auto",
        help="Acquisition source: 'auto' (tries GitHub then HuggingFace), 'github', or 'huggingface'",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional HuggingFace cache directory",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loader: RAGTruthLoader | None = None
    errors: list[str] = []

    # Strategy 1: GitHub Raw
    if args.source in ("auto", "github"):
        try:
            logger.info("Attempting acquisition from official GitHub repository...")
            loader = RAGTruthLoader.from_github_raw(cache_dir=output_dir)
        except Exception as e:
            msg = f"GitHub acquisition failed: {e}"
            logger.warning(msg)
            errors.append(msg)

    # Strategy 2: HuggingFace
    if loader is None and args.source in ("auto", "huggingface"):
        try:
            logger.info("Attempting acquisition from HuggingFace Hub (wandb/RAGTruth-processed)...")
            loader = RAGTruthLoader.from_huggingface(cache_dir=args.cache_dir)
        except Exception as e:
            msg = f"HuggingFace acquisition failed: {e}"
            logger.warning(msg)
            errors.append(msg)

    if loader is None:
        logger.error("DATASET_ACQUISITION_BLOCKED")
        logger.error(
            "Could not automatically download RAGTruth dataset. Errors encountered:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )
        logger.error(
            "To resolve manually:\n"
            "1. Obtain response.jsonl and source_info.jsonl from ParticleMedia/RAGTruth\n"
            f"2. Place them into: {output_dir.resolve()}/\n"
            "3. Run scripts/prepare_dataset.py"
        )
        return 1

    # Save raw records unchanged
    loader.save_raw(output_dir)
    info = loader.get_dataset_info()

    logger.info("=" * 60)
    logger.info("RAGTruth Dataset Acquisition Successful")
    logger.info("Source: %s", info.get("source"))
    logger.info("Responses loaded: %d", info.get("response_count", 0))
    logger.info("Sources loaded: %d", info.get("source_count", 0))
    logger.info("Saved to: %s", output_dir.resolve())
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
