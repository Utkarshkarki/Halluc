"""Deterministic dataset splitting with group-based leakage prevention.

RAGTruth contains 6 LLM responses per source document. When splitting into
train/val/test, all responses sharing the same source_id must stay in the
same partition to prevent data leakage.

This module provides:
- Official split preservation (RAGTruth's own train/test)
- Group-based stratified splitting with deterministic seeds
- Leakage detection
- Split persistence and loading
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from src.data.schema import NormalizedSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Official split preservation
# ---------------------------------------------------------------------------

def preserve_official_split(
    samples: list[NormalizedSample],
) -> dict[str, list[NormalizedSample]]:
    """Partition samples using RAGTruth's official train/test split.

    Each sample has a ``split`` field set to 'train' or 'test'.

    Returns:
        Dict mapping split name to its samples.

    Raises:
        ValueError: If any sample has an unexpected split value.
    """
    splits: dict[str, list[NormalizedSample]] = {}
    unexpected: set[str] = set()

    for s in samples:
        split_name = s.split.lower().strip()
        if split_name not in ("train", "test"):
            unexpected.add(split_name)
        splits.setdefault(split_name, []).append(s)

    if unexpected:
        logger.warning("Unexpected split values found: %s", unexpected)

    # Validate no overlap
    if "train" in splits and "test" in splits:
        train_ids = {s.sample_id for s in splits["train"]}
        test_ids = {s.sample_id for s in splits["test"]}
        overlap = train_ids & test_ids
        if overlap:
            logger.error(
                "Official split has %d overlapping sample_ids!", len(overlap)
            )
        else:
            logger.info("Official split verified: no sample_id overlap")

    for name, split_samples in splits.items():
        logger.info("Official split '%s': %d samples", name, len(split_samples))

    return splits


# ---------------------------------------------------------------------------
# Group-based splitting
# ---------------------------------------------------------------------------

def create_stratified_split(
    samples: list[NormalizedSample],
    ratios: dict[str, float],
    seed: int = 42,
    group_key: str = "source_id",
) -> dict[str, list[NormalizedSample]]:
    """Split samples into partitions using group-based assignment.

    All samples sharing the same group key (default: source_id) are placed
    in the same partition. This prevents leakage when multiple LLM responses
    were generated from the same reference context.

    The algorithm:
    1. Collect unique groups.
    2. Shuffle deterministically.
    3. Assign groups to partitions by cumulative ratio thresholds.

    Args:
        samples: Input samples.
        ratios: Mapping of split name to target fraction,
                e.g. ``{'train': 0.7, 'val': 0.15, 'test': 0.15}``.
        seed: Random seed for deterministic shuffling.
        group_key: Attribute to group by (must be a field on NormalizedSample).

    Returns:
        Dict mapping split name to its samples.

    Raises:
        ValueError: If ratios don't sum to ~1.0 or no samples are provided.
    """
    if not samples:
        raise ValueError("Cannot split empty sample list")

    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(
            f"Ratios must sum to ~1.0, got {total_ratio:.4f}: {ratios}"
        )

    # Group samples by key
    groups: dict[str, list[NormalizedSample]] = {}
    for s in samples:
        key = getattr(s, group_key)
        groups.setdefault(key, []).append(s)

    # Deterministic shuffle of group keys
    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    # Assign groups to partitions by cumulative thresholds
    split_names = list(ratios.keys())
    cumulative = []
    running = 0.0
    for name in split_names:
        running += ratios[name]
        cumulative.append(running)

    result: dict[str, list[NormalizedSample]] = {name: [] for name in split_names}
    n_groups = len(group_ids)

    for idx, gid in enumerate(group_ids):
        frac = (idx + 1) / n_groups
        # Find which partition this fraction falls into
        assigned = split_names[-1]  # default to last
        for i, threshold in enumerate(cumulative):
            if frac <= threshold + 1e-9:
                assigned = split_names[i]
                break
        result[assigned].extend(groups[gid])

    for name, split_samples in result.items():
        logger.info(
            "Split '%s': %d samples (target %.1f%%, actual %.1f%%)",
            name,
            len(split_samples),
            ratios[name] * 100,
            len(split_samples) / len(samples) * 100 if samples else 0,
        )

    return result


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

def check_leakage(
    splits: dict[str, list[NormalizedSample]],
    group_key: str = "source_id",
) -> dict[str, Any]:
    """Check whether any group appears in multiple splits.

    Args:
        splits: Dict of split name to sample list.
        group_key: Attribute to check for leakage.

    Returns:
        Dict with:
        - has_leakage: bool
        - leaked_groups: list of group IDs in multiple splits
        - split_group_counts: dict of split -> unique group count
        - split_sample_counts: dict of split -> sample count
    """
    split_groups: dict[str, set[str]] = {}
    for name, samples in splits.items():
        split_groups[name] = {getattr(s, group_key) for s in samples}

    # Find groups appearing in more than one split
    leaked: set[str] = set()
    split_names = list(split_groups.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            overlap = split_groups[split_names[i]] & split_groups[split_names[j]]
            leaked.update(overlap)

    result = {
        "has_leakage": len(leaked) > 0,
        "leaked_groups": sorted(leaked),
        "split_group_counts": {
            name: len(grps) for name, grps in split_groups.items()
        },
        "split_sample_counts": {
            name: len(samples) for name, samples in splits.items()
        },
    }

    if leaked:
        logger.error(
            "LEAKAGE DETECTED: %d groups appear in multiple splits: %s",
            len(leaked),
            sorted(leaked)[:10],
        )
    else:
        logger.info("Leakage check passed: no group overlap across splits")

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_splits(
    splits: dict[str, list[NormalizedSample]],
    output_dir: Path,
    seed: int | None = None,
) -> None:
    """Save splits to JSONL files with metadata.

    Creates::

        output_dir/
            train.jsonl
            test.jsonl
            (etc.)
            split_metadata.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "seed": seed,
        "splits": {},
    }

    for name, samples in splits.items():
        path = output_dir / f"{name}.jsonl"
        content_lines: list[str] = []
        for s in samples:
            line = s.model_dump_json()
            content_lines.append(line)

        with open(path, "w") as f:
            f.write("\n".join(content_lines) + "\n")

        # Compute checksum
        raw_bytes = "\n".join(content_lines).encode("utf-8")
        checksum = hashlib.sha256(raw_bytes).hexdigest()

        metadata["splits"][name] = {
            "count": len(samples),
            "file": path.name,
            "sha256": checksum,
        }
        logger.info("Saved split '%s' (%d samples) to %s", name, len(samples), path)

    meta_path = output_dir / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved split metadata to %s", meta_path)


def load_splits(splits_dir: Path) -> dict[str, list[NormalizedSample]]:
    """Load previously saved splits.

    Args:
        splits_dir: Directory containing split JSONL files and metadata.

    Returns:
        Dict mapping split name to list of NormalizedSample.
    """
    meta_path = splits_dir / "split_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"split_metadata.json not found in {splits_dir}")

    with open(meta_path) as f:
        metadata = json.load(f)

    result: dict[str, list[NormalizedSample]] = {}
    for name, info in metadata["splits"].items():
        path = splits_dir / info["file"]
        samples: list[NormalizedSample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(NormalizedSample.model_validate_json(line))
        result[name] = samples
        logger.info("Loaded split '%s': %d samples from %s", name, len(samples), path)

    return result
