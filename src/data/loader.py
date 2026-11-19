"""RAGTruth dataset loader.

Supports three acquisition strategies:
1. HuggingFace Hub (wandb/RAGTruth-processed) — includes response data only.
2. GitHub raw files (ParticleMedia/RAGTruth) — includes both response.jsonl
   and source_info.jsonl with full context.
3. Local JSONL files — for offline / cached data.

The loader preserves raw data unchanged and tracks provenance metadata
(source URL, revision, record counts).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

from src.data.schema import RawResponseRecord, RawSourceRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUGGINGFACE_DATASET = "wandb/RAGTruth-processed"
GITHUB_BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
GITHUB_RESPONSE_URL = f"{GITHUB_BASE}/response.jsonl"
GITHUB_SOURCE_URL = f"{GITHUB_BASE}/source_info.jsonl"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DatasetAcquisitionError(Exception):
    """Raised when dataset cannot be acquired from any source."""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class RAGTruthLoader:
    """Loads and stores raw RAGTruth data from various sources.

    Usage::

        # From HuggingFace
        loader = RAGTruthLoader.from_huggingface()

        # From GitHub (recommended — includes source info)
        loader = RAGTruthLoader.from_github_raw()

        # From local files
        loader = RAGTruthLoader.from_local(Path("data/raw"))

        responses = loader.get_responses()
        sources = loader.get_sources()
        loader.save_raw(Path("data/raw"))
    """

    def __init__(
        self,
        responses: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        dataset_info: dict[str, Any],
    ) -> None:
        self._raw_responses = responses
        self._raw_sources = sources
        self._dataset_info = dataset_info
        logger.info(
            "RAGTruthLoader initialised: %d responses, %d sources",
            len(responses),
            len(sources),
        )

    # ----- Factory: HuggingFace -------------------------------------------

    @classmethod
    def from_huggingface(
        cls, cache_dir: Path | None = None
    ) -> RAGTruthLoader:
        """Load from wandb/RAGTruth-processed on HuggingFace Hub.

        Note: this dataset contains only response records. Source info must
        be loaded separately (e.g. from GitHub) for full context.

        Raises:
            DatasetAcquisitionError: If the download fails.
        """
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]

            logger.info("Downloading %s from HuggingFace Hub …", HUGGINGFACE_DATASET)
            kwargs: dict[str, Any] = {}
            if cache_dir is not None:
                kwargs["cache_dir"] = str(cache_dir)

            ds = load_dataset(HUGGINGFACE_DATASET, **kwargs)

            responses: list[dict[str, Any]] = []
            for split_name in ds:
                for row in ds[split_name]:
                    record = dict(row)
                    # Ensure split field is populated
                    if "split" not in record or not record["split"]:
                        record["split"] = split_name
                    responses.append(record)

            logger.info("Loaded %d response records from HuggingFace", len(responses))

            # Attempt to also fetch source info from GitHub
            sources: list[dict[str, Any]] = []
            try:
                sources = cls._download_jsonl(GITHUB_SOURCE_URL)
                logger.info(
                    "Loaded %d source records from GitHub", len(sources)
                )
            except Exception as e:
                logger.warning(
                    "Could not load source_info from GitHub: %s", e
                )

            revision = "unknown"
            try:
                from datasets import get_dataset_config_info  # type: ignore[import-untyped]
            except ImportError:
                pass

            info = {
                "source": "huggingface",
                "dataset_name": HUGGINGFACE_DATASET,
                "revision": revision,
                "response_count": len(responses),
                "source_count": len(sources),
            }
            return cls(responses, sources, info)

        except Exception as exc:
            msg = (
                f"DATASET_ACQUISITION_BLOCKED: Failed to load {HUGGINGFACE_DATASET} "
                f"from HuggingFace Hub: {exc}"
            )
            logger.error(msg)
            raise DatasetAcquisitionError(msg) from exc

    # ----- Factory: GitHub raw files --------------------------------------

    @classmethod
    def from_github_raw(cls, cache_dir: Path | None = None) -> RAGTruthLoader:
        """Download response.jsonl and source_info.jsonl from GitHub.

        This is the recommended method as it includes full source context.

        Raises:
            DatasetAcquisitionError: If the download fails.
        """
        try:
            logger.info("Downloading RAGTruth from GitHub …")
            responses = cls._download_jsonl(GITHUB_RESPONSE_URL)
            sources = cls._download_jsonl(GITHUB_SOURCE_URL)
            logger.info(
                "Downloaded %d responses, %d sources from GitHub",
                len(responses),
                len(sources),
            )

            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cls._save_jsonl(responses, cache_dir / "response.jsonl")
                cls._save_jsonl(sources, cache_dir / "source_info.jsonl")
                logger.info("Cached raw data to %s", cache_dir)

            info = {
                "source": "github",
                "repo": "ParticleMedia/RAGTruth",
                "branch": "main",
                "response_url": GITHUB_RESPONSE_URL,
                "source_url": GITHUB_SOURCE_URL,
                "revision": "main",
                "response_count": len(responses),
                "source_count": len(sources),
            }
            return cls(responses, sources, info)

        except Exception as exc:
            msg = (
                f"DATASET_ACQUISITION_BLOCKED: Failed to download RAGTruth "
                f"from GitHub: {exc}"
            )
            logger.error(msg)
            raise DatasetAcquisitionError(msg) from exc

    # ----- Factory: local files -------------------------------------------

    @classmethod
    def from_local(cls, data_dir: Path) -> RAGTruthLoader:
        """Load from local JSONL files.

        Expected structure::

            data_dir/
                response.jsonl
                source_info.jsonl   (optional)

        Raises:
            FileNotFoundError: If response.jsonl is missing.
        """
        response_path = data_dir / "response.jsonl"
        source_path = data_dir / "source_info.jsonl"

        if not response_path.exists():
            raise FileNotFoundError(
                f"response.jsonl not found in {data_dir}. "
                "Run the download script first."
            )

        responses = cls._load_jsonl(response_path)
        sources = cls._load_jsonl(source_path) if source_path.exists() else []

        logger.info(
            "Loaded %d responses, %d sources from %s",
            len(responses),
            len(sources),
            data_dir,
        )

        info = {
            "source": "local",
            "data_dir": str(data_dir),
            "response_count": len(responses),
            "source_count": len(sources),
        }
        return cls(responses, sources, info)

    # ----- Public interface -----------------------------------------------

    def get_responses(self) -> list[RawResponseRecord]:
        """Return parsed response records."""
        records: list[RawResponseRecord] = []
        for raw in self._raw_responses:
            try:
                records.append(RawResponseRecord(**raw))
            except Exception as e:
                logger.warning("Skipping malformed response record: %s", e)
        return records

    def get_sources(self) -> list[RawSourceRecord]:
        """Return parsed source-info records."""
        records: list[RawSourceRecord] = []
        for raw in self._raw_sources:
            try:
                records.append(RawSourceRecord(**raw))
            except Exception as e:
                logger.warning("Skipping malformed source record: %s", e)
        return records

    def get_dataset_info(self) -> dict[str, Any]:
        """Return provenance metadata."""
        return dict(self._dataset_info)

    def save_raw(self, output_dir: Path) -> None:
        """Persist raw data to JSONL files, preserving original format."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_jsonl(self._raw_responses, output_dir / "response.jsonl")
        if self._raw_sources:
            self._save_jsonl(self._raw_sources, output_dir / "source_info.jsonl")

        # Save provenance
        info_path = output_dir / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(self._dataset_info, f, indent=2)

        logger.info("Saved raw data to %s", output_dir)

    # ----- Helpers --------------------------------------------------------

    @staticmethod
    def _download_jsonl(url: str) -> list[dict[str, Any]]:
        """Download a JSONL file from a URL using streaming."""
        logger.info("Downloading %s ...", url)
        records: list[dict[str, Any]] = []
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    line_str = line.strip()
                    if line_str:
                        records.append(json.loads(line_str))
        logger.info("Downloaded %d records from %s", len(records), url)
        return records

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Load a local JSONL file."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
        """Write records to a JSONL file."""
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
