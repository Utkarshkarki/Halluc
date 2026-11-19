"""Deterministic caching for claim extraction results.

Avoids duplicate LLM calls by hashing (response_text, extractor_model, prompt_version).
Cache entries are serialized to disk in JSON format and loaded transparently.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.extraction.schema import ClaimExtractionResult

logger = logging.getLogger(__name__)


class ClaimExtractionCache:
    """Persistent on-disk cache for extracted claims.

    Keyed deterministically by SHA-256 of:
        f"{model_name}:{prompt_version}:{response_text}"
    """

    def __init__(self, cache_dir: Path | str = Path("data/cache/extraction")) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, ClaimExtractionResult] = {}

    @staticmethod
    def compute_cache_key(response_text: str, model_name: str, prompt_version: str) -> str:
        """Compute deterministic SHA-256 cache key."""
        payload = f"{model_name.strip()}::{prompt_version.strip()}::{response_text.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self, response_text: str, model_name: str, prompt_version: str
    ) -> ClaimExtractionResult | None:
        """Retrieve cached extraction result if present, else None."""
        key = self.compute_cache_key(response_text, model_name, prompt_version)

        # Check in-memory tier first
        if key in self._memory_cache:
            res = self._memory_cache[key].model_copy(deep=True)
            res.cached = True
            return res

        # Check filesystem tier
        file_path = self.cache_dir / f"{key}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                result = ClaimExtractionResult.model_validate(data)
                result.cached = True
                self._memory_cache[key] = result
                logger.debug("Cache hit for key: %s (model=%s)", key[:12], model_name)
                return result
            except Exception as e:
                logger.warning("Failed to load cache entry %s: %s", file_path, e)

        return None

    def set(
        self,
        response_text: str,
        model_name: str,
        prompt_version: str,
        result: ClaimExtractionResult,
    ) -> None:
        """Store extraction result in disk and memory cache."""
        key = self.compute_cache_key(response_text, model_name, prompt_version)
        self._memory_cache[key] = result

        file_path = self.cache_dir / f"{key}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            logger.debug("Cached extraction result to %s", file_path)
        except Exception as e:
            logger.warning("Failed to persist cache entry to %s: %s", file_path, e)

    def clear(self) -> None:
        """Clear memory and disk cache."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass
