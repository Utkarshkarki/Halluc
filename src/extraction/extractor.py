"""Atomic factual claim extractor engine.

Decomposes text responses into independent atomic claims, classifies them,
verifies span offsets, implements bounded retry/repair on malformed structured outputs,
and preserves provenance.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from pydantic import ValidationError

from src.data.schema import NormalizedSample
from src.extraction.cache import ClaimExtractionCache
from src.extraction.prompt import (
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
)
from src.extraction.providers.base import BaseLLMProvider
from src.extraction.schema import (
    ClaimExtractionPayload,
    ClaimExtractionResult,
    ClaimType,
    ExtractedClaim,
)

logger = logging.getLogger(__name__)


class AtomicClaimExtractor:
    """Orchestrates atomic claim extraction with caching, validation, and bounded retry."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        cache: ClaimExtractionCache | None = None,
        max_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.max_retries = max_retries

        logger.info(
            "AtomicClaimExtractor initialized with provider=%s (model=%s), max_retries=%d",
            self.provider.name,
            self.provider.model,
            self.max_retries,
        )

    def extract_claims(
        self,
        response_text: str,
        sample_id: str | None = None,
    ) -> ClaimExtractionResult:
        """Decompose response_text into validated atomic claims.

        Args:
            response_text: The LLM output text to decompose.
            sample_id: Optional ID of the dataset sample for provenance.

        Returns:
            ClaimExtractionResult with claims, classifications, and metrics.
        """
        text = response_text.strip() if response_text else ""

        # Handle empty/whitespace input immediately
        if not text:
            return ClaimExtractionResult(
                claims=[],
                raw_response=response_text or "",
                sample_id=sample_id,
                extractor_model=f"{self.provider.name}:{self.provider.model}",
                prompt_version=EXTRACTION_PROMPT_VERSION,
                latency_ms=0.0,
                cached=False,
                total_claims=0,
                factual_claims_count=0,
                excluded_claims_count=0,
            )

        # Check Cache
        if self.cache is not None:
            cached_result = self.cache.get(
                response_text=text,
                model_name=f"{self.provider.name}:{self.provider.model}",
                prompt_version=EXTRACTION_PROMPT_VERSION,
            )
            if cached_result is not None:
                cached_result.sample_id = sample_id
                return cached_result

        # Execute extraction with bounded retry/repair
        user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(response_text=text)
        current_prompt = user_prompt
        last_error = ""
        payload: ClaimExtractionPayload | None = None
        validation_errors: list[str] = []

        start_time = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                raw_dict = self.provider.generate_structured(
                    prompt=current_prompt,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    schema=ClaimExtractionPayload,
                )
                payload = ClaimExtractionPayload.model_validate(raw_dict)
                break  # Successful generation and validation
            except (ValidationError, Exception) as e:
                last_error = str(e)
                validation_errors.append(f"Attempt {attempt + 1} failed: {last_error}")
                logger.warning(
                    "Extraction attempt %d failed for model %s: %s",
                    attempt + 1,
                    self.provider.model,
                    last_error,
                )
                if attempt < self.max_retries:
                    # Provide feedback for repair
                    current_prompt = (
                        f"{user_prompt}\n\n"
                        f"[REPAIR INSTRUCTION]: Your previous output was invalid JSON or failed schema validation with error: {last_error}. "
                        "Please re-output strictly conforming to the requested JSON schema."
                    )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        raw_claims = payload.claims if payload is not None else []

        # Validate span offsets against source text
        validated_claims: list[ExtractedClaim] = []
        for c in raw_claims:
            s_start = c.source_start
            s_end = c.source_end

            # Verify offsets if provided
            if s_start is not None and s_end is not None:
                if 0 <= s_start < s_end <= len(text):
                    sub = text[s_start:s_end].strip()
                    # If substring does not reasonably match claim text, nullify to prevent fabrication
                    if sub.lower() not in c.text.lower() and c.text.lower().rstrip(".") not in sub.lower():
                        s_start, s_end = None, None
                else:
                    s_start, s_end = None, None

            validated_claims.append(
                ExtractedClaim(
                    claim_id=c.claim_id,
                    text=c.text.strip(),
                    claim_type=c.claim_type,
                    source_start=s_start,
                    source_end=s_end,
                    context_tags=c.context_tags,
                    confidence=c.confidence,
                )
            )

        factual_count = sum(1 for c in validated_claims if c.is_verifiable())
        excluded_count = len(validated_claims) - factual_count

        result = ClaimExtractionResult(
            claims=validated_claims,
            raw_response=text,
            sample_id=sample_id,
            extractor_model=f"{self.provider.name}:{self.provider.model}",
            prompt_version=EXTRACTION_PROMPT_VERSION,
            latency_ms=round(duration_ms, 2),
            cached=False,
            total_claims=len(validated_claims),
            factual_claims_count=factual_count,
            excluded_claims_count=excluded_count,
            validation_errors=validation_errors,
        )

        # Store in Cache if valid and cache enabled
        if self.cache is not None and not validation_errors:
            self.cache.set(
                response_text=text,
                model_name=f"{self.provider.name}:{self.provider.model}",
                prompt_version=EXTRACTION_PROMPT_VERSION,
                result=result,
            )

        return result

    def extract_from_sample(self, sample: NormalizedSample) -> ClaimExtractionResult:
        """Decompose a Phase 1 NormalizedSample into claims."""
        return self.extract_claims(
            response_text=sample.generated_response,
            sample_id=sample.sample_id,
        )

    def extract_batch(
        self, samples: Sequence[NormalizedSample]
    ) -> list[ClaimExtractionResult]:
        """Extract atomic claims for a batch of NormalizedSample objects."""
        return [self.extract_from_sample(s) for s in samples]
