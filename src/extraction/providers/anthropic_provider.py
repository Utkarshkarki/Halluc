"""Anthropic Claude LLM provider for structured atomic claim extraction."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from pydantic import BaseModel

from src.extraction.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """LLM provider using Anthropic Claude models with structured tool calling."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self._timeout = timeout

        if not self._api_key:
            logger.warning("AnthropicProvider initialized without ANTHROPIC_API_KEY")

        # Client is initialized lazily to avoid connection attempts during dry runs
        self._client: Any = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required for AnthropicProvider")
            import anthropic

            kwargs: dict[str, Any] = {"api_key": self._api_key, "timeout": self._timeout}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        schema: type[BaseModel],
    ) -> dict[str, Any]:
        """Generate structured JSON adhering to the given schema using Claude tool use."""
        client = self._get_client()

        # Build tool definition from Pydantic JSON schema
        json_schema = schema.model_json_schema()
        tool_definition = {
            "name": "record_extracted_claims",
            "description": "Record the decomposed atomic claims and their categories.",
            "input_schema": json_schema,
        }

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool_definition],
                tool_choice={"type": "tool", "name": "record_extracted_claims"},
            )

            # Extract tool use block
            for block in response.content:
                if block.type == "tool_use" and block.name == "record_extracted_claims":
                    return dict(block.input)

            # Fallback: parse raw text as json
            text_resp = response.content[0].text if response.content else "{}"
            return json.loads(text_resp)

        except Exception as e:
            # Never include raw API key in logs
            logger.error("Anthropic API generation failed for model %s: %s", self._model, str(e))
            raise
