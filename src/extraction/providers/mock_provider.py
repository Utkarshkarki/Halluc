"""Mock LLM provider for unit tests and bounded error simulation."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel

from src.extraction.providers.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Configurable mock provider that records calls and can simulate errors."""

    def __init__(
        self,
        canned_response: dict[str, Any] | None = None,
        fail_times: int = 0,
        model_name: str = "mock-gpt-4",
    ) -> None:
        self._canned_response = canned_response or {"claims": []}
        self._fail_times = fail_times
        self._calls_count = 0
        self._model_name = model_name
        self.call_history: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def calls_count(self) -> int:
        return self._calls_count

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        schema: type[BaseModel],
    ) -> dict[str, Any]:
        """Return canned response or raise exception for testing retry logic."""
        self._calls_count += 1
        self.call_history.append({"prompt": prompt, "system_prompt": system_prompt})

        if self._calls_count <= self._fail_times:
            raise ValueError(f"Simulated LLM failure (attempt {self._calls_count})")

        return dict(self._canned_response)
