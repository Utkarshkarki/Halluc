"""Base contract for LLM providers used in claim extraction.

Ensures vendor decoupling: allows plugging in Anthropic, OpenAI, local models,
or deterministic test providers seamlessly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel


class BaseLLMProvider(ABC):
    """Abstract interface for LLM extraction providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name, e.g. 'anthropic', 'openai', 'rule_based'."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier, e.g. 'claude-3-5-sonnet', 'rule-based-v1'."""
        ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        schema: type[BaseModel],
    ) -> dict[str, Any]:
        """Generate structured JSON adhering to the given Pydantic schema.

        Args:
            prompt: User prompt content.
            system_prompt: System prompt with decomposition guidelines.
            schema: Target Pydantic model for validation.

        Returns:
            Dictionary matching the target schema.

        Raises:
            Exception: If provider generation fails or response is malformed.
        """
        ...
