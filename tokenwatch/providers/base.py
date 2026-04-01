"""
Abstract base class for LLM provider wrappers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from ..tracker import CostTracker


class BaseProvider(ABC):
    """
    Abstract base for provider-specific wrappers.

    Each subclass knows how to:
    1. Intercept API calls for a specific provider SDK.
    2. Extract token usage from the response.
    3. Report usage back to the CostTracker.
    """

    def __init__(self, tracker: "CostTracker") -> None:
        self.tracker = tracker

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the canonical provider name (e.g., 'openai')."""
        ...

    @abstractmethod
    def extract_usage(self, response: Any) -> Tuple[int, int]:
        """
        Extract token counts from a provider response object.

        Returns:
            (input_tokens, output_tokens) as a tuple of ints.
        """
        ...

    def _record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Convenience: report usage to the tracker."""
        self.tracker._record_call(
            provider=self.get_provider_name(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
