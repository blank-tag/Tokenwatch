"""
Groq provider wrapper.

Groq uses the same chat.completions.create() interface as OpenAI,
so this provider mirrors OpenAIProvider with provider_name="groq".

Supports:
  - groq.Groq (sync client)
  - groq.AsyncGroq (async client)
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Tuple

from .base import BaseProvider

if TYPE_CHECKING:
    from ..tracker import CostTracker


class GroqProvider(BaseProvider):
    """Wraps a Groq client to track cost automatically."""

    def get_provider_name(self) -> str:
        return "groq"

    def extract_usage(self, response: Any) -> Tuple[int, int]:
        """Extract usage from a Groq ChatCompletion response."""
        try:
            usage = response.usage
            if usage is None:
                return 0, 0
            # Groq mirrors OpenAI: prompt_tokens / completion_tokens
            input_tokens  = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            return int(input_tokens), int(output_tokens)
        except AttributeError:
            return 0, 0

    def wrap(self, client: Any) -> Any:
        """
        Return the Groq client with cost tracking injected into
        ``client.chat.completions.create`` (sync and async).

        The original client object is modified in-place and returned.
        """
        try:
            import groq  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "groq package is required. Install it with: pip install groq"
            ) from exc

        provider = self
        original_create = client.chat.completions.create

        # ── async client (groq.AsyncGroq) ──────────────────────────────
        if inspect.iscoroutinefunction(original_create):
            @functools.wraps(original_create)
            async def _async_create(*args: Any, **kwargs: Any) -> Any:
                response = await original_create(*args, **kwargs)
                model = kwargs.get("model") or (args[0] if args else None)
                if model is None:
                    try:
                        model = response.model
                    except AttributeError:
                        model = "unknown"
                input_tokens, output_tokens = provider.extract_usage(response)
                provider._record(model, input_tokens, output_tokens)
                return response

            client.chat.completions.create = _async_create

        # ── sync client (groq.Groq) ────────────────────────────────────
        else:
            @functools.wraps(original_create)
            def _sync_create(*args: Any, **kwargs: Any) -> Any:
                response = original_create(*args, **kwargs)
                model = kwargs.get("model") or (args[0] if args else None)
                if model is None:
                    try:
                        model = response.model
                    except AttributeError:
                        model = "unknown"
                input_tokens, output_tokens = provider.extract_usage(response)
                provider._record(model, input_tokens, output_tokens)
                return response

            client.chat.completions.create = _sync_create

        return client
