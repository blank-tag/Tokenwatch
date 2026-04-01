"""
Anthropic provider wrapper.

Transparently intercepts client.messages.create() (sync and async)
to track token usage and cost without changing the return value.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Tuple

from .base import BaseProvider

if TYPE_CHECKING:
    from ..tracker import CostTracker


class AnthropicProvider(BaseProvider):
    """Wraps an Anthropic client to track cost automatically."""

    def get_provider_name(self) -> str:
        return "anthropic"

    def extract_usage(self, response: Any) -> Tuple[int, int]:
        """Extract usage from an Anthropic Messages response."""
        try:
            usage = response.usage
            if usage is None:
                return 0, 0
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            return int(input_tokens), int(output_tokens)
        except AttributeError:
            return 0, 0

    def wrap(self, client: Any) -> Any:
        """
        Return the client with cost tracking injected into
        ``client.messages.create`` (sync and async).

        The original client object is modified in-place and returned.
        """
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required. Install it with: pip install anthropic"
            ) from exc

        provider = self
        original_create = client.messages.create

        if inspect.iscoroutinefunction(original_create):
            # AsyncAnthropic client
            @functools.wraps(original_create)
            async def _patched(*args: Any, **kwargs: Any) -> Any:
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

            client.messages.create = _patched
        else:
            # Sync Anthropic client
            @functools.wraps(original_create)
            def _sync_patched(*args: Any, **kwargs: Any) -> Any:
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

            client.messages.create = _sync_patched

            # Also patch async variant if present
            async_create = getattr(client.messages, "acreate", None)
            if async_create is not None and inspect.iscoroutinefunction(async_create):
                @functools.wraps(async_create)
                async def _async_patched(*args: Any, **kwargs: Any) -> Any:
                    response = await async_create(*args, **kwargs)
                    model = kwargs.get("model") or (args[0] if args else None)
                    if model is None:
                        try:
                            model = response.model
                        except AttributeError:
                            model = "unknown"
                    input_tokens, output_tokens = provider.extract_usage(response)
                    provider._record(model, input_tokens, output_tokens)
                    return response

                client.messages.acreate = _async_patched

        return client
