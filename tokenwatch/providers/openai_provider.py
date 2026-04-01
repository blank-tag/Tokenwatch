"""
OpenAI provider wrapper.

Transparently intercepts client.chat.completions.create() (sync and async)
to track token usage and cost without changing the return value.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Tuple

from .base import BaseProvider

if TYPE_CHECKING:
    from ..tracker import CostTracker


class OpenAIProvider(BaseProvider):
    """Wraps an OpenAI client to track cost automatically."""

    def get_provider_name(self) -> str:
        return "openai"

    def extract_usage(self, response: Any) -> Tuple[int, int]:
        """Extract usage from an OpenAI ChatCompletion response."""
        try:
            usage = response.usage
            if usage is None:
                return 0, 0
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            return int(input_tokens), int(output_tokens)
        except AttributeError:
            return 0, 0

    def wrap(self, client: Any) -> Any:
        """
        Return the client with cost tracking injected into
        ``client.chat.completions.create`` (sync and async).

        The original client object is modified in-place and returned,
        so existing code needs no changes.
        """
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            ) from exc

        provider = self

        # ----- sync wrapper -----
        original_create = client.chat.completions.create

        @functools.wraps(original_create)
        def _sync_create(*args: Any, **kwargs: Any) -> Any:
            response = original_create(*args, **kwargs)
            # Extract model from kwargs or response
            model = kwargs.get("model") or (args[0] if args else None)
            if model is None:
                try:
                    model = response.model
                except AttributeError:
                    model = "unknown"
            input_tokens, output_tokens = provider.extract_usage(response)
            provider._record(model, input_tokens, output_tokens)
            return response

        # ----- async wrapper -----
        original_acreate = getattr(client.chat.completions, "acreate", None)

        async def _async_create(*args: Any, **kwargs: Any) -> Any:
            # OpenAI v1+ uses the same `create` for async via `await`
            # We patch acreate for compatibility
            if original_acreate is not None:
                response = await original_acreate(*args, **kwargs)
            else:
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

        # Patch the methods
        client.chat.completions.create = _sync_create
        if original_acreate is not None:
            client.chat.completions.acreate = _async_create

        # Also patch AsyncOpenAI if available
        _patch_async_client(client, provider)

        return client


def _patch_async_client(client: Any, provider: OpenAIProvider) -> None:
    """
    If this is an AsyncOpenAI client, patch the async create method.
    The regular `create` on async clients is already a coroutine function.
    """
    # Check if this is an async client by inspecting the type name
    client_type = type(client).__name__
    if "Async" not in client_type:
        return

    import asyncio
    import inspect

    original_create = client.chat.completions.create

    if not inspect.iscoroutinefunction(original_create):
        return

    @functools.wraps(original_create)
    async def _async_patched(*args: Any, **kwargs: Any) -> Any:
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

    client.chat.completions.create = _async_patched
