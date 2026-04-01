"""
Google Gemini provider wrapper.

Intercepts model.generate_content() and client.models.generate_content()
to track token usage and cost without changing the return value.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Optional, Tuple

from .base import BaseProvider

if TYPE_CHECKING:
    from ..tracker import CostTracker


class GeminiProvider(BaseProvider):
    """Wraps a Google Generative AI client or model to track cost automatically."""

    def get_provider_name(self) -> str:
        return "gemini"

    def extract_usage(self, response: Any) -> Tuple[int, int]:
        """Extract usage from a Gemini GenerateContentResponse."""
        try:
            meta = response.usage_metadata
            if meta is None:
                return 0, 0
            # prompt_token_count covers input; candidates_token_count covers output
            input_tokens = (
                getattr(meta, "prompt_token_count", None)
                or getattr(meta, "input_token_count", None)
                or 0
            )
            output_tokens = (
                getattr(meta, "candidates_token_count", None)
                or getattr(meta, "output_token_count", None)
                or 0
            )
            return int(input_tokens), int(output_tokens)
        except AttributeError:
            return 0, 0

    def _extract_model_name(self, model_obj_or_name: Any) -> str:
        """
        Extract a canonical model name string.
        model_obj_or_name may be a string like "gemini-1.5-pro" or an SDK
        GenerativeModel object.
        """
        if isinstance(model_obj_or_name, str):
            # Strip "models/" prefix if present
            return model_obj_or_name.replace("models/", "")
        # SDK GenerativeModel objects expose ._model_name or .model_name
        for attr in ("_model_name", "model_name", "name"):
            val = getattr(model_obj_or_name, attr, None)
            if val and isinstance(val, str):
                return val.replace("models/", "")
        return "gemini-unknown"

    def wrap(self, client_or_model: Any) -> Any:
        """
        Wrap a Gemini client or GenerativeModel.

        Handles two patterns:
        1. ``genai.GenerativeModel`` – patches ``generate_content`` directly.
        2. ``google.generativeai`` module or a ``genai.Client`` – patches
           ``client.models.generate_content``.

        The original object is modified in-place and returned.
        """
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            try:
                import google.genai  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "google-generativeai package is required. "
                    "Install it with: pip install google-generativeai"
                ) from exc

        provider = self
        obj_type = type(client_or_model).__name__

        # Pattern 1: GenerativeModel object
        if obj_type == "GenerativeModel" or hasattr(client_or_model, "generate_content"):
            model_name = self._extract_model_name(client_or_model)
            original_gc = client_or_model.generate_content

            if inspect.iscoroutinefunction(original_gc):
                @functools.wraps(original_gc)
                async def _async_gc(*args: Any, **kwargs: Any) -> Any:
                    response = await original_gc(*args, **kwargs)
                    input_t, output_t = provider.extract_usage(response)
                    provider._record(model_name, input_t, output_t)
                    return response

                client_or_model.generate_content = _async_gc
            else:
                @functools.wraps(original_gc)
                def _sync_gc(*args: Any, **kwargs: Any) -> Any:
                    response = original_gc(*args, **kwargs)
                    input_t, output_t = provider.extract_usage(response)
                    provider._record(model_name, input_t, output_t)
                    return response

                client_or_model.generate_content = _sync_gc

            # Also wrap generate_content_async if present
            async_gc = getattr(client_or_model, "generate_content_async", None)
            if async_gc is not None and inspect.iscoroutinefunction(async_gc):
                @functools.wraps(async_gc)
                async def _async_gc2(*args: Any, **kwargs: Any) -> Any:
                    response = await async_gc(*args, **kwargs)
                    input_t, output_t = provider.extract_usage(response)
                    provider._record(model_name, input_t, output_t)
                    return response

                client_or_model.generate_content_async = _async_gc2

        # Pattern 2: Module-level or client with .models.generate_content
        elif hasattr(client_or_model, "models") and hasattr(
            client_or_model.models, "generate_content"
        ):
            original_gc = client_or_model.models.generate_content

            if inspect.iscoroutinefunction(original_gc):
                @functools.wraps(original_gc)
                async def _async_client_gc(*args: Any, **kwargs: Any) -> Any:
                    response = await original_gc(*args, **kwargs)
                    model_name = provider._extract_model_name(
                        kwargs.get("model", args[0] if args else "gemini-unknown")
                    )
                    input_t, output_t = provider.extract_usage(response)
                    provider._record(model_name, input_t, output_t)
                    return response

                client_or_model.models.generate_content = _async_client_gc
            else:
                @functools.wraps(original_gc)
                def _sync_client_gc(*args: Any, **kwargs: Any) -> Any:
                    response = original_gc(*args, **kwargs)
                    model_name = provider._extract_model_name(
                        kwargs.get("model", args[0] if args else "gemini-unknown")
                    )
                    input_t, output_t = provider.extract_usage(response)
                    provider._record(model_name, input_t, output_t)
                    return response

                client_or_model.models.generate_content = _sync_client_gc
        else:
            raise ValueError(
                "Unsupported Gemini object. Pass a GenerativeModel instance or a genai client."
            )

        return client_or_model
