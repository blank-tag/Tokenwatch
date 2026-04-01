"""
Pricing tables for LLM providers.
Prices are in USD per 1,000,000 tokens (input/output).
Last updated: early 2025.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

# Structure: provider -> model -> (input_cost_per_1m, output_cost_per_1m)
PRICING: Dict[str, Dict[str, Tuple[float, float]]] = {
    "openai": {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-3.5-turbo": (0.50, 1.50),
    },
    "anthropic": {
        "claude-opus-4-6": (15.00, 75.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5": (0.80, 4.00),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-3-5-haiku-20241022": (0.80, 4.00),
    },
    "gemini": {
        "gemini-1.5-pro": (1.25, 5.00),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-1.0-pro": (0.50, 1.50),
    },
}

# Model aliases for common variations
MODEL_ALIASES: Dict[str, Tuple[str, str]] = {
    # OpenAI
    "gpt-4o-2024-08-06": ("openai", "gpt-4o"),
    "gpt-4o-2024-05-13": ("openai", "gpt-4o"),
    "gpt-4o-mini-2024-07-18": ("openai", "gpt-4o-mini"),
    "gpt-4-turbo-preview": ("openai", "gpt-4-turbo"),
    "gpt-4-turbo-2024-04-09": ("openai", "gpt-4-turbo"),
    "gpt-3.5-turbo-0125": ("openai", "gpt-3.5-turbo"),
    "gpt-3.5-turbo-1106": ("openai", "gpt-3.5-turbo"),
    # Anthropic
    "claude-3-opus-20240229": ("anthropic", "claude-opus-4-6"),
    "claude-3-sonnet-20240229": ("anthropic", "claude-sonnet-4-6"),
    "claude-3-haiku-20240307": ("anthropic", "claude-haiku-4-5"),
    # Gemini
    "gemini-pro": ("gemini", "gemini-1.0-pro"),
    "gemini-1.5-pro-latest": ("gemini", "gemini-1.5-pro"),
    "gemini-1.5-flash-latest": ("gemini", "gemini-1.5-flash"),
    "gemini-2.0-flash-exp": ("gemini", "gemini-2.0-flash"),
}


def get_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate the cost of an LLM API call.

    Args:
        provider: Provider name ("openai", "anthropic", "gemini")
        model: Model name
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens

    Returns:
        Cost in USD as a float

    Raises:
        ValueError: If provider or model is not found in pricing tables
    """
    provider = provider.lower().strip()
    model = model.lower().strip()

    # Check for alias first
    if model in MODEL_ALIASES:
        aliased_provider, aliased_model = MODEL_ALIASES[model]
        if provider == aliased_provider or provider not in PRICING:
            provider = aliased_provider
            model = aliased_model

    if provider not in PRICING:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Supported providers: {list(PRICING.keys())}"
        )

    provider_pricing = PRICING[provider]

    if model not in provider_pricing:
        # Try partial match
        matches = [m for m in provider_pricing if m.startswith(model) or model.startswith(m.split("-")[0])]
        if len(matches) == 1:
            model = matches[0]
        else:
            raise ValueError(
                f"Unknown model: '{model}' for provider '{provider}'. "
                f"Supported models: {list(provider_pricing.keys())}"
            )

    input_price_per_1m, output_price_per_1m = provider_pricing[model]

    input_cost = (input_tokens / 1_000_000) * input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * output_price_per_1m

    return input_cost + output_cost


def get_price_per_1m(provider: str, model: str) -> Tuple[float, float]:
    """
    Get pricing per 1M tokens for a given provider/model.

    Returns:
        Tuple of (input_price_per_1m, output_price_per_1m)
    """
    provider = provider.lower().strip()
    model = model.lower().strip()

    if model in MODEL_ALIASES:
        provider, model = MODEL_ALIASES[model]

    if provider not in PRICING:
        raise ValueError(f"Unknown provider: '{provider}'")

    if model not in PRICING[provider]:
        raise ValueError(f"Unknown model: '{model}' for provider '{provider}'")

    return PRICING[provider][model]


def list_models(provider: Optional[str] = None) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    List all supported models and their pricing.

    Args:
        provider: If specified, return only models for that provider.

    Returns:
        Dict mapping provider -> model -> (input_price_per_1m, output_price_per_1m)
    """
    if provider is not None:
        provider = provider.lower().strip()
        if provider not in PRICING:
            raise ValueError(
                f"Unknown provider: '{provider}'. "
                f"Supported providers: {list(PRICING.keys())}"
            )
        return {provider: PRICING[provider]}
    return dict(PRICING)


def add_custom_model(
    provider: str,
    model: str,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> None:
    """
    Add a custom model to the pricing tables.

    Args:
        provider: Provider name
        model: Model identifier
        input_price_per_1m: Input token price per 1M tokens in USD
        output_price_per_1m: Output token price per 1M tokens in USD
    """
    provider = provider.lower().strip()
    if provider not in PRICING:
        PRICING[provider] = {}
    PRICING[provider][model.lower().strip()] = (input_price_per_1m, output_price_per_1m)
