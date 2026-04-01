"""
Tests for the pricing module.
"""

from __future__ import annotations

import pytest

from tokenwatch.pricing.tables import (
    PRICING,
    add_custom_model,
    get_cost,
    get_price_per_1m,
    list_models,
)


class TestGetCost:
    # --- OpenAI ---
    def test_openai_gpt4o(self):
        cost = get_cost("openai", "gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.50 + 10.00)

    def test_openai_gpt4o_mini(self):
        cost = get_cost("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.15 + 0.60)

    def test_openai_gpt4_turbo(self):
        cost = get_cost("openai", "gpt-4-turbo", 1_000_000, 1_000_000)
        assert cost == pytest.approx(10.00 + 30.00)

    def test_openai_gpt35_turbo(self):
        cost = get_cost("openai", "gpt-3.5-turbo", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.50 + 1.50)

    # --- Anthropic ---
    def test_anthropic_claude_opus(self):
        cost = get_cost("anthropic", "claude-opus-4-6", 1_000_000, 1_000_000)
        assert cost == pytest.approx(15.00 + 75.00)

    def test_anthropic_claude_sonnet(self):
        cost = get_cost("anthropic", "claude-sonnet-4-6", 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.00 + 15.00)

    def test_anthropic_claude_haiku(self):
        cost = get_cost("anthropic", "claude-haiku-4-5", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.80 + 4.00)

    def test_anthropic_claude_35_sonnet(self):
        cost = get_cost("anthropic", "claude-3-5-sonnet-20241022", 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.00 + 15.00)

    def test_anthropic_claude_35_haiku(self):
        cost = get_cost("anthropic", "claude-3-5-haiku-20241022", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.80 + 4.00)

    # --- Gemini ---
    def test_gemini_15_pro(self):
        cost = get_cost("gemini", "gemini-1.5-pro", 1_000_000, 1_000_000)
        assert cost == pytest.approx(1.25 + 5.00)

    def test_gemini_15_flash(self):
        cost = get_cost("gemini", "gemini-1.5-flash", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.075 + 0.30)

    def test_gemini_20_flash(self):
        cost = get_cost("gemini", "gemini-2.0-flash", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.10 + 0.40)

    def test_gemini_10_pro(self):
        cost = get_cost("gemini", "gemini-1.0-pro", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.50 + 1.50)

    # --- Edge cases ---
    def test_zero_tokens(self):
        cost = get_cost("openai", "gpt-4o", 0, 0)
        assert cost == 0.0

    def test_only_input_tokens(self):
        cost = get_cost("openai", "gpt-4o", 1_000_000, 0)
        assert cost == pytest.approx(2.50)

    def test_only_output_tokens(self):
        cost = get_cost("openai", "gpt-4o", 0, 1_000_000)
        assert cost == pytest.approx(10.00)

    def test_partial_million_tokens(self):
        cost = get_cost("openai", "gpt-4o-mini", 500_000, 250_000)
        expected = (500_000 / 1_000_000) * 0.15 + (250_000 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_cost("unknown_provider", "gpt-4o", 100, 50)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_cost("openai", "gpt-99-nonexistent-model-xyz", 100, 50)

    def test_case_insensitive_provider(self):
        cost1 = get_cost("openai", "gpt-4o", 1_000, 500)
        cost2 = get_cost("OpenAI", "gpt-4o", 1_000, 500)
        assert cost1 == pytest.approx(cost2)

    def test_model_alias_resolution(self):
        # "gpt-3.5-turbo-0125" is an alias for "gpt-3.5-turbo"
        aliased_cost = get_cost("openai", "gpt-3.5-turbo-0125", 1_000_000, 1_000_000)
        direct_cost = get_cost("openai", "gpt-3.5-turbo", 1_000_000, 1_000_000)
        assert aliased_cost == pytest.approx(direct_cost)


class TestGetPricePer1M:
    def test_returns_tuple(self):
        inp, out = get_price_per_1m("openai", "gpt-4o")
        assert inp == pytest.approx(2.50)
        assert out == pytest.approx(10.00)

    def test_all_models_have_positive_prices(self):
        for provider, models in PRICING.items():
            for model, (inp, out) in models.items():
                assert inp >= 0, f"{provider}/{model} has negative input price"
                assert out >= 0, f"{provider}/{model} has negative output price"
                # Output is typically >= input (or at least positive)
                assert out > 0, f"{provider}/{model} has zero output price"


class TestListModels:
    def test_returns_all_providers(self):
        all_models = list_models()
        assert "openai" in all_models
        assert "anthropic" in all_models
        assert "gemini" in all_models

    def test_filter_by_provider(self):
        openai_models = list_models("openai")
        assert "openai" in openai_models
        assert "anthropic" not in openai_models

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            list_models("unknown_xyz")

    def test_openai_has_expected_models(self):
        models = list_models("openai")["openai"]
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gpt-4-turbo" in models
        assert "gpt-3.5-turbo" in models

    def test_anthropic_has_expected_models(self):
        models = list_models("anthropic")["anthropic"]
        assert "claude-opus-4-6" in models
        assert "claude-sonnet-4-6" in models
        assert "claude-haiku-4-5" in models
        assert "claude-3-5-sonnet-20241022" in models
        assert "claude-3-5-haiku-20241022" in models

    def test_gemini_has_expected_models(self):
        models = list_models("gemini")["gemini"]
        assert "gemini-1.5-pro" in models
        assert "gemini-1.5-flash" in models
        assert "gemini-2.0-flash" in models
        assert "gemini-1.0-pro" in models


class TestAddCustomModel:
    def test_add_and_use_custom_model(self):
        add_custom_model("openai", "gpt-test-custom", 1.00, 2.00)
        cost = get_cost("openai", "gpt-test-custom", 1_000_000, 1_000_000)
        assert cost == pytest.approx(1.00 + 2.00)

    def test_add_new_provider(self):
        add_custom_model("custom_provider", "my-model", 0.50, 1.00)
        cost = get_cost("custom_provider", "my-model", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.50 + 1.00)
        # Clean up
        del PRICING["custom_provider"]

    def test_overwrite_existing_model(self):
        original_inp, original_out = get_price_per_1m("openai", "gpt-4o")
        add_custom_model("openai", "gpt-4o", 99.00, 199.00)
        cost = get_cost("openai", "gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(99.00 + 199.00)
        # Restore original pricing
        PRICING["openai"]["gpt-4o"] = (original_inp, original_out)


class TestPricingConsistency:
    """Verify pricing table structural consistency."""

    def test_all_prices_are_floats(self):
        for provider, models in PRICING.items():
            for model, (inp, out) in models.items():
                assert isinstance(inp, (int, float)), f"{provider}/{model} input not numeric"
                assert isinstance(out, (int, float)), f"{provider}/{model} output not numeric"

    def test_output_price_generally_higher_than_input(self):
        """Output tokens are typically more expensive than input."""
        for provider, models in PRICING.items():
            for model, (inp, out) in models.items():
                assert out >= inp, (
                    f"{provider}/{model}: output price ${out} < input price ${inp}"
                )

    def test_cost_proportional_to_tokens(self):
        """Doubling tokens should double cost."""
        cost_1k = get_cost("openai", "gpt-4o", 1_000, 500)
        cost_2k = get_cost("openai", "gpt-4o", 2_000, 1_000)
        assert cost_2k == pytest.approx(cost_1k * 2)
