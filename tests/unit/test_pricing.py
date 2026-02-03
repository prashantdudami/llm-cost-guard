"""
Unit tests for pricing module.
"""

import pytest

from llm_cost_guard.pricing.loader import PricingLoader, get_pricing
from llm_cost_guard.exceptions import PricingNotFoundError
from llm_cost_guard.models import ModelType


class TestPricingLoader:
    """Tests for PricingLoader class."""

    def test_loader_creation(self):
        """Test pricing loader creation."""
        loader = PricingLoader()
        assert loader is not None
        assert loader.last_updated is not None

    def test_get_openai_pricing(self):
        """Test getting OpenAI model pricing."""
        loader = PricingLoader()

        pricing = loader.get_pricing("openai", "gpt-4o")
        assert pricing is not None
        assert pricing.input_cost_per_1k > 0
        assert pricing.output_cost_per_1k > 0

    def test_get_anthropic_pricing(self):
        """Test getting Anthropic model pricing."""
        loader = PricingLoader()

        pricing = loader.get_pricing("anthropic", "claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing.input_cost_per_1k > 0
        assert pricing.output_cost_per_1k > 0

    def test_get_bedrock_pricing(self):
        """Test getting Bedrock model pricing."""
        loader = PricingLoader()

        pricing = loader.get_pricing("bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert pricing is not None
        assert pricing.input_cost_per_1k > 0

    def test_pricing_not_found(self):
        """Test error for unknown model."""
        loader = PricingLoader()

        with pytest.raises(PricingNotFoundError) as exc_info:
            loader.get_pricing("openai", "nonexistent-model-xyz")

        assert "nonexistent-model-xyz" in str(exc_info.value)

    def test_calculate_cost(self):
        """Test cost calculation."""
        loader = PricingLoader()

        input_cost, output_cost, total_cost = loader.calculate_cost(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        assert input_cost > 0
        assert output_cost > 0
        assert total_cost == input_cost + output_cost

    def test_calculate_cost_with_cache(self):
        """Test cost calculation with cached tokens."""
        loader = PricingLoader()

        # Without cache
        _, _, cost_no_cache = loader.calculate_cost(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        # With cache (some tokens cached)
        _, _, cost_with_cache = loader.calculate_cost(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
        )

        # Cached should be cheaper or equal
        assert cost_with_cache <= cost_no_cache

    def test_estimate_cost(self):
        """Test cost estimation for budget reservation."""
        loader = PricingLoader()

        estimated = loader.estimate_cost(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            max_output_tokens=4096,
        )

        assert estimated > 0

    def test_pricing_overrides(self):
        """Test custom pricing overrides."""
        loader = PricingLoader(
            pricing_overrides={
                "openai/gpt-4": {
                    "input_cost_per_1k": 0.01,
                    "output_cost_per_1k": 0.02,
                }
            }
        )

        pricing = loader.get_pricing("openai", "gpt-4")
        assert pricing.input_cost_per_1k == 0.01
        assert pricing.output_cost_per_1k == 0.02

    def test_get_all_models(self):
        """Test getting all models."""
        loader = PricingLoader()

        all_models = loader.get_all_models()
        assert "openai" in all_models
        assert "anthropic" in all_models
        assert len(all_models["openai"]) > 0

    def test_get_models_by_provider(self):
        """Test getting models for specific provider."""
        loader = PricingLoader()

        models = loader.get_all_models("openai")
        assert "openai" in models
        assert len(models) == 1

    def test_pricing_freshness(self):
        """Test pricing freshness checks."""
        loader = PricingLoader()

        # Just loaded, should not be stale
        assert loader.is_stale is False
        assert loader.is_very_stale is False

    def test_refresh(self):
        """Test pricing refresh."""
        loader = PricingLoader()

        old_updated = loader.last_updated
        loader.refresh()

        # Should have new timestamp
        assert loader.last_updated >= old_updated


class TestGlobalPricing:
    """Tests for global pricing functions."""

    def test_get_pricing_function(self):
        """Test global get_pricing function."""
        pricing = get_pricing("openai", "gpt-4o")
        assert pricing is not None
        assert pricing.input_cost_per_1k > 0


class TestEmbeddingPricing:
    """Tests for embedding model pricing."""

    def test_openai_embedding_pricing(self):
        """Test OpenAI embedding model pricing."""
        loader = PricingLoader()

        pricing = loader.get_pricing("openai", "text-embedding-3-small")
        assert pricing is not None
        assert pricing.model_type == ModelType.EMBEDDING
        assert pricing.output_cost_per_1k == 0  # Embeddings don't have output tokens

    def test_embedding_cost_calculation(self):
        """Test embedding cost calculation."""
        loader = PricingLoader()

        input_cost, output_cost, total_cost = loader.calculate_cost(
            provider="openai",
            model="text-embedding-3-small",
            input_tokens=1000,
            output_tokens=0,
        )

        assert input_cost > 0
        assert output_cost == 0
        assert total_cost == input_cost
