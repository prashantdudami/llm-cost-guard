"""
Pricing data loader for LLM Cost Guard.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import yaml

from llm_cost_guard.models import ModelPricing, ModelType
from llm_cost_guard.exceptions import PricingNotFoundError

logger = logging.getLogger(__name__)

# Default pricing data directory
PRICING_DIR = Path(__file__).parent


class PricingLoader:
    """Loads and manages pricing data for LLM providers."""

    def __init__(
        self,
        pricing_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        pricing_stale_warning_days: int = 7,
        pricing_stale_error_days: int = 30,
        bedrock_region: str = "us-east-1",
    ):
        self._pricing_data: Dict[str, Dict[str, ModelPricing]] = {}
        self._pricing_versions: Dict[str, str] = {}
        self._pricing_overrides = pricing_overrides or {}
        self._stale_warning_days = pricing_stale_warning_days
        self._stale_error_days = pricing_stale_error_days
        self._bedrock_region = bedrock_region
        self._last_loaded: Optional[datetime] = None

        self._load_all_pricing()

    def _load_all_pricing(self) -> None:
        """Load pricing from all YAML files."""
        for yaml_file in PRICING_DIR.glob("*.yaml"):
            provider = yaml_file.stem
            self._load_provider_pricing(provider, yaml_file)

        self._last_loaded = datetime.now()

    def _load_provider_pricing(self, provider: str, yaml_path: Path) -> None:
        """Load pricing for a specific provider."""
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                return

            self._pricing_versions[provider] = data.get("version", "unknown")
            self._pricing_data[provider] = {}

            models = data.get("models", {})
            for model_name, model_data in models.items():
                model_type_str = model_data.get("model_type", "chat")
                model_type = ModelType(model_type_str) if model_type_str else ModelType.CHAT

                pricing = ModelPricing(
                    input_cost_per_1k=model_data.get("input_cost_per_1k", 0.0),
                    output_cost_per_1k=model_data.get("output_cost_per_1k", 0.0),
                    cached_input_cost_per_1k=model_data.get("cached_input_cost_per_1k"),
                    context_window=model_data.get("context_window", 128000),
                    model_type=model_type,
                    image_cost_per_image=model_data.get("image_cost_per_image"),
                    audio_cost_per_minute=model_data.get("audio_cost_per_minute"),
                    embedding_dimensions=model_data.get("embedding_dimensions"),
                )
                self._pricing_data[provider][model_name] = pricing

        except Exception as e:
            logger.warning(f"Failed to load pricing for {provider}: {e}")

    def get_pricing(self, provider: str, model: str) -> ModelPricing:
        """Get pricing for a specific model."""
        # Normalize provider and model names
        provider = provider.lower()
        model_lower = model.lower()

        # Check overrides first
        override_key = f"{provider}/{model}"
        if override_key in self._pricing_overrides:
            override = self._pricing_overrides[override_key]
            return ModelPricing(
                input_cost_per_1k=override.get("input_cost_per_1k", 0.0),
                output_cost_per_1k=override.get("output_cost_per_1k", 0.0),
                cached_input_cost_per_1k=override.get("cached_input_cost_per_1k"),
                context_window=override.get("context_window", 128000),
            )

        # Check loaded pricing
        if provider in self._pricing_data:
            provider_pricing = self._pricing_data[provider]

            # Try exact match
            if model in provider_pricing:
                return provider_pricing[model]

            # Try lowercase match
            if model_lower in provider_pricing:
                return provider_pricing[model_lower]

            # Try prefix match (for versioned models like gpt-4-0613)
            for known_model in provider_pricing:
                if model_lower.startswith(known_model) or known_model.startswith(model_lower):
                    return provider_pricing[known_model]

        raise PricingNotFoundError(
            f"Pricing not found for {provider}/{model}", provider=provider, model=model
        )

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> tuple[float, float, float]:
        """
        Calculate cost for a call.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4o", "claude-3-sonnet")
            input_tokens: Number of input tokens (must be >= 0)
            output_tokens: Number of output tokens (must be >= 0)
            cached_tokens: Number of cached input tokens (must be >= 0)
            
        Returns:
            Tuple of (input_cost, output_cost, total_cost)
            
        Raises:
            ValueError: If any token count is negative
            PricingNotFoundError: If pricing not found for model
        """
        # Validate inputs
        if input_tokens < 0:
            raise ValueError(f"input_tokens must be >= 0, got {input_tokens}")
        if output_tokens < 0:
            raise ValueError(f"output_tokens must be >= 0, got {output_tokens}")
        if cached_tokens < 0:
            raise ValueError(f"cached_tokens must be >= 0, got {cached_tokens}")
        if cached_tokens > input_tokens:
            logger.warning(
                f"cached_tokens ({cached_tokens}) > input_tokens ({input_tokens}), "
                "capping to input_tokens"
            )
            cached_tokens = input_tokens
            
        pricing = self.get_pricing(provider, model)

        # Calculate input cost (considering cache)
        regular_input_tokens = max(0, input_tokens - cached_tokens)
        input_cost = (regular_input_tokens / 1000) * pricing.input_cost_per_1k

        # Add cached token cost if applicable
        if cached_tokens > 0 and pricing.cached_input_cost_per_1k is not None:
            input_cost += (cached_tokens / 1000) * pricing.cached_input_cost_per_1k

        # Calculate output cost
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k

        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def estimate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        max_output_tokens: int = 4096,
    ) -> float:
        """Estimate maximum cost for a call (for budget reservation)."""
        pricing = self.get_pricing(provider, model)

        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (max_output_tokens / 1000) * pricing.output_cost_per_1k

        return input_cost + output_cost

    @property
    def last_updated(self) -> Optional[datetime]:
        """Get when pricing was last loaded."""
        return self._last_loaded

    @property
    def pricing_version(self) -> Dict[str, str]:
        """Get pricing versions for all providers."""
        return dict(self._pricing_versions)

    @property
    def is_stale(self) -> bool:
        """Check if pricing data is stale (beyond warning threshold)."""
        if self._last_loaded is None:
            return True

        age_days = (datetime.now() - self._last_loaded).days
        return age_days >= self._stale_warning_days

    @property
    def is_very_stale(self) -> bool:
        """Check if pricing data is very stale (beyond error threshold)."""
        if self._last_loaded is None:
            return True

        age_days = (datetime.now() - self._last_loaded).days
        return age_days >= self._stale_error_days

    def get_all_models(self, provider: Optional[str] = None) -> Dict[str, list[str]]:
        """Get all known models, optionally filtered by provider."""
        if provider:
            return {provider: list(self._pricing_data.get(provider, {}).keys())}
        return {p: list(models.keys()) for p, models in self._pricing_data.items()}

    def refresh(self) -> None:
        """Reload pricing data from files."""
        self._pricing_data.clear()
        self._pricing_versions.clear()
        self._load_all_pricing()


# Global pricing loader instance
_global_loader: Optional[PricingLoader] = None


def get_pricing(provider: str, model: str) -> ModelPricing:
    """Get pricing for a model using the global loader."""
    global _global_loader
    if _global_loader is None:
        _global_loader = PricingLoader()
    return _global_loader.get_pricing(provider, model)


def get_pricing_loader() -> PricingLoader:
    """Get the global pricing loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = PricingLoader()
    return _global_loader
