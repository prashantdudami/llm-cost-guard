"""
LLM provider integrations for LLM Cost Guard.
"""

from llm_cost_guard.providers.base import Provider
from llm_cost_guard.providers.openai import OpenAIProvider
from llm_cost_guard.providers.anthropic import AnthropicProvider
from llm_cost_guard.providers.bedrock import BedrockProvider

__all__ = [
    "Provider",
    "OpenAIProvider",
    "AnthropicProvider",
    "BedrockProvider",
    "get_provider",
    "detect_provider",
]


def get_provider(name: str) -> Provider:
    """Get a provider by name."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "bedrock": BedrockProvider,
    }

    name = name.lower()
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")

    return providers[name]()


def detect_provider(model: str) -> str:
    """Detect the provider from a model name."""
    model_lower = model.lower()

    # OpenAI models
    if any(
        prefix in model_lower
        for prefix in ["gpt-", "o1", "text-embedding", "dall-e", "whisper", "tts-"]
    ):
        return "openai"

    # Anthropic models
    if "claude" in model_lower and not model_lower.startswith("anthropic."):
        return "anthropic"

    # AWS Bedrock models (have provider prefix)
    if any(
        model_lower.startswith(prefix)
        for prefix in [
            "anthropic.",
            "amazon.",
            "meta.",
            "mistral.",
            "cohere.",
            "ai21.",
        ]
    ):
        return "bedrock"

    # Google Vertex AI
    if any(prefix in model_lower for prefix in ["gemini", "palm", "text-bison", "chat-bison"]):
        return "vertex"

    # Default to OpenAI
    return "openai"
