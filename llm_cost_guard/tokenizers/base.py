"""
Token counting utilities for LLM Cost Guard.
"""

import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class TokenCounter:
    """Token counter with support for multiple providers."""

    def __init__(self):
        self._tiktoken_encodings: dict[str, Any] = {}
        self._tiktoken_available = False
        self._check_tiktoken()

    def _check_tiktoken(self) -> None:
        """Check if tiktoken is available."""
        try:
            import tiktoken  # noqa: F401

            self._tiktoken_available = True
        except ImportError:
            self._tiktoken_available = False
            logger.warning("tiktoken not available, using estimation for OpenAI models")

    def _get_tiktoken_encoding(self, model: str) -> Any:
        """Get or create a tiktoken encoding for a model."""
        if not self._tiktoken_available:
            return None

        if model in self._tiktoken_encodings:
            return self._tiktoken_encodings[model]

        import tiktoken

        try:
            # Try to get encoding for specific model
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base for GPT-4 and newer models
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoding = tiktoken.get_encoding("gpt2")

        self._tiktoken_encodings[model] = encoding
        return encoding

    def count_tokens(
        self,
        text: Union[str, list[dict[str, str]]],
        model: str,
        provider: str = "openai",
    ) -> int:
        """
        Count tokens in text or messages.

        Args:
            text: Either a string or a list of message dicts
            model: Model name
            provider: Provider name (openai, anthropic, etc.)

        Returns:
            Number of tokens
        """
        provider = provider.lower()

        if isinstance(text, list):
            # It's a list of messages
            return self._count_message_tokens(text, model, provider)

        # It's a string
        return self._count_string_tokens(text, model, provider)

    def _count_string_tokens(self, text: str, model: str, provider: str) -> int:
        """Count tokens in a string."""
        if provider == "openai" and self._tiktoken_available:
            encoding = self._get_tiktoken_encoding(model)
            if encoding:
                return len(encoding.encode(text))

        # Fallback estimation: ~4 characters per token
        return self._estimate_tokens(text)

    def _count_message_tokens(
        self, messages: list[dict[str, str]], model: str, provider: str
    ) -> int:
        """Count tokens in a list of messages."""
        if provider == "openai" and self._tiktoken_available:
            return self._count_openai_message_tokens(messages, model)

        if provider == "anthropic":
            return self._count_anthropic_message_tokens(messages)

        # Fallback: count tokens in all message content
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._count_string_tokens(content, model, provider)
            elif isinstance(content, list):
                # Handle multi-modal content
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        total += self._count_string_tokens(item["text"], model, provider)
        return total

    def _count_openai_message_tokens(
        self, messages: list[dict[str, str]], model: str
    ) -> int:
        """Count tokens for OpenAI chat messages."""
        encoding = self._get_tiktoken_encoding(model)
        if not encoding:
            return self._estimate_message_tokens(messages)

        # Token overhead per message (varies by model)
        if model.startswith("gpt-4o") or model.startswith("gpt-4-"):
            tokens_per_message = 3
            tokens_per_name = 1
        elif model.startswith("gpt-3.5"):
            tokens_per_message = 4
            tokens_per_name = -1
        else:
            tokens_per_message = 3
            tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _count_anthropic_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate tokens for Anthropic messages."""
        # Anthropic doesn't provide a public tokenizer
        # Use estimation based on character count
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        total += self._estimate_tokens(item["text"])
            # Add overhead for role
            total += 4

        return total

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on text length."""
        if not text:
            return 0
        # Rough estimate: ~4 characters per token for English
        # This is a common approximation used when tokenizers aren't available
        return max(1, len(text) // 4)

    def _estimate_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate tokens for a list of messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens(content)
            total += 4  # Overhead per message
        return total


# Global token counter instance
_global_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get the global token counter instance."""
    global _global_counter
    if _global_counter is None:
        _global_counter = TokenCounter()
    return _global_counter


def count_tokens(
    text: Union[str, list[dict[str, str]]],
    model: str,
    provider: str = "openai",
) -> int:
    """
    Count tokens in text or messages using the global counter.

    Args:
        text: Either a string or a list of message dicts
        model: Model name
        provider: Provider name

    Returns:
        Number of tokens
    """
    return get_token_counter().count_tokens(text, model, provider)
