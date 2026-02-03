"""
Unit tests for LLM providers.
"""

import pytest
from unittest.mock import MagicMock

from llm_cost_guard.providers import detect_provider, get_provider
from llm_cost_guard.providers.openai import OpenAIProvider
from llm_cost_guard.providers.anthropic import AnthropicProvider
from llm_cost_guard.providers.bedrock import BedrockProvider
from llm_cost_guard.models import UsageData


class TestProviderDetection:
    """Tests for provider detection."""

    def test_detect_openai(self):
        """Test detecting OpenAI models."""
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gpt-4-turbo") == "openai"
        assert detect_provider("gpt-3.5-turbo") == "openai"
        assert detect_provider("text-embedding-3-small") == "openai"
        assert detect_provider("o1-preview") == "openai"

    def test_detect_anthropic(self):
        """Test detecting Anthropic models."""
        assert detect_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert detect_provider("claude-3-opus-20240229") == "anthropic"
        assert detect_provider("claude-2.1") == "anthropic"

    def test_detect_bedrock(self):
        """Test detecting Bedrock models."""
        assert detect_provider("anthropic.claude-3-5-sonnet-20241022-v2:0") == "bedrock"
        assert detect_provider("amazon.titan-text-express-v1") == "bedrock"
        assert detect_provider("meta.llama3-70b-instruct-v1:0") == "bedrock"
        assert detect_provider("cohere.command-r-plus-v1:0") == "bedrock"

    def test_detect_vertex(self):
        """Test detecting Vertex AI models."""
        assert detect_provider("gemini-1.5-pro") == "vertex"
        assert detect_provider("gemini-1.5-flash") == "vertex"

    def test_get_provider(self):
        """Test getting provider instances."""
        openai = get_provider("openai")
        assert isinstance(openai, OpenAIProvider)

        anthropic = get_provider("anthropic")
        assert isinstance(anthropic, AnthropicProvider)

        bedrock = get_provider("bedrock")
        assert isinstance(bedrock, BedrockProvider)


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_name(self):
        """Test provider name."""
        provider = OpenAIProvider()
        assert provider.name == "openai"

    def test_extract_usage_from_object(self, mock_openai_response):
        """Test extracting usage from OpenAI response object."""
        provider = OpenAIProvider()

        usage = provider.extract_usage(mock_openai_response)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_extract_usage_from_dict(self):
        """Test extracting usage from dict response."""
        provider = OpenAIProvider()

        response = {
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        usage = provider.extract_usage(response)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_extract_model(self, mock_openai_response):
        """Test extracting model from response."""
        provider = OpenAIProvider()

        model = provider.extract_model(mock_openai_response)
        assert model == "gpt-4o"

    def test_extract_cached_tokens(self):
        """Test extracting cached tokens."""
        provider = OpenAIProvider()

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

            class prompt_tokens_details:
                cached_tokens = 30

        class MockResponse:
            model = "gpt-4o"
            usage = MockUsage()

        usage = provider.extract_usage(MockResponse())
        assert usage.cached_tokens == 30


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_name(self):
        """Test provider name."""
        provider = AnthropicProvider()
        assert provider.name == "anthropic"

    def test_extract_usage_from_object(self, mock_anthropic_response):
        """Test extracting usage from Anthropic response object."""
        provider = AnthropicProvider()

        usage = provider.extract_usage(mock_anthropic_response)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_extract_usage_from_dict(self):
        """Test extracting usage from dict response."""
        provider = AnthropicProvider()

        response = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        }

        usage = provider.extract_usage(response)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_extract_model(self, mock_anthropic_response):
        """Test extracting model from response."""
        provider = AnthropicProvider()

        model = provider.extract_model(mock_anthropic_response)
        assert model == "claude-3-5-sonnet-20241022"


class TestBedrockProvider:
    """Tests for Bedrock provider."""

    def test_name(self):
        """Test provider name."""
        provider = BedrockProvider()
        assert provider.name == "bedrock"

    def test_extract_usage_from_dict_with_body(self):
        """Test extracting usage from Bedrock dict response with body."""
        provider = BedrockProvider()

        import json

        response = {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "body": json.dumps(
                {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                    }
                }
            ).encode("utf-8"),
        }

        usage = provider.extract_usage(response)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_extract_usage_from_headers(self):
        """Test extracting usage from Bedrock headers."""
        provider = BedrockProvider()

        response = {
            "modelId": "amazon.titan-text-express-v1",
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-amzn-bedrock-input-token-count": "100",
                    "x-amzn-bedrock-output-token-count": "50",
                }
            },
        }

        usage = provider.extract_usage(response)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_extract_model(self):
        """Test extracting model from Bedrock response."""
        provider = BedrockProvider()

        response = {"modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"}

        model = provider.extract_model(response)
        assert model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
