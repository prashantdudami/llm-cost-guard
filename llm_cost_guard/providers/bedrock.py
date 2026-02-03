"""
AWS Bedrock provider for LLM Cost Guard.
"""

import json
from typing import Any

from llm_cost_guard.models import UsageData
from llm_cost_guard.providers.base import Provider


class BedrockProvider(Provider):
    """AWS Bedrock API provider."""

    @property
    def name(self) -> str:
        return "bedrock"

    def extract_usage(self, response: Any) -> UsageData:
        """Extract token usage from a Bedrock API response."""
        usage = UsageData()

        # Handle dictionary response (from boto3)
        if isinstance(response, dict):
            # Check for standard usage field
            if "usage" in response:
                usage_data = response["usage"]
                usage.input_tokens = usage_data.get("inputTokens", 0)
                usage.output_tokens = usage_data.get("outputTokens", 0)
                usage.total_tokens = usage_data.get("totalTokens", 0)
                return usage

            # Check for Bedrock response metadata
            metadata = response.get("ResponseMetadata", {})
            headers = metadata.get("HTTPHeaders", {})

            # Bedrock includes token counts in headers for some models
            if "x-amzn-bedrock-input-token-count" in headers:
                usage.input_tokens = int(headers["x-amzn-bedrock-input-token-count"])
            if "x-amzn-bedrock-output-token-count" in headers:
                usage.output_tokens = int(headers["x-amzn-bedrock-output-token-count"])

            # Also check the body for Claude-style responses
            body = response.get("body", {})
            if isinstance(body, bytes):
                try:
                    body = json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body = {}

            if isinstance(body, dict) and "usage" in body:
                body_usage = body["usage"]
                usage.input_tokens = body_usage.get(
                    "input_tokens", body_usage.get("inputTokens", usage.input_tokens)
                )
                usage.output_tokens = body_usage.get(
                    "output_tokens", body_usage.get("outputTokens", usage.output_tokens)
                )

            usage.total_tokens = usage.input_tokens + usage.output_tokens

        return usage

    def extract_model(self, response: Any) -> str:
        """Extract the model name from a Bedrock API response."""
        if isinstance(response, dict):
            # Check for model ID in various locations
            if "modelId" in response:
                return response["modelId"]

            # Check response metadata
            metadata = response.get("ResponseMetadata", {})
            if "modelId" in metadata:
                return metadata["modelId"]

        return "unknown"

    def normalize_model_name(self, model: str) -> str:
        """Normalize Bedrock model name for pricing lookup."""
        # Bedrock model IDs include version info that should be preserved
        # e.g., "anthropic.claude-3-sonnet-20240229-v1:0"
        return model


class BedrockStreamingHandler:
    """Handler for streaming Bedrock responses."""

    def __init__(self, model_id: str = "unknown"):
        self.input_tokens = 0
        self.output_tokens = 0
        self.model = model_id
        self._chunks: list = []

    def handle_chunk(self, chunk: Any) -> None:
        """Process a streaming chunk."""
        if isinstance(chunk, dict):
            # Handle Bedrock's streaming chunk format
            if "chunk" in chunk:
                chunk_data = chunk["chunk"]
                if "bytes" in chunk_data:
                    try:
                        parsed = json.loads(chunk_data["bytes"].decode("utf-8"))
                        self._process_parsed_chunk(parsed)
                    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                        pass

            # Some responses have direct usage info
            if "usage" in chunk:
                usage = chunk["usage"]
                self.input_tokens = usage.get("inputTokens", self.input_tokens)
                self.output_tokens = usage.get("outputTokens", self.output_tokens)

        self._chunks.append(chunk)

    def _process_parsed_chunk(self, parsed: dict) -> None:
        """Process a parsed JSON chunk."""
        # Claude on Bedrock format
        if "usage" in parsed:
            usage = parsed["usage"]
            self.input_tokens = usage.get("input_tokens", self.input_tokens)
            self.output_tokens = usage.get("output_tokens", self.output_tokens)

        # Llama/Titan format
        if "amazon-bedrock-invocationMetrics" in parsed:
            metrics = parsed["amazon-bedrock-invocationMetrics"]
            self.input_tokens = metrics.get("inputTokenCount", self.input_tokens)
            self.output_tokens = metrics.get("outputTokenCount", self.output_tokens)

    def get_usage(self) -> UsageData:
        """Get final usage data."""
        return UsageData(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
        )
