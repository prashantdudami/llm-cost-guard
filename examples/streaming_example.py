#!/usr/bin/env python3
"""
LLM Cost Guard - Streaming Response Example

This example demonstrates how to track costs for streaming API responses.
"""

import asyncio
from llm_cost_guard import CostTracker


def main():
    print("=== LLM Cost Guard - Streaming Example ===\n")

    # Create tracker configured for streaming
    tracker = CostTracker(
        backend="memory",
        streaming_budget_mode="estimate",  # Estimate costs before completion
        streaming_max_output_estimate=4096,  # Assume max output for budget checks
    )

    # ==========================================================================
    # Simulating Streaming with Manual Recording
    # ==========================================================================

    print("--- Simulating Streaming Response ---\n")

    # In real usage, you'd wrap a streaming response and track tokens as they arrive
    # Here we simulate the process

    # Before streaming starts, you might estimate the cost
    from llm_cost_guard.pricing.loader import PricingLoader

    pricing = PricingLoader()

    input_tokens = 500
    max_output = 4096

    estimated_cost = pricing.estimate_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=input_tokens,
        max_output_tokens=max_output,
    )

    print(f"Estimated max cost for stream: ${estimated_cost:.4f}")
    print("Starting stream...")

    # Simulate streaming chunks arriving
    output_tokens_received = 0
    chunks_received = 0

    for chunk_size in [50, 75, 100, 125, 150, 100, 80, 60, 40, 20]:
        output_tokens_received += chunk_size
        chunks_received += 1
        print(f"  Chunk {chunks_received}: +{chunk_size} tokens (total: {output_tokens_received})")

    print(f"\nStream complete. Total output tokens: {output_tokens_received}")

    # Record the actual cost after streaming completes
    record = tracker.record(
        provider="openai",
        model="gpt-4o",
        input_tokens=input_tokens,
        output_tokens=output_tokens_received,
        tags={"type": "streaming"},
    )

    print(f"Actual cost: ${record.total_cost:.6f}")
    print(f"Savings vs estimate: ${estimated_cost - record.total_cost:.6f}")

    # ==========================================================================
    # Async Streaming Pattern
    # ==========================================================================

    print("\n--- Async Streaming Pattern ---\n")

    async def stream_with_tracking():
        """Example of tracking async streaming responses."""

        # Simulate async stream
        total_output_tokens = 0

        async def mock_stream():
            for chunk_size in [100, 150, 200, 150, 100, 50]:
                await asyncio.sleep(0.01)  # Simulate network delay
                yield chunk_size

        print("Processing async stream...")
        async for tokens in mock_stream():
            total_output_tokens += tokens

        # Record after stream completes
        return tracker.record(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=total_output_tokens,
            tags={"type": "async_streaming"},
        )

    # Run async example
    record = asyncio.run(stream_with_tracking())
    print(f"Async stream cost: ${record.total_cost:.6f}")

    # ==========================================================================
    # Streaming with Span Tracking
    # ==========================================================================

    print("\n--- Streaming within Span ---\n")

    with tracker.span("rag_pipeline", tags={"pipeline": "streaming"}) as span:
        # Embedding call
        tracker.record(
            provider="openai",
            model="text-embedding-3-small",
            input_tokens=500,
            output_tokens=0,
            tags={"step": "embedding"},
        )
        print("  Embedding step: recorded")

        # Streaming generation
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=2000,
            output_tokens=1500,
            tags={"step": "generation", "type": "streaming"},
        )
        print("  Generation step: recorded")

    print(f"\nSpan '{span.name}' summary:")
    print(f"  Total calls: {span.call_count}")
    print(f"  Total cost: ${span.total_cost:.6f}")
    print(f"  Models used: {span.models_used}")

    # ==========================================================================
    # Summary Report
    # ==========================================================================

    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)

    report = tracker.daily_report()
    print(f"Total streaming calls: {report.total_calls}")
    print(f"Total cost: ${report.total_cost:.6f}")
    print(f"Total input tokens: {report.total_input_tokens:,}")
    print(f"Total output tokens: {report.total_output_tokens:,}")

    tracker.close()


if __name__ == "__main__":
    main()
