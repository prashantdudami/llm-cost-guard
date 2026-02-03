#!/usr/bin/env python3
"""
LLM Cost Guard - AWS Bedrock Example

This example demonstrates how to track costs for AWS Bedrock API calls
across different foundation models.

Note: This example simulates Bedrock calls. For real usage, you need:
    pip install llm-cost-guard[bedrock]
"""

from llm_cost_guard import CostTracker, Budget, BudgetAction


def main():
    print("=== LLM Cost Guard - AWS Bedrock Example ===\n")

    # ==========================================================================
    # Setup for Bedrock
    # ==========================================================================

    tracker = CostTracker(
        backend="memory",
        budgets=[
            Budget(
                name="bedrock-daily",
                limit=50.00,
                period="day",
                action=BudgetAction.WARN,
            ),
        ],
    )

    # ==========================================================================
    # Bedrock Model Pricing
    # ==========================================================================

    print("--- Available Bedrock Models ---\n")

    from llm_cost_guard.pricing.loader import PricingLoader

    pricing = PricingLoader()
    bedrock_models = pricing.get_all_models("bedrock")

    # Show a selection of popular models
    popular_models = [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
        "mistral.mistral-large-2407-v1:0",
        "amazon.titan-text-express-v1",
    ]

    print("Popular Bedrock models and pricing:")
    for model in popular_models:
        try:
            model_pricing = pricing.get_pricing("bedrock", model)
            print(f"  {model}:")
            print(f"    Input:  ${model_pricing.input_cost_per_1k:.6f}/1K tokens")
            print(f"    Output: ${model_pricing.output_cost_per_1k:.6f}/1K tokens")
        except Exception:
            print(f"  {model}: (pricing not available)")

    # ==========================================================================
    # Simulated Bedrock Calls
    # ==========================================================================

    print("\n--- Simulated Bedrock API Calls ---\n")

    # Claude on Bedrock
    print("1. Claude 3.5 Sonnet on Bedrock:")
    record = tracker.record(
        provider="bedrock",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        input_tokens=2000,
        output_tokens=500,
        tags={"application": "customer-support"},
    )
    print(f"   Input tokens: {record.input_tokens}")
    print(f"   Output tokens: {record.output_tokens}")
    print(f"   Total cost: ${record.total_cost:.6f}")

    # Claude Haiku for faster, cheaper tasks
    print("\n2. Claude 3 Haiku on Bedrock (cost-optimized):")
    record = tracker.record(
        provider="bedrock",
        model="anthropic.claude-3-haiku-20240307-v1:0",
        input_tokens=2000,
        output_tokens=500,
        tags={"application": "classification"},
    )
    print(f"   Input tokens: {record.input_tokens}")
    print(f"   Output tokens: {record.output_tokens}")
    print(f"   Total cost: ${record.total_cost:.6f}")

    # Llama on Bedrock
    print("\n3. Llama 3.1 70B on Bedrock:")
    record = tracker.record(
        provider="bedrock",
        model="meta.llama3-1-70b-instruct-v1:0",
        input_tokens=2000,
        output_tokens=500,
        tags={"application": "code-generation"},
    )
    print(f"   Input tokens: {record.input_tokens}")
    print(f"   Output tokens: {record.output_tokens}")
    print(f"   Total cost: ${record.total_cost:.6f}")

    # Titan for simple tasks
    print("\n4. Amazon Titan Text Express:")
    record = tracker.record(
        provider="bedrock",
        model="amazon.titan-text-express-v1",
        input_tokens=1000,
        output_tokens=200,
        tags={"application": "summarization"},
    )
    print(f"   Input tokens: {record.input_tokens}")
    print(f"   Output tokens: {record.output_tokens}")
    print(f"   Total cost: ${record.total_cost:.6f}")

    # Embeddings
    print("\n5. Titan Embeddings:")
    record = tracker.record(
        provider="bedrock",
        model="amazon.titan-embed-text-v2:0",
        input_tokens=500,
        output_tokens=0,
        tags={"application": "search"},
    )
    print(f"   Input tokens: {record.input_tokens}")
    print(f"   Total cost: ${record.total_cost:.6f}")

    # ==========================================================================
    # Cost Comparison
    # ==========================================================================

    print("\n--- Model Cost Comparison ---\n")

    # Compare costs for the same workload across models
    test_input_tokens = 5000
    test_output_tokens = 1000

    models_to_compare = [
        ("bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        ("bedrock", "anthropic.claude-3-haiku-20240307-v1:0"),
        ("bedrock", "meta.llama3-1-70b-instruct-v1:0"),
        ("bedrock", "amazon.titan-text-express-v1"),
    ]

    print(f"Cost for {test_input_tokens} input + {test_output_tokens} output tokens:\n")

    for provider, model in models_to_compare:
        try:
            _, _, cost = pricing.calculate_cost(
                provider=provider,
                model=model,
                input_tokens=test_input_tokens,
                output_tokens=test_output_tokens,
            )
            model_short = model.split(".")[-1].split("-")[0:3]
            model_name = "-".join(model_short)[:30]
            print(f"  {model_name:32} ${cost:.6f}")
        except Exception:
            pass

    # ==========================================================================
    # Usage Report
    # ==========================================================================

    print("\n" + "=" * 50)
    print("BEDROCK USAGE REPORT")
    print("=" * 50)

    report = tracker.daily_report()
    print(f"\nTotal API calls: {report.total_calls}")
    print(f"Total cost: ${report.total_cost:.6f}")
    print(f"Total input tokens: {report.total_input_tokens:,}")
    print(f"Total output tokens: {report.total_output_tokens:,}")

    # Group by model
    by_model = tracker.get_costs(group_by=["model"])
    if by_model.grouped_data.get("groups"):
        print("\nCost breakdown by model:")
        for group in by_model.grouped_data["groups"]:
            pct = (group["cost"] / report.total_cost) * 100 if report.total_cost > 0 else 0
            print(f"  {group['model'][:50]}")
            print(f"    Cost: ${group['cost']:.6f} ({pct:.1f}%)")
            print(f"    Calls: {group['calls']}")

    # Group by application
    by_app = tracker.get_costs(group_by=["application"])
    if by_app.grouped_data.get("groups"):
        print("\nCost breakdown by application:")
        for group in by_app.grouped_data["groups"]:
            print(f"  {group['application']}: ${group['cost']:.6f} ({group['calls']} calls)")

    tracker.close()
    print("\n✅ Bedrock example completed!")


if __name__ == "__main__":
    main()
