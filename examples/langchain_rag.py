#!/usr/bin/env python3
"""
LLM Cost Guard - LangChain RAG Example

This example demonstrates how to integrate LLM Cost Guard with LangChain
for a Retrieval-Augmented Generation (RAG) pipeline.

Note: This example requires the langchain optional dependency:
    pip install llm-cost-guard[langchain]
"""

from llm_cost_guard import CostTracker, Budget, BudgetAction

# Check if LangChain is available
try:
    from langchain_core.callbacks.base import BaseCallbackHandler

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not installed. Install with: pip install llm-cost-guard[langchain]")
    print("Running in simulation mode...\n")


def main():
    print("=== LLM Cost Guard - LangChain RAG Example ===\n")

    # ==========================================================================
    # Setup Cost Tracker
    # ==========================================================================

    tracker = CostTracker(
        backend="memory",
        budgets=[
            Budget(
                name="rag-pipeline",
                limit=10.00,
                period="day",
                action=BudgetAction.WARN,
            ),
        ],
    )

    # ==========================================================================
    # LangChain Integration
    # ==========================================================================

    if LANGCHAIN_AVAILABLE:
        from llm_cost_guard.integrations.langchain import CostTrackingCallback, track_chain

        # Create the callback handler
        cost_callback = CostTrackingCallback(
            tracker=tracker,
            tags={"pipeline": "rag", "version": "1.0"},
        )

        print("LangChain callback handler created.")
        print("You can add this to any LangChain LLM or chain:\n")
        print("  llm = ChatOpenAI(")
        print('      model="gpt-4o",')
        print("      callbacks=[cost_callback]")
        print("  )")

    # ==========================================================================
    # Simulated RAG Pipeline with Span Tracking
    # ==========================================================================

    print("\n--- Simulated RAG Pipeline ---\n")

    # Use spans to group related calls
    with tracker.span("rag_query", tags={"query_type": "factual"}) as query_span:
        # Step 1: Embed the query
        print("Step 1: Embedding query...")
        with tracker.span("embed_query") as embed_span:
            tracker.record(
                provider="openai",
                model="text-embedding-3-small",
                input_tokens=50,
                output_tokens=0,
                tags={"step": "embed_query"},
            )
        print(f"  Embedding cost: ${embed_span.total_cost:.6f}")

        # Step 2: Retrieve documents (simulated - no LLM cost)
        print("\nStep 2: Retrieving documents...")
        print("  Retrieved 5 relevant documents")

        # Step 3: Rerank documents (optional LLM step)
        print("\nStep 3: Reranking documents...")
        with tracker.span("rerank") as rerank_span:
            tracker.record(
                provider="openai",
                model="gpt-4o-mini",
                input_tokens=2000,  # Context from documents
                output_tokens=100,  # Ranking output
                tags={"step": "rerank"},
            )
        print(f"  Rerank cost: ${rerank_span.total_cost:.6f}")

        # Step 4: Generate answer
        print("\nStep 4: Generating answer...")
        with tracker.span("generate") as gen_span:
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=3000,  # System + context + query
                output_tokens=500,  # Generated answer
                tags={"step": "generate"},
            )
        print(f"  Generation cost: ${gen_span.total_cost:.6f}")

    # ==========================================================================
    # Query Span Summary
    # ==========================================================================

    print("\n" + "=" * 50)
    print("RAG QUERY SUMMARY")
    print("=" * 50)
    print(f"Total calls: {query_span.call_count}")
    print(f"Total cost: ${query_span.total_cost:.6f}")
    print(f"Total input tokens: {query_span.total_input_tokens:,}")
    print(f"Total output tokens: {query_span.total_output_tokens:,}")
    print(f"Models used: {query_span.models_used}")

    # Breakdown by step
    print("\nCost breakdown:")
    print(f"  Embedding:   ${embed_span.total_cost:.6f} ({embed_span.total_cost/query_span.total_cost*100:.1f}%)")
    print(f"  Reranking:   ${rerank_span.total_cost:.6f} ({rerank_span.total_cost/query_span.total_cost*100:.1f}%)")
    print(f"  Generation:  ${gen_span.total_cost:.6f} ({gen_span.total_cost/query_span.total_cost*100:.1f}%)")

    # ==========================================================================
    # Multiple Queries Simulation
    # ==========================================================================

    print("\n--- Simulating Multiple Queries ---\n")

    query_types = ["factual", "analytical", "creative"]

    for i, query_type in enumerate(query_types):
        with tracker.span(f"query_{i+1}", tags={"query_type": query_type}) as span:
            # Simplified pipeline
            tracker.record(
                provider="openai",
                model="text-embedding-3-small",
                input_tokens=50,
                output_tokens=0,
            )
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=2000 + i * 500,
                output_tokens=300 + i * 100,
            )

        print(f"Query {i+1} ({query_type}): ${span.total_cost:.6f}")

    # ==========================================================================
    # Session Report
    # ==========================================================================

    print("\n" + "=" * 50)
    print("SESSION REPORT")
    print("=" * 50)

    report = tracker.daily_report()
    print(f"Total queries: {report.total_calls}")
    print(f"Total cost: ${report.total_cost:.6f}")
    print(f"Average cost per call: ${report.total_cost/report.total_calls:.6f}")

    # Check budget
    budget_util = tracker.get_budget_utilization("rag-pipeline")
    print(f"\nBudget utilization: {budget_util:.1f}%")

    # Group by model
    by_model = tracker.get_costs(group_by=["model"])
    if by_model.grouped_data.get("groups"):
        print("\nCost by model:")
        for group in by_model.grouped_data["groups"]:
            print(f"  {group['model']}: ${group['cost']:.6f} ({group['calls']} calls)")

    tracker.close()
    print("\n✅ LangChain RAG example completed!")


if __name__ == "__main__":
    main()
