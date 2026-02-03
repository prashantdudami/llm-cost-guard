#!/usr/bin/env python3
"""
LLM Cost Guard - Distributed Tracking Example

This example demonstrates patterns for distributed cost tracking
across multiple application instances using SQLite (for demo) or Redis.

In production, you would use Redis or a database backend for true
distributed tracking.
"""

import threading
import time
import tempfile
import os

from llm_cost_guard import CostTracker, Budget, BudgetAction
from llm_cost_guard.exceptions import BudgetExceededError


def main():
    print("=== LLM Cost Guard - Distributed Tracking Example ===\n")

    # ==========================================================================
    # Setup Shared Backend
    # ==========================================================================

    # In production, you'd use Redis:
    # tracker = CostTracker(backend="redis://localhost:6379/0")

    # For this demo, we use a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"Using shared SQLite database: {db_path}\n")

    # ==========================================================================
    # Simulate Multiple Application Instances
    # ==========================================================================

    print("--- Simulating Multiple Application Instances ---\n")

    instance_stats = {}
    errors = []
    lock = threading.Lock()

    def create_instance_tracker(instance_id: str) -> CostTracker:
        """Create a tracker for a specific instance."""
        return CostTracker(
            backend=f"sqlite:///{db_path}",
            budgets=[
                Budget(
                    name="shared-daily",
                    limit=1.00,  # Low limit for demo
                    period="day",
                    action=BudgetAction.BLOCK,
                ),
            ],
        )

    def simulate_instance_workload(instance_id: str, num_requests: int):
        """Simulate workload for one application instance."""
        tracker = create_instance_tracker(instance_id)

        successful = 0
        blocked = 0
        total_cost = 0.0

        for i in range(num_requests):
            try:
                record = tracker.record(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=500 + (i * 100),
                    output_tokens=200 + (i * 50),
                    tags={
                        "instance": instance_id,
                        "request": str(i),
                    },
                )
                successful += 1
                total_cost += record.total_cost
                time.sleep(0.01)  # Small delay to simulate real work

            except BudgetExceededError:
                blocked += 1

            except Exception as e:
                with lock:
                    errors.append((instance_id, str(e)))

        tracker.close()

        with lock:
            instance_stats[instance_id] = {
                "successful": successful,
                "blocked": blocked,
                "cost": total_cost,
            }

    # Start multiple "instances" as threads
    num_instances = 4
    requests_per_instance = 10

    threads = []
    for i in range(num_instances):
        instance_id = f"instance-{i+1}"
        t = threading.Thread(
            target=simulate_instance_workload,
            args=(instance_id, requests_per_instance),
        )
        threads.append(t)

    print(f"Starting {num_instances} instances, each making {requests_per_instance} requests...")
    start_time = time.time()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds\n")

    # ==========================================================================
    # Instance Statistics
    # ==========================================================================

    print("=" * 60)
    print("INSTANCE STATISTICS")
    print("=" * 60)

    total_successful = 0
    total_blocked = 0
    total_cost = 0.0

    for instance_id, stats in sorted(instance_stats.items()):
        print(f"\n{instance_id}:")
        print(f"  Successful requests: {stats['successful']}")
        print(f"  Blocked (budget):    {stats['blocked']}")
        print(f"  Total cost:          ${stats['cost']:.6f}")

        total_successful += stats["successful"]
        total_blocked += stats["blocked"]
        total_cost += stats["cost"]

    print(f"\nTotal across all instances:")
    print(f"  Successful: {total_successful}")
    print(f"  Blocked:    {total_blocked}")
    print(f"  Cost:       ${total_cost:.6f}")

    if errors:
        print(f"\nErrors: {len(errors)}")
        for instance_id, error in errors[:5]:
            print(f"  {instance_id}: {error}")

    # ==========================================================================
    # Verify Shared State
    # ==========================================================================

    print("\n" + "=" * 60)
    print("SHARED DATABASE VERIFICATION")
    print("=" * 60)

    # Create a new tracker to read the shared state
    verify_tracker = CostTracker(backend=f"sqlite:///{db_path}")

    report = verify_tracker.daily_report()
    print(f"\nTotal records in database: {report.total_calls}")
    print(f"Total cost recorded: ${report.total_cost:.6f}")

    # Group by instance
    by_instance = verify_tracker.get_costs(group_by=["instance"])
    if by_instance.grouped_data.get("groups"):
        print("\nRecords by instance:")
        for group in by_instance.grouped_data["groups"]:
            print(f"  {group['instance']}: {group['calls']} calls, ${group['cost']:.6f}")

    verify_tracker.close()

    # ==========================================================================
    # Budget Enforcement Demonstration
    # ==========================================================================

    print("\n" + "=" * 60)
    print("DISTRIBUTED BUDGET ENFORCEMENT")
    print("=" * 60)

    print("\nThe shared budget was enforced across all instances.")
    print(f"Budget limit: $1.00")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Requests blocked: {total_blocked}")

    if total_blocked > 0:
        print("\n✓ Budget enforcement worked - requests were blocked when the")
        print("  shared budget was exceeded, even across different instances.")
    else:
        print("\n✓ All requests completed within budget.")

    # Cleanup
    os.unlink(db_path)

    # ==========================================================================
    # Redis Configuration Notes
    # ==========================================================================

    print("\n" + "=" * 60)
    print("PRODUCTION REDIS CONFIGURATION")
    print("=" * 60)

    print("""
For production distributed tracking, use Redis:

    tracker = CostTracker(
        backend="redis://localhost:6379/0",
        
        # Distributed budget mode
        budget_mode="distributed",
        budget_sync_interval_ms=100,
        budget_reservation="pessimistic",
    )

Budget reservation strategies:

1. pessimistic (safest):
   - Reserves estimated cost BEFORE the LLM call
   - May over-block but never over-spends
   - Best for strict budget enforcement

2. optimistic (highest availability):
   - Checks budget AFTER the call
   - May briefly over-spend before blocking
   - Best for high-throughput applications

3. sampling (balanced):
   - Checks budget every N calls
   - Low overhead, less accurate
   - Best for very high volume applications
""")

    print("✅ Distributed tracking example completed!")


if __name__ == "__main__":
    main()
