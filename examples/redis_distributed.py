"""
Redis Distributed Budget Enforcement Example

Demonstrates distributed budget enforcement across multiple instances
using Redis with atomic Lua scripts for consistency.

This is REQUIRED for:
- Multi-pod/container deployments (Kubernetes)
- Serverless functions (Lambda, Cloud Functions)
- Any distributed LLM application

Install:
    pip install llm-cost-guard[redis]

Requirements:
    - Redis server running (docker run -p 6379:6379 redis:alpine)

Run:
    python redis_distributed.py
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Note: This example requires Redis. If Redis is not available,
# it will fall back to demonstrating the API with mocks.

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  Redis package not installed. Install with: pip install redis")
    print("   Running in demo mode with mocked responses.\n")


def check_redis_connection():
    """Check if Redis is running."""
    if not REDIS_AVAILABLE:
        return False
    try:
        r = redis.Redis(host="localhost", port=6379, db=0)
        r.ping()
        return True
    except:
        return False


def example_distributed_budget_basic():
    """Basic distributed budget setup with Redis."""
    print("\n" + "=" * 60)
    print("Distributed Budget with Redis")
    print("=" * 60)
    
    from llm_cost_guard import CostTracker, Budget, BudgetAction
    
    if not check_redis_connection():
        print("\n⚠️  Redis not available. Using memory backend for demo.")
        backend_url = "memory"
    else:
        backend_url = "redis://localhost:6379/0"
        print(f"\n✅ Connected to Redis: {backend_url}")
    
    # Create tracker with Redis backend
    tracker = CostTracker(
        backend=backend_url,
        budgets=[
            Budget(
                name="global-daily",
                limit=100.00,
                period="day",
                action=BudgetAction.BLOCK,
                warning_threshold=0.8,
            ),
        ],
        budget_mode="distributed",  # Enable distributed mode
    )
    
    print("\n1. Recording costs from 'Pod-1'...")
    for i in range(5):
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=500,
            output_tokens=200,
            tags={"pod": "pod-1", "request_id": f"req-{i}"},
        )
        print(f"   Request {i+1}: ${record.total_cost:.4f}")
    
    # Check budget utilization
    utilization = tracker.get_budget_utilization("global-daily")
    print(f"\n2. Budget utilization: {utilization*100:.2f}%")
    
    # Get metrics
    metrics = tracker.get_metrics()
    print(f"\n3. Metrics: {metrics}")
    
    tracker.close()


def example_concurrent_instances():
    """Simulate multiple service instances hitting the same budget."""
    print("\n" + "=" * 60)
    print("Concurrent Instances (Multi-Pod Simulation)")
    print("=" * 60)
    
    from llm_cost_guard import CostTracker, Budget, BudgetAction
    from llm_cost_guard.exceptions import BudgetExceededError
    
    if not check_redis_connection():
        print("\n⚠️  Redis not available. Using memory backend for demo.")
        backend_url = "memory"
    else:
        backend_url = "redis://localhost:6379/1"  # Use different db
        print(f"\n✅ Connected to Redis: {backend_url}")
    
    # Shared budget across instances
    budget = Budget(
        name="shared-budget",
        limit=0.50,  # Low limit to trigger exceeded quickly
        period="day",
        action=BudgetAction.BLOCK,
    )
    
    results = {"success": 0, "exceeded": 0, "errors": []}
    lock = threading.Lock()
    
    def simulate_pod(pod_id: int, requests: int):
        """Simulate a pod making requests."""
        # Each "pod" has its own tracker instance
        tracker = CostTracker(
            backend=backend_url,
            budgets=[budget],
            budget_mode="distributed",
        )
        
        for i in range(requests):
            try:
                tracker.record(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                    tags={"pod": f"pod-{pod_id}", "request": str(i)},
                )
                with lock:
                    results["success"] += 1
            except BudgetExceededError:
                with lock:
                    results["exceeded"] += 1
            except Exception as e:
                with lock:
                    results["errors"].append(str(e))
        
        tracker.close()
    
    # Simulate 4 pods making concurrent requests
    print("\n1. Simulating 4 pods with 25 requests each...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(simulate_pod, pod_id, 25)
            for pod_id in range(4)
        ]
        for f in futures:
            f.result()
    
    print(f"\n2. Results:")
    print(f"   ✅ Successful requests: {results['success']}")
    print(f"   🚫 Budget exceeded: {results['exceeded']}")
    if results["errors"]:
        print(f"   ❌ Errors: {len(results['errors'])}")
    
    total = results["success"] + results["exceeded"]
    print(f"\n3. Total requests processed: {total}")
    print(f"   Budget enforcement worked across all pods!")


def example_budget_reservation():
    """Demonstrate budget reservation for streaming/long requests."""
    print("\n" + "=" * 60)
    print("Budget Reservation (Pessimistic Locking)")
    print("=" * 60)
    
    if not check_redis_connection():
        print("\n⚠️  Redis not available. Showing conceptual example.")
        print("""
   Budget reservation is useful for streaming requests where
   you don't know the final token count upfront.
   
   Flow:
   1. Reserve estimated budget before LLM call
   2. Make the LLM call
   3. Finalize with actual cost (release reservation + record actual)
   
   This prevents overspending when multiple requests are in-flight.
   """)
        return
    
    from llm_cost_guard.backends.redis_backend import RedisBackend
    
    backend = RedisBackend(url="redis://localhost:6379/2")
    
    print("\n1. Simulating streaming request with reservation...")
    
    # Reserve budget before making call
    reservation_id = f"stream-{datetime.now().timestamp()}"
    estimated_cost = 0.05  # Estimate for max 4096 output tokens
    
    allowed, effective = backend.reserve_budget(
        budget_name="streaming",
        estimated_amount=estimated_cost,
        limit=1.00,
        reservation_id=reservation_id,
        period_seconds=86400,
    )
    
    if not allowed:
        print("   ❌ Budget reservation denied!")
        backend.close()
        return
    
    print(f"   ✅ Reserved ${estimated_cost:.4f}")
    print(f"   Effective spending: ${effective:.4f}")
    
    # Simulate streaming (actual cost is less than estimated)
    print("\n2. Simulating streaming response...")
    time.sleep(0.5)  # Simulate streaming delay
    actual_cost = 0.03  # Actual was less than estimated
    
    # Finalize reservation
    print("\n3. Finalizing with actual cost...")
    new_total = backend.finalize_budget(
        budget_name="streaming",
        reserved_amount=estimated_cost,
        actual_amount=actual_cost,
        reservation_id=reservation_id,
        period_seconds=86400,
    )
    
    print(f"   Actual cost: ${actual_cost:.4f}")
    print(f"   New total spending: ${new_total:.4f}")
    print(f"   Saved ${estimated_cost - actual_cost:.4f} from reservation")
    
    backend.close()


def example_failover_and_recovery():
    """Demonstrate graceful degradation when Redis fails."""
    print("\n" + "=" * 60)
    print("Failover and Recovery")
    print("=" * 60)
    
    from llm_cost_guard import CostTracker, Budget, BudgetAction
    
    # Intentionally use wrong Redis URL to trigger fallback
    print("\n1. Connecting to unavailable Redis...")
    
    tracker = CostTracker(
        backend="redis://localhost:9999",  # Wrong port
        budgets=[
            Budget(name="fallback-test", limit=100.00, action=BudgetAction.WARN),
        ],
        on_tracking_failure="fallback",  # Fall back to memory
    )
    
    print("\n2. Checking fallback status...")
    metrics = tracker.get_metrics()
    print(f"   Using fallback: {metrics['using_fallback']}")
    print(f"   Fallback activations: {metrics['fallback_activations']}")
    
    # Tracking still works with fallback
    print("\n3. Recording with fallback backend...")
    record = tracker.record(
        provider="openai",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
    )
    print(f"   ✅ Record saved: ${record.total_cost:.4f}")
    
    # Health check shows degraded state
    print("\n4. Health check...")
    health = tracker.health_check()
    print(f"   Healthy: {health.healthy}")
    print(f"   Errors: {health.errors}")
    
    # Audit logs the fallback
    print("\n5. Audit shows fallback event...")
    from llm_cost_guard.audit import AuditEventType
    fallback_events = tracker.audit.query(event_type=AuditEventType.FALLBACK_ACTIVATED)
    print(f"   Fallback events logged: {len(fallback_events)}")
    
    tracker.close()


def example_observability_metrics():
    """Demonstrate metrics for monitoring and alerting."""
    print("\n" + "=" * 60)
    print("Observability Metrics")
    print("=" * 60)
    
    from llm_cost_guard import CostTracker, Budget, BudgetAction
    from llm_cost_guard.exceptions import BudgetExceededError
    
    if not check_redis_connection():
        backend_url = "memory"
    else:
        backend_url = "redis://localhost:6379/3"
    
    tracker = CostTracker(
        backend=backend_url,
        budgets=[
            Budget(name="monitored", limit=0.10, action=BudgetAction.BLOCK),
        ],
    )
    
    # Generate some activity
    print("\n1. Generating activity...")
    for i in range(10):
        try:
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        except BudgetExceededError:
            pass
    
    # Get metrics
    print("\n2. Metrics for Prometheus/DataDog:")
    metrics = tracker.get_metrics()
    print(f"""
   llm_cost_guard_backend_failures: {metrics['backend_failures']}
   llm_cost_guard_fallback_activations: {metrics['fallback_activations']}
   llm_cost_guard_budget_checks: {metrics['budget_checks']}
   llm_cost_guard_budget_exceeded: {metrics['budget_exceeded_count']}
   llm_cost_guard_rate_limit_exceeded: {metrics['rate_limit_exceeded_count']}
   llm_cost_guard_tracking_errors: {metrics['tracking_errors']}
   llm_cost_guard_using_fallback: {1 if metrics['using_fallback'] else 0}
   """)
    
    # Health check for K8s readiness probe
    print("3. Health check for K8s probes:")
    health = tracker.health_check()
    print(f"""
   healthy: {health.healthy}
   backend_connected: {health.backend_connected}
   pricing_fresh: {health.pricing_fresh}
   """)
    
    tracker.close()


if __name__ == "__main__":
    print("\nLLM Cost Guard - Redis Distributed Budget Examples")
    print("=" * 60)
    
    if not check_redis_connection():
        print("\n⚠️  Redis is not running.")
        print("   Start Redis with: docker run -p 6379:6379 redis:alpine")
        print("   Some examples will use memory backend as fallback.\n")
    
    example_distributed_budget_basic()
    example_concurrent_instances()
    example_budget_reservation()
    example_failover_and_recovery()
    example_observability_metrics()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
