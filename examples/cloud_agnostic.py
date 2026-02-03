"""
Cloud-Agnostic Features Example

Demonstrates cloud-agnostic patterns for:
- Encryption (Fernet - local, no cloud dependency)
- Secrets management (ENV, files, Vault)
- Resilience (circuit breaker, retry)
- Metrics export (Prometheus, StatsD)

These patterns work on any cloud provider or on-premises.

Install:
    pip install llm-cost-guard[full]  # All cloud-agnostic features
    
Run:
    python cloud_agnostic.py
"""

import os
import time
import tempfile


def example_encryption():
    """Demonstrate encryption providers."""
    print("\n" + "=" * 60)
    print("Encryption Providers (Cloud-Agnostic)")
    print("=" * 60)
    
    from llm_cost_guard import (
        NoEncryption,
        FernetEncryption,
        FieldEncryption,
        get_encryption_provider,
    )
    
    # 1. No encryption (development)
    print("\n1. No Encryption (Development)")
    provider = get_encryption_provider("none")
    encrypted = provider.encrypt_string("sensitive data")
    print(f"   Original: 'sensitive data'")
    print(f"   Encrypted: '{encrypted}'")  # Same as original
    
    # 2. Fernet encryption (production - local keys)
    print("\n2. Fernet Encryption (Local Keys)")
    try:
        key = FernetEncryption.generate_key()
        print(f"   Generated key: {key[:20]}...")
        
        provider = FernetEncryption(key)
        encrypted = provider.encrypt_string("sensitive data")
        decrypted = provider.decrypt_string(encrypted)
        
        print(f"   Original: 'sensitive data'")
        print(f"   Encrypted: '{encrypted[:40]}...'")
        print(f"   Decrypted: '{decrypted}'")
    except ImportError:
        print("   [Skip] cryptography package not installed")
    
    # 3. Field-level encryption
    print("\n3. Field-Level Encryption")
    try:
        key = FernetEncryption.generate_key()
        field_enc = FieldEncryption(
            provider=FernetEncryption(key),
            encrypted_fields=["metadata"],
            hashed_fields=["user_id"],
            hash_salt="production-salt",
        )
        
        record = {
            "model": "gpt-4o",
            "cost": 0.05,
            "metadata": {"prompt": "What is the meaning of life?"},
            "user_id": "user@company.com",
        }
        
        encrypted_record = field_enc.encrypt_record(record)
        print(f"   Original user_id: '{record['user_id']}'")
        print(f"   Hashed user_id: '{encrypted_record['user_id'][:40]}...'")
        print(f"   Metadata encrypted: {encrypted_record.get('_metadata_encrypted')}")
        
        decrypted_record = field_enc.decrypt_record(encrypted_record)
        print(f"   Decrypted metadata: {decrypted_record['metadata']}")
    except ImportError:
        print("   [Skip] cryptography package not installed")


def example_secrets():
    """Demonstrate secrets providers."""
    print("\n" + "=" * 60)
    print("Secrets Providers (Cloud-Agnostic)")
    print("=" * 60)
    
    from llm_cost_guard import (
        EnvironmentSecretsProvider,
        FileSecretsProvider,
        get_secrets_provider,
    )
    from llm_cost_guard.secrets import CompositeSecretsProvider
    
    # 1. Environment variables
    print("\n1. Environment Variables")
    os.environ["LLM_COST_GUARD_API_KEY"] = "sk-test-123"
    
    provider = EnvironmentSecretsProvider(prefix="LLM_COST_GUARD_")
    api_key = provider.get_secret("API_KEY")
    print(f"   LLM_COST_GUARD_API_KEY: '{api_key}'")
    
    del os.environ["LLM_COST_GUARD_API_KEY"]
    
    # 2. File-based secrets (K8s style)
    print("\n2. File-Based Secrets (Kubernetes Style)")
    with tempfile.TemporaryDirectory() as secrets_dir:
        # Create secret files
        with open(os.path.join(secrets_dir, "redis-password"), "w") as f:
            f.write("super-secret-password")
        with open(os.path.join(secrets_dir, "db-connection-string"), "w") as f:
            f.write("postgresql://user:pass@host/db")
        
        provider = FileSecretsProvider(secrets_dir)
        
        redis_pwd = provider.get_secret("redis-password")
        db_conn = provider.get_secret("db-connection-string")
        
        print(f"   redis-password: '{redis_pwd}'")
        print(f"   db-connection-string: '{db_conn}'")
    
    # 3. Composite provider (fallback chain)
    print("\n3. Composite Provider (Fallback Chain)")
    os.environ["FALLBACK_TEST"] = "from-environment"
    
    with tempfile.TemporaryDirectory() as secrets_dir:
        with open(os.path.join(secrets_dir, "file-only-secret"), "w") as f:
            f.write("from-file")
        
        provider = CompositeSecretsProvider([
            EnvironmentSecretsProvider(),
            FileSecretsProvider(secrets_dir),
        ])
        
        # Found in env
        env_value = provider.get_secret("FALLBACK_TEST")
        print(f"   FALLBACK_TEST (from env): '{env_value}'")
        
        # Falls back to file
        file_value = provider.get_secret("file-only-secret")
        print(f"   file-only-secret (from file): '{file_value}'")
    
    del os.environ["FALLBACK_TEST"]


def example_resilience():
    """Demonstrate resilience patterns."""
    print("\n" + "=" * 60)
    print("Resilience Patterns (Cloud-Agnostic)")
    print("=" * 60)
    
    from llm_cost_guard import (
        CircuitBreaker,
        CircuitState,
        CircuitOpenError,
        retry_with_backoff,
        ResilientOperation,
        RetryConfig,
    )
    
    # 1. Retry with exponential backoff
    print("\n1. Retry with Exponential Backoff")
    
    attempt_count = 0
    
    @retry_with_backoff(
        max_attempts=3,
        initial_delay=0.1,
        max_delay=1.0,
        jitter=True,
    )
    def flaky_api_call():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            print(f"   Attempt {attempt_count}: Failed (transient error)")
            raise ConnectionError("Transient failure")
        print(f"   Attempt {attempt_count}: Success!")
        return {"status": "ok"}
    
    result = flaky_api_call()
    print(f"   Result: {result}")
    
    # 2. Circuit breaker
    print("\n2. Circuit Breaker")
    
    breaker = CircuitBreaker(
        failure_threshold=3,
        success_threshold=2,
        timeout=1.0,
        name="external-api",
    )
    
    print(f"   Initial state: {breaker.state.value}")
    
    # Simulate failures
    for i in range(3):
        breaker.record_failure()
        print(f"   After failure {i+1}: {breaker.state.value}")
    
    # Try to make request while open
    try:
        if not breaker.allow_request():
            print("   Request blocked (circuit open)")
    except CircuitOpenError:
        print("   Request blocked (circuit open)")
    
    # Wait for half-open
    print("   Waiting for timeout...")
    time.sleep(1.1)
    print(f"   After timeout: {breaker.state.value}")
    
    # Recover
    breaker.record_success()
    breaker.record_success()
    print(f"   After recovery: {breaker.state.value}")
    
    # 3. Combined resilience
    print("\n3. Combined Resilience (Retry + Circuit Breaker)")
    
    resilient = ResilientOperation(
        circuit_breaker=CircuitBreaker(failure_threshold=5),
        retry_config=RetryConfig(max_attempts=2, initial_delay=0.1),
    )
    
    @resilient
    def resilient_call():
        return "Success with full resilience!"
    
    result = resilient_call()
    print(f"   Result: {result}")


def example_metrics():
    """Demonstrate metrics exporters."""
    print("\n" + "=" * 60)
    print("Metrics Export (Cloud-Agnostic)")
    print("=" * 60)
    
    from llm_cost_guard import (
        NoOpExporter,
        LoggingExporter,
        get_metrics_exporter,
    )
    from llm_cost_guard.metrics import TrackerMetrics
    
    # 1. Logging exporter (development)
    print("\n1. Logging Exporter (Development)")
    
    import logging
    logging.basicConfig(level=logging.INFO)
    
    exporter = LoggingExporter(level=logging.INFO)
    exporter.counter(TrackerMetrics.REQUESTS_TOTAL, 1, {"model": "gpt-4o"})
    exporter.gauge(TrackerMetrics.BUDGET_UTILIZATION, 0.75, {"budget": "daily"})
    exporter.histogram(TrackerMetrics.COST_PER_REQUEST, 0.05)
    exporter.timing(TrackerMetrics.REQUEST_LATENCY, 150.5)
    
    # 2. Prometheus exporter
    print("\n2. Prometheus Exporter (Production)")
    try:
        from llm_cost_guard.metrics import PrometheusExporter
        
        exporter = get_metrics_exporter("prometheus", port=9999)
        print("   Prometheus exporter created (port 9999)")
        print("   Would start server with: exporter.start_server()")
        
        # Record some metrics
        exporter.counter("test_requests", 1, {"model": "gpt-4"})
        exporter.gauge("test_utilization", 0.8)
        print("   Metrics recorded successfully")
    except ImportError:
        print("   [Skip] prometheus_client not installed")
    
    # 3. Standard metric names
    print("\n3. Standard Metric Names")
    print(f"   Counters:")
    print(f"     - {TrackerMetrics.REQUESTS_TOTAL}")
    print(f"     - {TrackerMetrics.COST_DOLLARS_TOTAL}")
    print(f"     - {TrackerMetrics.BUDGET_EXCEEDED_TOTAL}")
    print(f"   Gauges:")
    print(f"     - {TrackerMetrics.BUDGET_UTILIZATION}")
    print(f"     - {TrackerMetrics.BACKEND_HEALTHY}")
    print(f"   Histograms:")
    print(f"     - {TrackerMetrics.COST_PER_REQUEST}")
    print(f"     - {TrackerMetrics.REQUEST_LATENCY}")


def example_full_integration():
    """Full integration example with all patterns."""
    print("\n" + "=" * 60)
    print("Full Integration Example")
    print("=" * 60)
    
    from llm_cost_guard import CostTracker, Budget, BudgetAction
    
    # Setup with cloud-agnostic patterns
    tracker = CostTracker(
        backend="memory",  # Works everywhere
        budgets=[
            Budget(name="daily", limit=100.0, action=BudgetAction.WARN),
        ],
        audit_enabled=True,  # Audit logging
    )
    
    print("\n1. Recording LLM calls...")
    for i in range(5):
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100 * (i + 1),
            output_tokens=50 * (i + 1),
            tags={"team": "engineering"},
        )
        print(f"   Call {i+1}: ${record.total_cost:.4f}")
    
    print("\n2. Getting metrics...")
    metrics = tracker.get_metrics()
    print(f"   Budget checks: {metrics['budget_checks']}")
    print(f"   Backend failures: {metrics['backend_failures']}")
    
    print("\n3. Health check...")
    health = tracker.health_check()
    print(f"   Healthy: {health.healthy}")
    print(f"   Backend connected: {health.backend_connected}")
    
    print("\n4. Cost report...")
    report = tracker.daily_report()
    print(f"   Total calls: {report.total_calls}")
    print(f"   Total cost: ${report.total_cost:.4f}")
    
    tracker.close()


if __name__ == "__main__":
    print("\nLLM Cost Guard - Cloud-Agnostic Features")
    print("=" * 60)
    print("These patterns work on ANY cloud or on-premises!")
    
    example_encryption()
    example_secrets()
    example_resilience()
    example_metrics()
    example_full_integration()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
