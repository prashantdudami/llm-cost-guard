"""
Audit Logging Example

Demonstrates enterprise-ready audit trails for compliance with llm-cost-guard.
Audit logging tracks all budget events, configuration changes, and security-relevant
operations for regulatory compliance (SOC2, HIPAA, etc.).

Install:
    pip install llm-cost-guard

Run:
    python audit_logging.py
"""

from datetime import datetime, timedelta

from llm_cost_guard import (
    CostTracker,
    Budget,
    BudgetAction,
    AuditLogger,
    AuditEventType,
    FileAuditBackend,
    LoggingAuditBackend,
)


def example_basic_audit_logging():
    """Basic audit logging with default logging backend."""
    print("\n" + "=" * 60)
    print("Basic Audit Logging")
    print("=" * 60)
    
    # Create tracker with audit logging enabled (default)
    tracker = CostTracker(
        backend="memory",
        budgets=[
            Budget(
                name="daily",
                limit=10.00,
                period="day",
                action=BudgetAction.WARN,
                warning_threshold=0.5,
            ),
        ],
        audit_enabled=True,  # Default is True
    )
    
    print("\n1. Budget created (logged to audit)")
    
    # Make some API calls
    print("\n2. Making API calls...")
    for i in range(5):
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000 * (i + 1),
            output_tokens=500 * (i + 1),
            tags={"user": f"user_{i % 3}"},
        )
    
    # Query audit events
    print("\n3. Querying audit events...")
    events = tracker.audit.query()
    print(f"   Total audit events: {len(events)}")
    
    for event in events[-5:]:  # Last 5 events
        print(f"   - {event.event_type.value}: {event.resource or 'N/A'}")
    
    tracker.close()


def example_file_audit_logging():
    """File-based audit logging for compliance."""
    print("\n" + "=" * 60)
    print("File-based Audit Logging (Compliance)")
    print("=" * 60)
    
    import tempfile
    import os
    
    # Create a temporary audit log file
    audit_file = tempfile.mktemp(suffix=".jsonl")
    print(f"\n1. Audit file: {audit_file}")
    
    # Create file audit backend
    audit_backend = FileAuditBackend(audit_file)
    
    tracker = CostTracker(
        backend="memory",
        budgets=[
            Budget(
                name="project-alpha",
                limit=100.00,
                period="month",
                action=BudgetAction.BLOCK,
            ),
        ],
        audit_backend=audit_backend,
    )
    
    print("\n2. Recording API calls...")
    for i in range(3):
        tracker.record(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            input_tokens=500,
            output_tokens=200,
            tags={"project": "alpha", "team": "backend"},
        )
    
    # Read the audit file
    print("\n3. Audit file contents:")
    with open(audit_file, "r") as f:
        for line in f:
            print(f"   {line.strip()[:100]}...")
    
    tracker.close()
    os.unlink(audit_file)


def example_audit_event_callbacks():
    """Register callbacks for audit events."""
    print("\n" + "=" * 60)
    print("Audit Event Callbacks")
    print("=" * 60)
    
    # Custom handler for exceeded events
    def on_budget_exceeded(event):
        print(f"\n   [ALERT] Budget exceeded: {event.resource}")
        print(f"   Current: ${event.details['current_spending']:.4f}")
        print(f"   Limit: ${event.details['limit']:.4f}")
        # In production: send to Slack, PagerDuty, etc.
    
    # Custom handler for warnings
    def on_budget_warning(event):
        print(f"\n   [WARNING] Budget warning: {event.resource}")
        print(f"   Utilization: {event.details['utilization_percent']:.1f}%")
    
    tracker = CostTracker(
        backend="memory",
        budgets=[
            Budget(
                name="tight-budget",
                limit=0.05,  # Very low limit
                period="day",
                action=BudgetAction.BLOCK,
                warning_threshold=0.3,
            ),
        ],
    )
    
    # Register callbacks
    tracker.audit.on_event(AuditEventType.BUDGET_EXCEEDED, on_budget_exceeded)
    tracker.audit.on_event(AuditEventType.BUDGET_WARNING, on_budget_warning)
    
    print("\n1. Making API calls until budget exceeded...")
    try:
        for i in range(20):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
    except Exception as e:
        print(f"\n   Budget enforcement: {e}")
    
    tracker.close()


def example_audit_history_query():
    """Query audit history for a specific budget."""
    print("\n" + "=" * 60)
    print("Audit History Query")
    print("=" * 60)
    
    tracker = CostTracker(
        backend="memory",
        budgets=[
            Budget(name="team-search", limit=50.00, action=BudgetAction.WARN),
            Budget(name="team-chat", limit=100.00, action=BudgetAction.WARN),
        ],
    )
    
    # Make calls for different teams
    print("\n1. Simulating team activity...")
    for team, model in [("search", "gpt-4o"), ("chat", "gpt-4o-mini")]:
        for _ in range(5):
            tracker.record(
                provider="openai",
                model=model,
                input_tokens=500,
                output_tokens=200,
                tags={"team": team},
            )
    
    # Query history for specific budget
    print("\n2. Audit history for 'team-search' budget:")
    history = tracker.audit.get_budget_history("team-search")
    for event in history:
        print(f"   - {event.timestamp.strftime('%H:%M:%S')} | {event.event_type.value}")
    
    print("\n3. Audit history for 'team-chat' budget:")
    history = tracker.audit.get_budget_history("team-chat")
    for event in history:
        print(f"   - {event.timestamp.strftime('%H:%M:%S')} | {event.event_type.value}")
    
    tracker.close()


def example_compliance_report():
    """Generate compliance report from audit logs."""
    print("\n" + "=" * 60)
    print("Compliance Report Generation")
    print("=" * 60)
    
    tracker = CostTracker(
        backend="memory",
        budgets=[
            Budget(name="daily", limit=100.00, action=BudgetAction.WARN),
        ],
    )
    
    # Simulate activity
    print("\n1. Simulating day's activity...")
    for i in range(20):
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100 * (i + 1),
            output_tokens=50 * (i + 1),
        )
    
    # Generate compliance report
    print("\n2. Generating compliance report...")
    
    # Get all events for today
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    events = tracker.audit.query(start_date=today_start)
    
    # Categorize events
    budget_events = [e for e in events if e.event_type.value.startswith("budget.")]
    
    print(f"""
   =============================================
   LLM COST GUARD - DAILY COMPLIANCE REPORT
   Date: {datetime.now().strftime('%Y-%m-%d')}
   =============================================
   
   Total Audit Events: {len(events)}
   
   Budget Events:
   - Created:  {len([e for e in budget_events if e.event_type == AuditEventType.BUDGET_CREATED])}
   - Warnings: {len([e for e in budget_events if e.event_type == AuditEventType.BUDGET_WARNING])}
   - Exceeded: {len([e for e in budget_events if e.event_type == AuditEventType.BUDGET_EXCEEDED])}
   
   =============================================
   """)
    
    tracker.close()


if __name__ == "__main__":
    print("\nLLM Cost Guard - Audit Logging Examples")
    print("=" * 60)
    
    example_basic_audit_logging()
    example_file_audit_logging()
    example_audit_event_callbacks()
    example_audit_history_query()
    example_compliance_report()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
