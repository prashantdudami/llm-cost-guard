# LLM Cost Guard

[![PyPI version](https://badge.fury.io/py/llm-cost-guard.svg)](https://badge.fury.io/py/llm-cost-guard)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time cost tracking, budget enforcement, and usage analytics for LLM applications. Supports OpenAI, Anthropic, AWS Bedrock, and more.

## Features

- **Real-time Cost Tracking**: Track costs as they happen, not when the bill arrives
- **Budget Enforcement**: Set limits with configurable actions (warn, throttle, block)
- **Multi-Provider Support**: OpenAI, Anthropic, AWS Bedrock, Google Vertex AI
- **LangChain Integration**: Native callback support for LangChain applications
- **Rate Limiting**: Control request rates per model, provider, or custom tags
- **Hierarchical Tracking**: Group related LLM calls with spans
- **Flexible Storage**: In-memory, SQLite, PostgreSQL, Redis, DynamoDB backends
- **Zero External Dependencies**: Works offline with no external services required

## Installation

```bash
pip install llm-cost-guard
```

With optional integrations:

```bash
# LangChain support
pip install llm-cost-guard[langchain]

# AWS Bedrock support
pip install llm-cost-guard[bedrock]

# All optional dependencies
pip install llm-cost-guard[all]
```

## Quick Start

### Basic Usage

```python
from llm_cost_guard import CostTracker

tracker = CostTracker()

# Decorator-based tracking
@tracker.track
def my_llm_call():
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    return response

result = my_llm_call()

# Check costs
print(tracker.last_call().total_cost)  # $0.0015
```

### With Budget Enforcement

```python
from llm_cost_guard import CostTracker, Budget, BudgetAction

tracker = CostTracker(
    budgets=[
        Budget(
            name="daily",
            limit=10.00,
            period="day",
            action=BudgetAction.WARN
        ),
        Budget(
            name="monthly",
            limit=500.00,
            period="month",
            action=BudgetAction.BLOCK
        ),
    ]
)

# Get notified when approaching limits
@tracker.on_budget_warning
def handle_warning(budget, current):
    print(f"Warning: Budget '{budget.name}' at {current/budget.limit*100:.0f}%")

@tracker.on_budget_exceeded
def handle_exceeded(budget):
    print(f"Budget '{budget.name}' exceeded!")
```

### Manual Recording

```python
# For custom integrations
record = tracker.record(
    provider="openai",
    model="gpt-4o",
    input_tokens=1234,
    output_tokens=567,
    tags={"team": "search", "feature": "autocomplete"}
)

print(record.total_cost)  # $0.0208
```

### Wrapped Clients

```python
from llm_cost_guard import CostTracker
from llm_cost_guard.clients import TrackedOpenAI

tracker = CostTracker()
client = TrackedOpenAI(tracker=tracker)

# Automatic tracking - no decorators needed
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### LangChain Integration

```python
from llm_cost_guard import CostTracker
from llm_cost_guard.integrations.langchain import CostTrackingCallback

tracker = CostTracker()

llm = ChatOpenAI(
    model="gpt-4o",
    callbacks=[CostTrackingCallback(tracker)]
)

result = llm.invoke("Hello!")
print(tracker.last_call().total_cost)
```

### Hierarchical Tracking (Spans)

```python
# Track costs for complex operations like agents
with tracker.span("customer_support_agent", tags={"user_id": "123"}) as span:
    result = agent.invoke(query)
    
    print(span.total_cost)      # $0.45 (sum of all calls)
    print(span.call_count)      # 5
    print(span.models_used)     # ["gpt-4o", "gpt-3.5-turbo"]
```

## Configuration

### Storage Backends

```python
# In-memory (default, development)
tracker = CostTracker(backend="memory")

# SQLite (single-machine persistence)
tracker = CostTracker(backend="sqlite:///costs.db")

# PostgreSQL (production)
tracker = CostTracker(backend="postgresql://user:pass@host/db")

# Redis (distributed, real-time)
tracker = CostTracker(backend="redis://localhost:6379/0")
```

### Rate Limiting

```python
from llm_cost_guard import CostTracker, RateLimit

tracker = CostTracker(
    rate_limits=[
        RateLimit(
            name="requests-per-minute",
            limit=100,
            period="minute",
            scope="global"
        ),
        RateLimit(
            name="user-requests",
            limit=10,
            period="minute",
            scope="tag:user_id"
        )
    ]
)
```

### Fail-Safe Modes

```python
tracker = CostTracker(
    # Block LLM calls if tracking fails (strict)
    on_tracking_failure="block",
    
    # Allow LLM calls but log warning (available)
    # on_tracking_failure="allow",
    
    # Use in-memory fallback temporarily
    # on_tracking_failure="fallback",
)
```

## CLI

```bash
# View current costs
llm-cost-guard status

# Generate report
llm-cost-guard report --period day --group-by model

# Check health
llm-cost-guard health

# List supported models and pricing
llm-cost-guard models --provider openai

# Export data
llm-cost-guard export --format csv --output costs.csv
```

## Supported Providers

| Provider | Models |
|----------|--------|
| OpenAI | GPT-4o, GPT-4, GPT-3.5, o1, Embeddings, DALL-E |
| Anthropic | Claude 3.5, Claude 3, Claude 2 |
| AWS Bedrock | Claude, Titan, Llama, Mistral, Cohere |
| Google Vertex AI | Gemini 1.5, Gemini 1.0, PaLM 2 |

## Reporting

```python
# Daily summary
tracker.daily_report()

# Cost by model
tracker.report_by_model(period="week")

# Query with filters
report = tracker.get_costs(
    start_date="2024-01-01",
    end_date="2024-01-31",
    tags={"team": "search"},
    group_by=["model", "feature"]
)

# Export to DataFrame
df = tracker.to_dataframe()
```

## Security

- **No API key logging**: Keys are never stored, logged, or transmitted
- **No prompt storage by default**: Only metadata (tokens, cost) stored
- **PII redaction**: Optional redaction for user IDs
- **Encryption support**: For SQL/Redis backends

```python
tracker = CostTracker(
    store_prompts=False,          # Default: never store prompts
    redact_user_ids=True,         # Hash user IDs in storage
)
```

## Audit Logging (v0.2.0+)

Enterprise-ready audit trails for compliance:

```python
from llm_cost_guard import CostTracker, FileAuditBackend

# Enable audit logging
tracker = CostTracker(
    audit_enabled=True,
    audit_backend=FileAuditBackend("audit.log"),
)

# Query audit history
events = tracker.audit.query(
    event_type=AuditEventType.BUDGET_EXCEEDED,
    start_date="2024-01-01",
)

# Get budget-specific history
history = tracker.audit.get_budget_history("daily")
```

Audit events include:
- Budget created/modified/deleted
- Budget warnings and exceeded events
- Rate limit exceeded events
- Tracking failures and fallback activations

## Observability Metrics (v0.2.0+)

Track health and degradation:

```python
# Get tracker metrics
metrics = tracker.get_metrics()
print(metrics)
# {
#   "backend_failures": 0,
#   "fallback_activations": 0,
#   "budget_exceeded_count": 3,
#   "tracking_errors": 0,
#   "using_fallback": False,
# }

# Health check
health = tracker.health_check()
print(health.healthy)  # True/False
print(health.errors)   # List of issues
```

## Custom Pricing

For negotiated enterprise rates:

```python
tracker = CostTracker(
    pricing_overrides={
        "openai/gpt-4": {
            "input_cost_per_1k": 0.02,    # Your negotiated rate
            "output_cost_per_1k": 0.04,
        }
    }
)
```

## Current Limitations

Being transparent about what's not yet production-ready:

| Feature | Status | Notes |
|---------|--------|-------|
| Distributed budgets (Redis) | ✅ v0.2.0 | Atomic operations with Lua scripts |
| Audit logging | ✅ v0.2.0 | File and logging backends |
| Graceful degradation metrics | ✅ v0.2.0 | Track failures and fallbacks |
| PostgreSQL backend | 🚧 Planned | Use SQLite or Redis for now |
| DynamoDB backend | 🚧 Planned | Use SQLite or Redis for now |
| Encryption at rest | 🚧 Planned | Use encrypted volumes as workaround |
| Multi-tenancy optimization | 🚧 Planned | Use tag-scoped budgets for now |
| Streaming cost estimation | ⚠️ Limited | Actual cost tracked on completion |
| Fine-tuning cost tracking | ❌ Not supported | |

### Recommended for Production

| Deployment Size | Backend | Notes |
|-----------------|---------|-------|
| Single instance | SQLite | Simple, no setup |
| Multiple instances | Redis | Distributed budget enforcement |
| High-volume (>1k req/s) | Redis | With sampling (coming soon) |

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Prashant Dudami** - AI/ML Architect & LLM Infrastructure Expert
- Website: [prashantdudami.com](https://www.prashantdudami.com/)
- LinkedIn: [linkedin.com/in/prashantdudami](https://www.linkedin.com/in/prashantdudami/)
- GitHub: [github.com/prashantdudami](https://github.com/prashantdudami)
