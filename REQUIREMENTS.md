# Instructions for implementation
1. You are an experienced AI/ML Architect and a python developer 
2. LLM Cost Guard requirements are listed below
3. Implement these requirements.
4. Write unit tests, Run unit tests, fix failing tests.
5. Write end to end tests, Run end to end tests, fix failing tests.
6. Write sample code for users with examples.


# LLM Cost Guard - Requirements Document

## 1. Executive Summary

**LLM Cost Guard** is a Python library that provides real-time cost tracking, budget enforcement, and usage analytics for Large Language Model (LLM) applications. It supports multiple providers (AWS Bedrock, Google Vertex AI, OpenAI, Anthropic, Azure OpenAI) and integrates seamlessly with LangChain.

### Problem Statement

Organizations using LLMs face significant challenges:
- **No visibility** into real-time costs until monthly bills arrive
- **Runaway costs** from bugs, infinite loops, or unexpected traffic
- **Attribution difficulty** - which feature/team/user is driving costs?
- **Budget overruns** with no automated safeguards
- **Manual tracking** is error-prone and time-consuming

### Solution

A lightweight, decorator-based library that:
- Tracks costs in real-time across all major LLM providers
- Enforces budgets with configurable actions (warn, throttle, block)
- Provides detailed attribution and analytics
- Integrates with existing observability tools

---

## 2. Goals & Success Metrics

### Primary Goals

| Goal | Success Metric |
|------|----------------|
| Cost visibility | <1% variance from actual provider bills |
| Budget protection | Zero unintentional budget overruns |
| Easy integration | <10 lines of code to integrate |
| Performance | <5ms overhead per LLM call |

### Secondary Goals

- Support for 5+ LLM providers
- LangChain native integration
- Export to common observability tools (Prometheus, DataDog, CloudWatch)

---

## 3. Target Users

| User Type | Needs |
|-----------|-------|
| **Individual Developers** | Track personal API usage, set spending limits |
| **Startup Teams** | Attribute costs to features, prevent runaway spending |
| **Enterprise** | Multi-tenant tracking, compliance reporting, budget allocation |

---

## 4. Supported Providers

### Phase 1 (MVP)

| Provider | Models | Pricing Source |
|----------|--------|----------------|
| AWS Bedrock | Claude, Titan, Llama, Mistral | AWS Pricing API |
| OpenAI | GPT-4, GPT-4o, GPT-3.5-turbo | OpenAI Pricing Page |
| Anthropic | Claude 3.5, Claude 3, Claude 2 | Anthropic Pricing |

### Phase 2

| Provider | Models |
|----------|--------|
| Google Vertex AI | Gemini 1.5, PaLM 2 |
| Azure OpenAI | GPT-4, GPT-3.5 |
| Cohere | Command, Embed |

### Phase 3

| Provider | Models |
|----------|--------|
| Mistral AI | Mistral Large, Medium, Small |
| Groq | Llama, Mixtral |
| Together AI | Various open-source models |

### 4.1 Model Types Support

| Model Type | Cost Basis | Fields |
|------------|------------|--------|
| Chat/Completion | Input + Output tokens | `input_tokens`, `output_tokens` |
| Embeddings | Input tokens only | `embedding_tokens`, `dimensions` |
| Image Generation | Per image | `image_size`, `image_count`, `quality` |
| Audio Transcription | Per minute | `audio_duration_seconds` |
| Fine-tuning | Training tokens | `training_tokens`, `epochs` |

---

## 5. Security Requirements

### 5.1 Data Protection

| Requirement | Implementation |
|-------------|----------------|
| **No API key logging** | Keys are never stored, logged, or transmitted |
| **No prompt storage by default** | Only metadata (tokens, cost) stored |
| **PII redaction** | Optional redaction for user IDs in logs |
| **Encryption at rest** | Required for SQL/Redis backends |
| **TLS in transit** | Required for all backend connections |

```python
tracker = CostTracker(
    # Security settings
    store_prompts=False,          # Default: never store prompts
    redact_user_ids=True,         # Hash user IDs in storage
    encryption_key="...",         # Encrypt sensitive data at rest
)
```

### 5.2 Credential Management

```python
# Supported credential sources (never hardcode!)
tracker = CostTracker(
    backend="postgresql://...",
    credentials_source="env",              # Environment variables
    # credentials_source="aws_secrets",    # AWS Secrets Manager
    # credentials_source="vault",          # HashiCorp Vault
    # credentials_source="gcp_secrets",    # GCP Secret Manager
)

# For AWS backends, use IAM roles
tracker = CostTracker(
    backend="dynamodb://cost-tracking",
    aws_auth="iam_role",  # Use instance/task role, not access keys
)
```

### 5.3 Fail-Safe Modes

**Critical**: What happens when tracking infrastructure fails?

```python
tracker = CostTracker(
    # Fail-closed: Block LLM calls if tracking fails (secure, strict)
    on_tracking_failure="block",
    
    # Fail-open: Allow LLM calls but log warning (available, risky)
    # on_tracking_failure="allow",
    
    # Fallback: Switch to in-memory tracking temporarily
    # on_tracking_failure="fallback",
)
```

| Mode | Behavior | Use Case |
|------|----------|----------|
| `block` | Raise `TrackingUnavailableError` | Strict budget enforcement |
| `allow` | Log warning, proceed without tracking | High availability priority |
| `fallback` | Use in-memory, sync later | Balanced approach |

### 5.4 Audit Logging

```python
tracker = CostTracker(
    audit_log=True,
    audit_destination="cloudwatch://llm-cost-guard-audit",
)

# Audit events:
# - Budget created/modified/deleted
# - Budget exceeded events
# - Configuration changes
# - Export/report generation
```

---

## 6. Core Features

### 6.1 Cost Tracking

#### Real-time Cost Calculation

```python
from llm_cost_guard import CostTracker

tracker = CostTracker()

# Decorator-based tracking
@tracker.track
def my_llm_call():
    response = openai.chat.completions.create(...)
    return response

# Context manager
with tracker.track_context(tags={"feature": "summarizer"}):
    response = bedrock.invoke_model(...)
```

#### Integration Methods

| Method | Pros | Cons | Recommended For |
|--------|------|------|-----------------|
| **Wrapper clients** | Reliable, explicit | Requires code changes | New projects |
| **Decorator** | Easy to add | Manual per function | Existing code |
| **Monkey-patching** | Zero code changes | Fragile, version-dependent | Quick testing |
| **LangChain callback** | Native integration | LangChain only | LangChain users |

```python
# Method 1: Wrapper client (RECOMMENDED)
from llm_cost_guard.clients import TrackedOpenAI

client = TrackedOpenAI(tracker=tracker)
response = client.chat.completions.create(...)

# Method 2: Decorator
@tracker.track
def my_call():
    return openai.chat.completions.create(...)

# Method 3: Monkey-patch (use with caution)
llm_cost_guard.auto_instrument()  # Patches all supported providers

# Method 4: LangChain callback
from llm_cost_guard.integrations import CostTrackingCallback
llm = ChatOpenAI(callbacks=[CostTrackingCallback(tracker)])
```

#### Token Counting

- Accurate token counting using provider-specific tokenizers
- Support for tiktoken (OpenAI), Anthropic tokenizer, etc.
- Fallback estimation for providers without public tokenizers

#### Cost Calculation

```python
# Per-call cost breakdown
result = tracker.last_call()
print(result.input_tokens)      # 1,234
print(result.output_tokens)     # 567
print(result.input_cost)        # $0.0123
print(result.output_cost)       # $0.0085
print(result.total_cost)        # $0.0208
print(result.model)             # "gpt-4o"
print(result.provider)          # "openai"
print(result.cached)            # False (prompt cache hit)
print(result.cache_savings)     # $0.00
```

### 6.2 Streaming Support

**Requirement**: Track costs for streaming responses accurately.

```python
@tracker.track(streaming=True)
async def streaming_call():
    async for chunk in openai.chat.completions.create(stream=True):
        yield chunk
    # Cost recorded on stream completion

# Budget estimation for streaming (before completion)
tracker = CostTracker(
    streaming_budget_mode="estimate",      # Estimate based on input + max_tokens
    streaming_max_output_estimate=4096,    # Assume max output for budget check
)
```

| Streaming Scenario | Behavior |
|--------------------|----------|
| Stream completes normally | Track actual tokens |
| User cancels mid-stream | Track tokens received so far |
| Connection timeout | Track input tokens + partial output |
| Budget exceeded mid-stream | Complete current stream, block next |

### 6.3 Retry and Failure Tracking

```python
tracker = CostTracker(
    track_failed_calls=True,      # Failed calls still cost tokens
    dedupe_window_seconds=5,       # Detect duplicate retry attempts
)

# Result includes failure info
result = tracker.last_call()
print(result.success)             # False
print(result.error_type)          # "RateLimitError"
print(result.tokens_charged)      # True (provider charged for attempt)
```

| Failure Type | Tokens Charged? | Track Cost? |
|--------------|-----------------|-------------|
| Rate limit (429) | Sometimes | Yes |
| Context too long | Yes (input) | Yes |
| Timeout after send | Yes | Yes |
| Connection error before send | No | No |
| Invalid API key | No | No |

### 6.4 Budget Enforcement

#### Budget Configuration

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
        Budget(
            name="per-request",
            limit=0.50,
            period="request",
            action=BudgetAction.WARN
        ),
        # Tag-scoped budgets
        Budget(
            name="team-search-daily",
            limit=50.00,
            period="day",
            action=BudgetAction.THROTTLE,
            tags={"team": "search"}  # Only applies to this team
        )
    ]
)
```

#### Budget Actions

| Action | Behavior |
|--------|----------|
| `WARN` | Log warning, continue execution |
| `THROTTLE` | Add delay, reduce request rate |
| `BLOCK` | Raise `BudgetExceededError` |
| `CALLBACK` | Call custom function |

#### Budget Events

```python
@tracker.on_budget_warning
def handle_warning(budget: Budget, current: float):
    slack.send(f"⚠️ Budget '{budget.name}' at {current/budget.limit*100:.0f}%")

@tracker.on_budget_exceeded
def handle_exceeded(budget: Budget):
    pagerduty.alert(f"🚨 Budget '{budget.name}' exceeded!")
```

### 6.5 Distributed Budget Enforcement

**Critical for production**: Budget enforcement across multiple instances.

```python
tracker = CostTracker(
    backend="redis://...",
    
    # Distributed budget mode
    budget_mode="distributed",        # vs "local" (single instance)
    budget_sync_interval_ms=100,      # Sync frequency with central store
    budget_reservation="pessimistic", # Reserve budget before call
)
```

| Strategy | Behavior | Trade-off |
|----------|----------|-----------|
| `pessimistic` | Reserve estimated cost before call | May over-block, safest |
| `optimistic` | Check after call, reconcile | May over-spend briefly |
| `sampling` | Check budget every N calls | Low overhead, less accurate |

```python
# Pessimistic reservation example
async def tracked_call():
    estimated_cost = estimate_cost(prompt, max_tokens)
    
    # 1. Reserve budget (atomic Redis operation)
    reservation = await tracker.reserve(estimated_cost)
    
    try:
        # 2. Make LLM call
        response = await llm.invoke(...)
        
        # 3. Adjust reservation to actual cost
        await tracker.finalize(reservation, actual_cost)
    except Exception:
        # 4. Release reservation on failure
        await tracker.release(reservation)
        raise
```

### 6.6 Hierarchical Tracking (Spans)

**For agents and chains**: Group multiple LLM calls under a parent operation.

```python
# Hierarchical tracking for agents
with tracker.span("customer_support_agent", tags={"user_id": "123"}) as span:
    # Agent makes multiple internal calls
    result = agent.invoke(query)
    
    # All nested calls attributed to this span
    print(span.total_cost)      # $0.45 (sum of all nested calls)
    print(span.call_count)      # 5
    print(span.models_used)     # ["gpt-4o", "gpt-3.5-turbo"]

# Nested spans
with tracker.span("rag_pipeline") as outer:
    with tracker.span("retrieval") as inner1:
        embeddings = embed(query)
    with tracker.span("generation") as inner2:
        response = generate(context)
    
    print(outer.children)  # [inner1, inner2]
```

### 6.7 Attribution & Tagging

```python
# Tag calls for attribution
@tracker.track(tags={"team": "search", "feature": "autocomplete", "user_id": "123"})
def search_autocomplete(query: str):
    ...

# Query by tags
report = tracker.get_costs(
    start_date="2024-01-01",
    end_date="2024-01-31",
    group_by=["team", "feature"]
)

# Output:
# | team    | feature      | calls | tokens    | cost    |
# |---------|--------------|-------|-----------|---------|
# | search  | autocomplete | 45000 | 2,340,000 | $234.00 |
# | search  | rerank       | 12000 | 890,000   | $89.00  |
# | chat    | support      | 8000  | 1,200,000 | $180.00 |
```

#### Tag Cardinality Limits

**Prevent storage explosion from high-cardinality tags.**

```python
tracker = CostTracker(
    # Cardinality protection
    max_unique_tag_values=1000,       # Per tag key
    high_cardinality_tags=["request_id", "trace_id"],  # Store separately
    
    # Warn on high cardinality
    cardinality_warning_threshold=500,
)

# High-cardinality fields go in metadata, not tags
@tracker.track(
    tags={"team": "search"},                    # Low cardinality: indexed
    metadata={"request_id": str(uuid.uuid4())}  # High cardinality: not indexed
)
def my_call():
    ...
```

### 6.8 Caching Integration

**Show cost savings from caching.**

```python
tracker = CostTracker(
    track_cache_savings=True,
)

# Integrates with common caching solutions
from llm_cost_guard.integrations import CacheTracker

@cache_tracker.track
@semantic_cache.cached
def my_call(prompt):
    return llm.invoke(prompt)

# Report shows savings
report = tracker.get_costs()
print(report.cache_hits)          # 4,500
print(report.cache_savings)       # $123.45
print(report.effective_cost)      # $234.56 (after cache savings)
```

### 6.9 Reporting & Analytics

#### Built-in Reports

```python
# Daily summary
tracker.daily_report()

# Cost by model
tracker.report_by_model(period="week")

# Cost trends
tracker.trend_analysis(
    metric="cost",
    granularity="hour",
    last_n_days=7
)

# Export to DataFrame
df = tracker.to_dataframe()
```

#### Export Formats

- JSON
- CSV
- Pandas DataFrame
- Prometheus metrics
- OpenTelemetry spans

### 6.10 LangChain Integration

```python
from llm_cost_guard.integrations.langchain import CostTrackingCallback

tracker = CostTracker()

# As a callback
llm = ChatOpenAI(
    model="gpt-4o",
    callbacks=[CostTrackingCallback(tracker)]
)

# Or wrap the entire chain
from llm_cost_guard.integrations.langchain import track_chain

@track_chain(tracker, tags={"chain": "rag_pipeline"})
def my_rag_chain():
    ...
```

### 6.11 Async Support

```python
@tracker.track
async def async_llm_call():
    response = await openai.chat.completions.create(...)
    return response

# Parallel tracking
async with tracker.track_context():
    results = await asyncio.gather(
        call_gpt4(),
        call_claude(),
        call_bedrock()
    )
```

### 6.12 Rate Limiting

**Separate from budgets**: Control request/token rate.

```python
tracker = CostTracker(
    rate_limits=[
        RateLimit(
            name="requests-per-minute",
            limit=100,
            period="minute",
            scope="global"
        ),
        RateLimit(
            name="tokens-per-minute",
            limit=100000,
            period="minute",
            scope="model",  # Per model
        ),
        RateLimit(
            name="user-requests",
            limit=10,
            period="minute",
            scope="tag:user_id"  # Per user
        )
    ]
)
```

---

## 7. Technical Requirements

### 7.1 Performance

| Requirement | Target |
|-------------|--------|
| Overhead per call | <5ms |
| Memory footprint | <50MB for 1M tracked calls |
| Startup time | <100ms |
| Redis round-trip (budget check) | <10ms |

### 7.2 Thread Safety

| Component | Requirement |
|-----------|-------------|
| In-memory backend | Thread-safe with `threading.Lock` |
| Budget counters | Atomic operations |
| Report generation | Read-consistent snapshots |

```python
# Concurrent usage is safe
tracker = CostTracker()

async def handle_request():
    # Multiple concurrent calls are tracked correctly
    await tracked_call()
```

### 7.3 Storage Backends

```python
# In-memory (default, development)
tracker = CostTracker(backend="memory")

# SQLite (single-machine persistence)
tracker = CostTracker(backend="sqlite:///costs.db")

# PostgreSQL (production)
tracker = CostTracker(backend="postgresql://...")

# Redis (distributed, real-time)
tracker = CostTracker(backend="redis://...")

# DynamoDB (serverless, AWS-native)
tracker = CostTracker(backend="dynamodb://table-name")
```

#### Backend Connection Pooling

```python
tracker = CostTracker(
    backend="postgresql://...",
    pool_size=10,
    pool_timeout=30,
    pool_recycle=3600,
)
```

### 7.4 Pricing Management

#### Pricing Source Hierarchy

Pricing is retrieved using a fallback hierarchy to ensure accuracy and availability:

| Priority | Source | When Used | Freshness |
|----------|--------|-----------|-----------|
| **1st** | Provider API (live) | Default for supported providers | Always current |
| **2nd** | Cached API response | API temporarily unavailable | Configurable TTL |
| **3rd** | YAML fallback files | Offline/air-gapped, API unsupported | Updated with releases |

```python
tracker = CostTracker(
    # Pricing source strategy
    pricing_source="auto",          # Default: API → Cache → YAML
    # pricing_source="api_only",    # Only use provider APIs (fail if unavailable)
    # pricing_source="yaml_only",   # Only use bundled YAML files
    
    pricing_cache_hours=24,         # Cache API responses for 24 hours
    pricing_stale_warning_days=7,   # Warn if cache/YAML >7 days old
)
```

#### Provider API Availability

| Provider | Pricing API | Method | Notes |
|----------|-------------|--------|-------|
| AWS Bedrock | ✅ Available | `boto3` Pricing API | `pricing.get_products(ServiceCode='AmazonBedrock')` |
| Google Vertex | ✅ Available | Cloud Billing API | `cloudbilling.services.skus.list` |
| Azure OpenAI | ✅ Available | Azure Retail Prices API | REST endpoint |
| OpenAI | ❌ No API | YAML fallback | Manual updates required |
| Anthropic | ❌ No API | YAML fallback | Manual updates required |
| Cohere | ❌ No API | YAML fallback | Manual updates required |

#### Pricing Loader Implementation

```python
class PricingLoader:
    """Fetches pricing with automatic fallback."""
    
    def get_price(self, provider: str, model: str) -> ModelPricing:
        # 1. Try provider API (if available)
        if self._has_api(provider):
            try:
                pricing = self._fetch_from_api(provider, model)
                self._cache(provider, model, pricing)
                return pricing
            except (APIError, Timeout):
                logger.warning(f"Provider API unavailable for {provider}")
        
        # 2. Try cached response
        if cached := self._get_cached(provider, model):
            if cached.age_hours < self.cache_hours:
                return cached.pricing
            logger.warning(f"Using stale cache for {provider}/{model}")
            return cached.pricing
        
        # 3. Fallback to YAML
        return self._load_from_yaml(provider, model)
```

#### YAML Fallback Structure

```yaml
# pricing/openai.yaml (fallback for providers without API)
version: "2026-01-15"
source: "manual"  # vs "api"
models:
  gpt-4o:
    input_cost_per_1k: 0.0025
    output_cost_per_1k: 0.01
    cached_input_cost_per_1k: 0.00125  # Prompt caching discount
    context_window: 128000
  gpt-4o-mini:
    input_cost_per_1k: 0.00015
    output_cost_per_1k: 0.0006
    context_window: 128000
```

#### Pricing Status & Monitoring

```python
# Check pricing status
status = tracker.pricing_status()
print(status.sources)
# {
#   "bedrock": {"source": "api", "fetched": "2026-01-20T10:00:00Z"},
#   "openai": {"source": "yaml", "version": "2026-01-15"},
#   "anthropic": {"source": "cache", "age_hours": 12}
# }

print(tracker.pricing_is_stale)       # False
print(tracker.pricing_warnings)       # ["openai: using YAML fallback"]
```

#### Custom Pricing (Enterprise Discounts)

```python
# Override with your negotiated rates
tracker = CostTracker(
    pricing_overrides={
        "openai": {
            "gpt-4": {
                "input_cost_per_1k": 0.02,    # Your negotiated rate
                "output_cost_per_1k": 0.04,
            }
        },
        "bedrock": {
            "anthropic.claude-3-sonnet": {
                "input_cost_per_1k": 0.002,   # Committed use discount
                "output_cost_per_1k": 0.008,
            }
        }
    }
)
```

#### Regional Pricing (AWS Bedrock)

```python
# AWS Bedrock has different pricing per region
tracker = CostTracker(
    bedrock_region="us-east-1",  # Fetch region-specific pricing from API
)

# API automatically fetches correct regional pricing
# Falls back to us-east-1 pricing if region not found
```

#### CLI Commands

```bash
# Check pricing sources and freshness
llm-cost-guard pricing-status

# Force refresh from provider APIs
llm-cost-guard pricing-refresh

# List all available models and their prices
llm-cost-guard pricing-list --provider openai

# Validate YAML files
llm-cost-guard pricing-validate
```

### 7.5 Error Handling

```python
from llm_cost_guard import (
    BudgetExceededError,
    PricingNotFoundError,
    TokenCountError,
    TrackingUnavailableError,
    RateLimitExceededError,
)

try:
    response = tracked_call()
except BudgetExceededError as e:
    print(f"Budget '{e.budget.name}' exceeded: ${e.current:.2f} / ${e.limit:.2f}")
except TrackingUnavailableError as e:
    print(f"Tracking backend unavailable: {e.backend}")
```

---

## 8. API Design

### 8.1 Core Classes

```python
class CostTracker:
    """Main entry point for cost tracking."""
    
    def __init__(
        self,
        budgets: list[Budget] = None,
        rate_limits: list[RateLimit] = None,
        backend: str = "memory",
        auto_detect_provider: bool = True,
        pricing_update: bool = True,
        on_tracking_failure: Literal["block", "allow", "fallback"] = "allow",
        store_prompts: bool = False,
        track_failed_calls: bool = True,
    ): ...
    
    def track(
        self,
        tags: dict = None,
        streaming: bool = False,
    ) -> Callable:
        """Decorator to track LLM call costs."""
        
    def track_context(
        self,
        tags: dict = None,
    ) -> ContextManager:
        """Context manager for tracking."""
    
    def span(
        self,
        name: str,
        tags: dict = None,
    ) -> Span:
        """Create a tracking span for grouping multiple calls."""
        
    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tags: dict = None,
        success: bool = True,
    ) -> CostRecord:
        """Manually record a call (for custom integrations)."""
        
    def get_costs(
        self,
        start_date: str = None,
        end_date: str = None,
        tags: dict = None,
        group_by: list[str] = None,
    ) -> CostReport:
        """Query tracked costs."""
    
    def health_check(self) -> HealthStatus:
        """Check tracker and backend health."""


class Budget:
    """Budget configuration."""
    
    name: str
    limit: float
    period: Literal["request", "minute", "hour", "day", "week", "month"]
    action: BudgetAction
    tags: dict = None  # Apply budget only to matching tags
    warning_threshold: float = 0.8  # Warn at 80%


class RateLimit:
    """Rate limit configuration."""
    
    name: str
    limit: int
    period: Literal["second", "minute", "hour"]
    scope: Literal["global", "model", "provider"] | str  # "tag:user_id"


class CostRecord:
    """Single LLM call record."""
    
    timestamp: datetime
    provider: str
    model: str
    model_type: Literal["chat", "embedding", "image", "audio"]
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    latency_ms: int
    tags: dict
    metadata: dict  # High-cardinality data
    success: bool
    error_type: str | None
    cached: bool
    cache_savings: float
    span_id: str | None


class Span:
    """Hierarchical tracking span."""
    
    name: str
    start_time: datetime
    end_time: datetime
    total_cost: float
    call_count: int
    children: list[Span]
    tags: dict
```

### 8.2 CLI Interface

```bash
# View current costs
llm-cost-guard status

# Health check
llm-cost-guard health

# Daily report
llm-cost-guard report --period day

# Cost by model
llm-cost-guard report --group-by model

# Update pricing data
llm-cost-guard update-pricing

# Check pricing freshness
llm-cost-guard pricing-status

# Export data
llm-cost-guard export --format csv --output costs.csv

# Validate configuration
llm-cost-guard validate-config
```

---

## 9. Project Structure

```
llm-cost-guard/
├── llm_cost_guard/
│   ├── __init__.py
│   ├── tracker.py              # CostTracker class
│   ├── budget.py               # Budget enforcement
│   ├── rate_limit.py           # Rate limiting
│   ├── span.py                 # Hierarchical tracking
│   ├── security.py             # Credential management, redaction
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── loader.py           # Pricing loader with fallback hierarchy
│   │   ├── cache.py            # API response caching
│   │   ├── apis/
│   │   │   ├── __init__.py
│   │   │   ├── bedrock.py      # AWS Pricing API integration
│   │   │   ├── vertex.py       # GCP Cloud Billing API
│   │   │   └── azure.py        # Azure Retail Prices API
│   │   └── fallback/           # YAML fallbacks (no API available)
│   │       ├── openai.yaml
│   │       ├── anthropic.yaml
│   │       └── cohere.yaml
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py             # Provider interface
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── bedrock.py
│   │   └── vertex.py
│   ├── clients/                # Wrapped clients
│   │   ├── __init__.py
│   │   ├── openai.py           # TrackedOpenAI
│   │   ├── anthropic.py
│   │   └── bedrock.py
│   ├── tokenizers/
│   │   ├── __init__.py
│   │   ├── tiktoken.py
│   │   └── anthropic.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py             # Backend interface
│   │   ├── memory.py
│   │   ├── sqlite.py
│   │   ├── postgres.py
│   │   ├── redis.py
│   │   └── dynamodb.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── langchain.py
│   │   ├── prometheus.py
│   │   ├── opentelemetry.py
│   │   └── cache.py
│   ├── reports.py              # Reporting utilities
│   └── cli.py                  # CLI commands
├── tests/
│   ├── unit/
│   │   ├── test_tracker.py
│   │   ├── test_budget.py
│   │   ├── test_rate_limit.py
│   │   └── test_security.py
│   ├── integration/
│   │   ├── test_openai.py
│   │   ├── test_bedrock.py
│   │   └── test_backends.py
│   ├── load/
│   │   └── test_concurrent.py
│   └── conftest.py
├── examples/
│   ├── quickstart.py
│   ├── langchain_rag.py
│   ├── bedrock_example.py
│   ├── budget_alerts.py
│   ├── distributed_tracking.py
│   └── streaming_example.py
├── docs/
│   ├── architecture.md
│   ├── security.md
│   ├── deployment.md
│   └── migration.md
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── SECURITY.md
└── LICENSE
```

---

## 10. Dependencies

### Required

| Package | Purpose | Version |
|---------|---------|---------|
| `tiktoken` | OpenAI token counting | >=0.5.0 |
| `pyyaml` | Pricing file loading | >=6.0 |
| `httpx` | Async HTTP (pricing updates) | >=0.24.0 |

### Optional

| Package | Purpose | Install Extra |
|---------|---------|---------------|
| `anthropic` | Anthropic tokenizer | `[anthropic]` |
| `boto3` | Bedrock integration | `[bedrock]` |
| `langchain` | LangChain callbacks | `[langchain]` |
| `sqlalchemy` | SQL backends | `[sql]` |
| `redis` | Redis backend | `[redis]` |
| `prometheus-client` | Prometheus export | `[prometheus]` |
| `opentelemetry-api` | OpenTelemetry export | `[otel]` |

---

## 11. Testing Requirements

### Test Categories

| Category | Coverage Target | Tools |
|----------|-----------------|-------|
| Unit tests | 90%+ | pytest, pytest-cov |
| Integration tests | Key paths | pytest, moto (AWS mocks) |
| Load tests | 10K concurrent | locust |
| Security tests | OWASP top 10 | bandit, safety |

### Accuracy Validation

```python
# Compare tracked costs to actual provider bills
def test_accuracy_against_provider():
    # 1. Run tracked calls
    # 2. Wait for provider usage API
    # 3. Compare tracked vs billed
    assert abs(tracked_cost - billed_cost) / billed_cost < 0.01  # <1% variance
```

### Chaos Testing

| Scenario | Expected Behavior |
|----------|-------------------|
| Redis connection lost | Fail-closed or fallback based on config |
| Pricing file corrupted | Use cached pricing, warn |
| Clock skew between nodes | Use Redis server time |
| High cardinality tag attack | Reject after limit |

---

## 12. Operational Requirements

### 12.1 Observability

```python
# Health check endpoint
status = tracker.health_check()
print(status.backend_connected)    # True
print(status.pricing_fresh)        # True
print(status.last_record_time)     # 2026-01-20T10:30:00Z
print(status.pending_records)      # 0 (async flush queue)
```

### 12.2 Metrics Export

```python
# Prometheus metrics
tracker = CostTracker(
    metrics_export="prometheus",
    metrics_port=9090,
)

# Exported metrics:
# llm_cost_guard_calls_total{provider, model, status}
# llm_cost_guard_tokens_total{provider, model, type}
# llm_cost_guard_cost_dollars{provider, model}
# llm_cost_guard_latency_seconds{provider, model}
# llm_cost_guard_budget_utilization{budget_name}
```

### 12.3 Data Retention

```python
tracker = CostTracker(
    retention_days=90,           # Keep detailed records for 90 days
    aggregation_after_days=30,   # Aggregate to daily after 30 days
)
```

### 12.4 Migration Support

```python
# Migrate from in-memory to PostgreSQL
from llm_cost_guard.migration import migrate_backend

migrate_backend(
    source="memory",
    destination="postgresql://...",
    tracker=tracker,
)
```

---

## 13. Milestones

### MVP (v0.1.0) - 2 weeks

- [ ] Core CostTracker with decorator/context manager
- [ ] OpenAI provider with accurate token counting
- [ ] Anthropic provider
- [ ] In-memory backend (thread-safe)
- [ ] Basic budget enforcement (WARN, BLOCK)
- [ ] Fail-safe modes (block/allow/fallback)
- [ ] No prompt/key logging (security)
- [ ] Simple daily/weekly reports
- [ ] README with quick start
- [ ] 80%+ unit test coverage

### v0.2.0 - 2 weeks

- [ ] AWS Bedrock provider (Claude, Titan, Llama)
- [ ] SQLite backend for persistence
- [ ] Streaming support
- [ ] Retry/failure tracking
- [ ] Tagging and attribution
- [ ] Group-by reporting
- [ ] CLI tool

### v0.3.0 - 2 weeks

- [ ] LangChain integration (callbacks)
- [ ] Hierarchical tracking (spans)
- [ ] Budget callbacks and events
- [ ] Redis backend
- [ ] Distributed budget enforcement
- [ ] Prometheus metrics export
- [ ] Google Vertex AI provider

### v1.0.0 - 2 weeks

- [ ] PostgreSQL backend
- [ ] DynamoDB backend
- [ ] OpenTelemetry integration
- [ ] Rate limiting
- [ ] Cache integration
- [ ] Comprehensive documentation
- [ ] 90%+ test coverage
- [ ] Security audit

---

## 14. Competitive Analysis

| Feature | llm-cost-guard | LangSmith | Helicone | OpenLLMetry |
|---------|----------------|-----------|----------|-------------|
| Open source | ✅ | ❌ | ✅ | ✅ |
| Self-hosted | ✅ | ❌ | ✅ | ✅ |
| Budget enforcement | ✅ | ❌ | ❌ | ❌ |
| Distributed budgets | ✅ | N/A | ❌ | ❌ |
| Rate limiting | ✅ | ❌ | ❌ | ❌ |
| Multi-provider | ✅ | ✅ | ✅ | ✅ |
| Bedrock support | ✅ | Limited | ✅ | ✅ |
| Zero config | ✅ | ❌ | ❌ | ❌ |
| No external service | ✅ | ❌ | ❌ | ✅ |
| Streaming support | ✅ | ✅ | ✅ | ✅ |
| Fail-safe modes | ✅ | N/A | ❌ | ❌ |

**Key differentiators**:
1. Budget enforcement with automated actions (unique)
2. Distributed budget enforcement across instances (unique)
3. Fail-safe modes for production reliability
4. Zero external dependencies for basic usage

---

## 15. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Budget bypass due to tracking failure | Medium | High | Fail-closed mode, health checks |
| Inaccurate pricing data | High | Medium | Staleness warnings, validation against bills |
| Multi-instance budget overrun | High | High | Pessimistic reservation, Redis atomic ops |
| Performance overhead blocks adoption | Low | High | <5ms target, benchmarks, sampling mode |
| Security breach via logged prompts | Medium | Critical | No prompt storage by default |
| High-cardinality tag explosion | Medium | Medium | Cardinality limits, separate metadata field |
| Provider API changes break tracking | Medium | Medium | Wrapper clients, version pinning |

---

## 16. Success Criteria for EB2-NIW

| Metric | Target | Timeline |
|--------|--------|----------|
| GitHub stars | 200+ | 3 months |
| PyPI downloads | 10,000+ | 3 months |
| External blog posts | 2-3 | 3 months |
| Company adoption letters | 1-2 | 6 months |
| Hacker News/Reddit traction | 1 viral post | 3 months |

---

## 17. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-20 | Prashant Dudami | Initial draft |
| 2.0 | 2026-01-20 | Prashant Dudami | Added security, distributed systems, edge cases per architecture review |
| 2.1 | 2026-01-20 | Prashant Dudami | Updated pricing to use Provider API → Cache → YAML fallback hierarchy |
