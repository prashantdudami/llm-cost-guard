"""
Metrics export for LLM Cost Guard.

Supports multiple metrics backends:
- Prometheus (push and pull)
- StatsD
- OpenTelemetry
- Custom exporters
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MetricsExporter(ABC):
    """
    Abstract metrics exporter.

    Implement this interface for any metrics backend.
    """

    @abstractmethod
    def counter(self, name: str, value: int = 1, tags: Optional[dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        pass

    @abstractmethod
    def gauge(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        pass

    @abstractmethod
    def histogram(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        """Record a histogram value."""
        pass

    @abstractmethod
    def timing(self, name: str, value_ms: float, tags: Optional[dict[str, str]] = None) -> None:
        """Record a timing value in milliseconds."""
        pass


class NoOpExporter(MetricsExporter):
    """No-op exporter that discards all metrics."""

    def counter(self, name: str, value: int = 1, tags: Optional[dict[str, str]] = None) -> None:
        pass

    def gauge(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        pass

    def histogram(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        pass

    def timing(self, name: str, value_ms: float, tags: Optional[dict[str, str]] = None) -> None:
        pass


class LoggingExporter(MetricsExporter):
    """Exporter that logs metrics (useful for debugging)."""

    def __init__(self, level: int = logging.DEBUG):
        self._level = level

    def counter(self, name: str, value: int = 1, tags: Optional[dict[str, str]] = None) -> None:
        logger.log(self._level, f"METRIC counter {name}={value} {tags or {}}")

    def gauge(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        logger.log(self._level, f"METRIC gauge {name}={value} {tags or {}}")

    def histogram(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        logger.log(self._level, f"METRIC histogram {name}={value} {tags or {}}")

    def timing(self, name: str, value_ms: float, tags: Optional[dict[str, str]] = None) -> None:
        logger.log(self._level, f"METRIC timing {name}={value_ms}ms {tags or {}}")


class PrometheusExporter(MetricsExporter):
    """
    Prometheus metrics exporter.

    Exposes metrics for Prometheus scraping.

    Usage:
        exporter = PrometheusExporter(port=9090)
        exporter.start_server()  # Start HTTP server for scraping

        exporter.counter("llm_requests_total", tags={"model": "gpt-4"})
        exporter.histogram("llm_cost_dollars", value=0.05)
    """

    def __init__(
        self,
        prefix: str = "llm_cost_guard",
        port: int = 9090,
        registry: Optional[Any] = None,
    ):
        """
        Initialize Prometheus exporter.

        Args:
            prefix: Metric name prefix
            port: Port for HTTP server
            registry: Custom Prometheus registry
        """
        try:
            from prometheus_client import (
                REGISTRY,
                Counter,
                Gauge,
                Histogram,
                start_http_server,
            )
        except ImportError:
            raise ImportError(
                "prometheus_client required for PrometheusExporter. "
                "Install with: pip install llm-cost-guard[prometheus]"
            )

        self._prefix = prefix
        self._port = port
        self._registry = registry or REGISTRY
        self._start_http_server = start_http_server

        # Metric storage
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def _get_counter(self, name: str, tags: Optional[dict[str, str]] = None) -> Any:
        """Get or create a counter."""
        from prometheus_client import Counter

        full_name = f"{self._prefix}_{name}"
        label_names = list(tags.keys()) if tags else []

        with self._lock:
            key = f"{full_name}:{','.join(sorted(label_names))}"
            if key not in self._counters:
                self._counters[key] = Counter(
                    full_name,
                    f"Counter for {name}",
                    labelnames=label_names,
                    registry=self._registry,
                )
            return self._counters[key]

    def _get_gauge(self, name: str, tags: Optional[dict[str, str]] = None) -> Any:
        """Get or create a gauge."""
        from prometheus_client import Gauge

        full_name = f"{self._prefix}_{name}"
        label_names = list(tags.keys()) if tags else []

        with self._lock:
            key = f"{full_name}:{','.join(sorted(label_names))}"
            if key not in self._gauges:
                self._gauges[key] = Gauge(
                    full_name,
                    f"Gauge for {name}",
                    labelnames=label_names,
                    registry=self._registry,
                )
            return self._gauges[key]

    def _get_histogram(self, name: str, tags: Optional[dict[str, str]] = None) -> Any:
        """Get or create a histogram."""
        from prometheus_client import Histogram

        full_name = f"{self._prefix}_{name}"
        label_names = list(tags.keys()) if tags else []

        with self._lock:
            key = f"{full_name}:{','.join(sorted(label_names))}"
            if key not in self._histograms:
                self._histograms[key] = Histogram(
                    full_name,
                    f"Histogram for {name}",
                    labelnames=label_names,
                    registry=self._registry,
                )
            return self._histograms[key]

    def counter(self, name: str, value: int = 1, tags: Optional[dict[str, str]] = None) -> None:
        """Increment a counter."""
        counter = self._get_counter(name, tags)
        if tags:
            counter.labels(**tags).inc(value)
        else:
            counter.inc(value)

    def gauge(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        """Set a gauge value."""
        gauge = self._get_gauge(name, tags)
        if tags:
            gauge.labels(**tags).set(value)
        else:
            gauge.set(value)

    def histogram(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        """Observe a histogram value."""
        histogram = self._get_histogram(name, tags)
        if tags:
            histogram.labels(**tags).observe(value)
        else:
            histogram.observe(value)

    def timing(self, name: str, value_ms: float, tags: Optional[dict[str, str]] = None) -> None:
        """Record timing (as histogram in seconds)."""
        self.histogram(f"{name}_seconds", value_ms / 1000.0, tags)

    def start_server(self) -> None:
        """Start Prometheus HTTP server for scraping."""
        self._start_http_server(self._port, registry=self._registry)
        logger.info(f"Prometheus metrics server started on port {self._port}")


class StatsDExporter(MetricsExporter):
    """
    StatsD metrics exporter.

    Works with any StatsD-compatible backend (Datadog, Graphite, etc.).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "llm_cost_guard",
    ):
        """
        Initialize StatsD exporter.

        Args:
            host: StatsD host
            port: StatsD port
            prefix: Metric name prefix
        """
        try:
            import statsd
        except ImportError:
            raise ImportError(
                "statsd package required for StatsDExporter. "
                "Install with: pip install statsd"
            )

        self._client = statsd.StatsClient(host, port, prefix=prefix)

    def counter(self, name: str, value: int = 1, tags: Optional[dict[str, str]] = None) -> None:
        """Increment a counter."""
        metric_name = self._format_name(name, tags)
        self._client.incr(metric_name, value)

    def gauge(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        """Set a gauge value."""
        metric_name = self._format_name(name, tags)
        self._client.gauge(metric_name, value)

    def histogram(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        """Record a histogram value (as timing in StatsD)."""
        metric_name = self._format_name(name, tags)
        self._client.timing(metric_name, value * 1000)  # Convert to ms

    def timing(self, name: str, value_ms: float, tags: Optional[dict[str, str]] = None) -> None:
        """Record timing."""
        metric_name = self._format_name(name, tags)
        self._client.timing(metric_name, value_ms)

    def _format_name(self, name: str, tags: Optional[dict[str, str]] = None) -> str:
        """Format metric name with tags."""
        if not tags:
            return name
        # StatsD uses dot notation for tags
        tag_str = ".".join(f"{k}.{v}" for k, v in sorted(tags.items()))
        return f"{name}.{tag_str}"


class CompositeExporter(MetricsExporter):
    """
    Composite exporter that sends to multiple backends.

    Usage:
        exporter = CompositeExporter([
            PrometheusExporter(),
            StatsDExporter(host="statsd.local"),
        ])
    """

    def __init__(self, exporters: list[MetricsExporter]):
        self._exporters = exporters

    def counter(self, name: str, value: int = 1, tags: Optional[dict[str, str]] = None) -> None:
        for exporter in self._exporters:
            try:
                exporter.counter(name, value, tags)
            except Exception as e:
                logger.warning(f"Exporter {type(exporter).__name__} failed: {e}")

    def gauge(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        for exporter in self._exporters:
            try:
                exporter.gauge(name, value, tags)
            except Exception as e:
                logger.warning(f"Exporter {type(exporter).__name__} failed: {e}")

    def histogram(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        for exporter in self._exporters:
            try:
                exporter.histogram(name, value, tags)
            except Exception as e:
                logger.warning(f"Exporter {type(exporter).__name__} failed: {e}")

    def timing(self, name: str, value_ms: float, tags: Optional[dict[str, str]] = None) -> None:
        for exporter in self._exporters:
            try:
                exporter.timing(name, value_ms, tags)
            except Exception as e:
                logger.warning(f"Exporter {type(exporter).__name__} failed: {e}")


@dataclass
class TrackerMetrics:
    """
    Pre-defined metrics for CostTracker.

    Standard metric names used by the tracker.
    """

    # Counters
    REQUESTS_TOTAL = "requests_total"
    TOKENS_INPUT_TOTAL = "tokens_input_total"
    TOKENS_OUTPUT_TOTAL = "tokens_output_total"
    COST_DOLLARS_TOTAL = "cost_dollars_total"
    BUDGET_EXCEEDED_TOTAL = "budget_exceeded_total"
    RATE_LIMITED_TOTAL = "rate_limited_total"
    ERRORS_TOTAL = "errors_total"

    # Gauges
    BUDGET_UTILIZATION = "budget_utilization_ratio"
    BACKEND_HEALTHY = "backend_healthy"
    USING_FALLBACK = "using_fallback"

    # Histograms
    REQUEST_LATENCY = "request_latency"
    COST_PER_REQUEST = "cost_per_request_dollars"


def get_metrics_exporter(
    exporter_type: str = "none",
    **kwargs: Any,
) -> MetricsExporter:
    """
    Factory function to create metrics exporters.

    Args:
        exporter_type: One of "none", "logging", "prometheus", "statsd"
        **kwargs: Exporter-specific configuration

    Returns:
        MetricsExporter instance
    """
    if exporter_type == "none":
        return NoOpExporter()

    if exporter_type == "logging":
        return LoggingExporter(level=kwargs.get("level", logging.DEBUG))

    if exporter_type == "prometheus":
        return PrometheusExporter(
            prefix=kwargs.get("prefix", "llm_cost_guard"),
            port=kwargs.get("port", 9090),
        )

    if exporter_type == "statsd":
        return StatsDExporter(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 8125),
            prefix=kwargs.get("prefix", "llm_cost_guard"),
        )

    raise ValueError(f"Unknown metrics exporter: {exporter_type}")
