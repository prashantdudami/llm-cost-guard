"""
Unit tests for metrics exporters.
"""

import logging
import pytest
from unittest.mock import MagicMock, patch

from llm_cost_guard.metrics import (
    MetricsExporter,
    NoOpExporter,
    LoggingExporter,
    CompositeExporter,
    TrackerMetrics,
    get_metrics_exporter,
)


class TestNoOpExporter:
    """Tests for NoOpExporter."""

    def test_counter_does_nothing(self):
        """Test counter does nothing."""
        exporter = NoOpExporter()
        
        # Should not raise
        exporter.counter("test_counter", 1, {"tag": "value"})

    def test_gauge_does_nothing(self):
        """Test gauge does nothing."""
        exporter = NoOpExporter()
        
        exporter.gauge("test_gauge", 42.5, {"tag": "value"})

    def test_histogram_does_nothing(self):
        """Test histogram does nothing."""
        exporter = NoOpExporter()
        
        exporter.histogram("test_histogram", 0.05, {"model": "gpt-4"})

    def test_timing_does_nothing(self):
        """Test timing does nothing."""
        exporter = NoOpExporter()
        
        exporter.timing("test_timing", 150.5, {"endpoint": "api"})


class TestLoggingExporter:
    """Tests for LoggingExporter."""

    def test_counter_logs(self, caplog):
        """Test counter logs metric."""
        exporter = LoggingExporter(level=logging.INFO)
        
        with caplog.at_level(logging.INFO):
            exporter.counter("requests_total", 1, {"model": "gpt-4"})
        
        assert "METRIC counter requests_total=1" in caplog.text
        assert "model" in caplog.text

    def test_gauge_logs(self, caplog):
        """Test gauge logs metric."""
        exporter = LoggingExporter(level=logging.INFO)
        
        with caplog.at_level(logging.INFO):
            exporter.gauge("budget_utilization", 0.75, {"budget": "daily"})
        
        assert "METRIC gauge budget_utilization=0.75" in caplog.text

    def test_histogram_logs(self, caplog):
        """Test histogram logs metric."""
        exporter = LoggingExporter(level=logging.INFO)
        
        with caplog.at_level(logging.INFO):
            exporter.histogram("cost_dollars", 0.05)
        
        assert "METRIC histogram cost_dollars=0.05" in caplog.text

    def test_timing_logs(self, caplog):
        """Test timing logs metric."""
        exporter = LoggingExporter(level=logging.INFO)
        
        with caplog.at_level(logging.INFO):
            exporter.timing("request_latency", 150.5)
        
        assert "METRIC timing request_latency=150.5ms" in caplog.text


class TestCompositeExporter:
    """Tests for CompositeExporter."""

    def test_sends_to_all_exporters(self):
        """Test metrics sent to all exporters."""
        exporter1 = MagicMock(spec=MetricsExporter)
        exporter2 = MagicMock(spec=MetricsExporter)
        
        composite = CompositeExporter([exporter1, exporter2])
        
        composite.counter("test", 1, {"tag": "value"})
        
        exporter1.counter.assert_called_once_with("test", 1, {"tag": "value"})
        exporter2.counter.assert_called_once_with("test", 1, {"tag": "value"})

    def test_continues_on_error(self):
        """Test continues if one exporter fails."""
        failing_exporter = MagicMock(spec=MetricsExporter)
        failing_exporter.counter.side_effect = Exception("Export failed")
        
        working_exporter = MagicMock(spec=MetricsExporter)
        
        composite = CompositeExporter([failing_exporter, working_exporter])
        
        # Should not raise
        composite.counter("test", 1)
        
        # Second exporter still called
        working_exporter.counter.assert_called_once()

    def test_all_metric_types(self):
        """Test all metric types work."""
        exporter = MagicMock(spec=MetricsExporter)
        composite = CompositeExporter([exporter])
        
        composite.counter("c", 1)
        composite.gauge("g", 42.0)
        composite.histogram("h", 0.5)
        composite.timing("t", 100.0)
        
        assert exporter.counter.called
        assert exporter.gauge.called
        assert exporter.histogram.called
        assert exporter.timing.called


class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    @pytest.fixture
    def mock_prometheus(self):
        """Mock prometheus_client."""
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(),
        }):
            yield

    def test_counter_creates_metric(self):
        """Test counter creates Prometheus metric."""
        try:
            from prometheus_client import CollectorRegistry
            from llm_cost_guard.metrics import PrometheusExporter
            
            registry = CollectorRegistry()
            exporter = PrometheusExporter(prefix="test", registry=registry)
            
            exporter.counter("requests", 1, {"model": "gpt-4"})
            exporter.counter("requests", 1, {"model": "gpt-4"})
            
            # Metric should exist
            assert len(exporter._counters) > 0
        except ImportError:
            pytest.skip("prometheus_client not installed")

    def test_gauge_creates_metric(self):
        """Test gauge creates Prometheus metric."""
        try:
            from prometheus_client import CollectorRegistry
            from llm_cost_guard.metrics import PrometheusExporter
            
            registry = CollectorRegistry()
            exporter = PrometheusExporter(prefix="test", registry=registry)
            
            exporter.gauge("utilization", 0.75)
            
            assert len(exporter._gauges) > 0
        except ImportError:
            pytest.skip("prometheus_client not installed")

    def test_histogram_creates_metric(self):
        """Test histogram creates Prometheus metric."""
        try:
            from prometheus_client import CollectorRegistry
            from llm_cost_guard.metrics import PrometheusExporter
            
            registry = CollectorRegistry()
            exporter = PrometheusExporter(prefix="test", registry=registry)
            
            exporter.histogram("cost", 0.05)
            
            assert len(exporter._histograms) > 0
        except ImportError:
            pytest.skip("prometheus_client not installed")


class TestTrackerMetrics:
    """Tests for TrackerMetrics constants."""

    def test_counter_names(self):
        """Test counter metric names."""
        assert TrackerMetrics.REQUESTS_TOTAL == "requests_total"
        assert TrackerMetrics.COST_DOLLARS_TOTAL == "cost_dollars_total"
        assert TrackerMetrics.BUDGET_EXCEEDED_TOTAL == "budget_exceeded_total"

    def test_gauge_names(self):
        """Test gauge metric names."""
        assert TrackerMetrics.BUDGET_UTILIZATION == "budget_utilization_ratio"
        assert TrackerMetrics.BACKEND_HEALTHY == "backend_healthy"
        assert TrackerMetrics.USING_FALLBACK == "using_fallback"

    def test_histogram_names(self):
        """Test histogram metric names."""
        assert TrackerMetrics.REQUEST_LATENCY == "request_latency"
        assert TrackerMetrics.COST_PER_REQUEST == "cost_per_request_dollars"


class TestGetMetricsExporter:
    """Tests for get_metrics_exporter factory."""

    def test_get_none_exporter(self):
        """Test getting no-op exporter."""
        exporter = get_metrics_exporter("none")
        
        assert isinstance(exporter, NoOpExporter)

    def test_get_logging_exporter(self):
        """Test getting logging exporter."""
        exporter = get_metrics_exporter("logging", level=logging.WARNING)
        
        assert isinstance(exporter, LoggingExporter)

    def test_get_prometheus_exporter(self):
        """Test getting Prometheus exporter."""
        try:
            exporter = get_metrics_exporter("prometheus", port=9999)
            
            from llm_cost_guard.metrics import PrometheusExporter
            assert isinstance(exporter, PrometheusExporter)
        except ImportError:
            pytest.skip("prometheus_client not installed")

    def test_unknown_exporter_raises(self):
        """Test unknown exporter raises."""
        with pytest.raises(ValueError, match="Unknown metrics exporter"):
            get_metrics_exporter("unknown")
