"""
Unit tests for span tracking.
"""

import pytest
from datetime import datetime

from llm_cost_guard.span import Span, get_current_span


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = Span(name="test-span")
        assert span.name == "test-span"
        assert span.span_id is not None
        assert span.total_cost == 0.0
        assert span.call_count == 0

    def test_span_context_manager(self):
        """Test span as context manager."""
        with Span(name="test-span") as span:
            assert span.start_time is not None
            assert span.end_time is None

        assert span.end_time is not None
        assert span.duration_ms is not None

    def test_span_record_call(self):
        """Test recording calls in a span."""
        with Span(name="test-span") as span:
            span.record_call(
                cost=0.01,
                input_tokens=100,
                output_tokens=50,
                model="gpt-4o",
            )
            span.record_call(
                cost=0.02,
                input_tokens=200,
                output_tokens=100,
                model="gpt-4o",
            )

        assert span.call_count == 2
        assert span.total_cost == 0.03
        assert span.total_input_tokens == 300
        assert span.total_output_tokens == 150
        assert "gpt-4o" in span.models_used

    def test_nested_spans(self):
        """Test nested span tracking."""
        with Span(name="outer") as outer:
            outer.record_call(cost=0.01, input_tokens=100, output_tokens=50, model="gpt-4o")

            with Span(name="inner") as inner:
                inner.record_call(cost=0.02, input_tokens=200, output_tokens=100, model="claude")

            # After inner exits, its cost should be propagated to outer
            assert inner.call_count == 1
            assert inner.total_cost == 0.02

        assert outer.call_count == 2  # Outer's direct call + inner's propagated
        assert outer.total_cost == 0.03
        assert len(outer.children) == 1
        assert outer.children[0].name == "inner"

    def test_get_current_span(self):
        """Test getting current span."""
        assert get_current_span() is None

        with Span(name="test-span") as span:
            current = get_current_span()
            assert current is span

        assert get_current_span() is None

    def test_span_with_tags(self):
        """Test span with tags."""
        with Span(name="test-span", tags={"team": "search"}) as span:
            pass

        assert span.tags["team"] == "search"

    def test_span_to_dict(self):
        """Test span serialization."""
        with Span(name="test-span", tags={"team": "search"}) as span:
            span.record_call(cost=0.01, input_tokens=100, output_tokens=50, model="gpt-4o")

        data = span.to_dict()

        assert data["name"] == "test-span"
        assert data["total_cost"] == 0.01
        assert data["call_count"] == 1
        assert "gpt-4o" in data["models_used"]
        assert data["tags"]["team"] == "search"

    def test_span_parent_child_ids(self):
        """Test parent-child ID relationship."""
        with Span(name="parent") as parent:
            with Span(name="child") as child:
                assert child.parent_id == parent.span_id

    def test_multiple_models_in_span(self):
        """Test tracking multiple models in a span."""
        with Span(name="multi-model") as span:
            span.record_call(cost=0.01, input_tokens=100, output_tokens=50, model="gpt-4o")
            span.record_call(cost=0.02, input_tokens=100, output_tokens=50, model="claude")
            span.record_call(cost=0.01, input_tokens=100, output_tokens=50, model="gpt-4o")

        assert len(span.models_used) == 2
        assert "gpt-4o" in span.models_used
        assert "claude" in span.models_used
