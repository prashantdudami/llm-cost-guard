"""
Hierarchical tracking spans for LLM Cost Guard.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
import contextvars
import threading
import uuid

if TYPE_CHECKING:
    from llm_cost_guard.models import CostRecord


# Context variable to track the current span
_current_span: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "current_span", default=None
)


@dataclass
class Span:
    """Hierarchical tracking span for grouping multiple LLM calls."""

    name: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Aggregated metrics
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    models_used: Set[str] = field(default_factory=set)

    # Hierarchy
    children: List["Span"] = field(default_factory=list)
    _records: List["CostRecord"] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # For context management
    _token: Optional[contextvars.Token] = field(default=None, repr=False)
    _previous_span: Optional["Span"] = field(default=None, repr=False)

    def __enter__(self) -> "Span":
        """Enter the span context."""
        self.start_time = datetime.now()

        # Save previous span and set this as current
        self._previous_span = _current_span.get()
        self._token = _current_span.set(self)

        # If there's a parent span, register as child
        if self._previous_span is not None:
            self.parent_id = self._previous_span.span_id
            self._previous_span._add_child(self)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the span context."""
        self.end_time = datetime.now()

        # Restore previous span
        if self._token is not None:
            _current_span.reset(self._token)

        # Propagate costs to parent
        if self._previous_span is not None:
            self._previous_span._propagate_child_costs(self)

    def _add_child(self, child: "Span") -> None:
        """Add a child span."""
        with self._lock:
            self.children.append(child)

    def _propagate_child_costs(self, child: "Span") -> None:
        """Propagate child span costs to this span."""
        with self._lock:
            self.total_cost += child.total_cost
            self.total_input_tokens += child.total_input_tokens
            self.total_output_tokens += child.total_output_tokens
            self.call_count += child.call_count
            self.models_used.update(child.models_used)

    def record_call(
        self,
        cost: float,
        input_tokens: int,
        output_tokens: int,
        model: str,
        record: Optional["CostRecord"] = None,
    ) -> None:
        """Record an LLM call in this span."""
        with self._lock:
            self.total_cost += cost
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.call_count += 1
            self.models_used.add(model)
            if record is not None:
                self._records.append(record)

    @property
    def duration_ms(self) -> Optional[int]:
        """Get the span duration in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)

    @property
    def records(self) -> List["CostRecord"]:
        """Get all records in this span."""
        return list(self._records)

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
            "models_used": list(self.models_used),
            "children": [child.to_dict() for child in self.children],
        }


def get_current_span() -> Optional[Span]:
    """Get the current active span, if any."""
    return _current_span.get()
