"""
Storage backends for LLM Cost Guard.
"""

from llm_cost_guard.backends.base import Backend
from llm_cost_guard.backends.memory import MemoryBackend

__all__ = ["Backend", "MemoryBackend", "get_backend"]


def get_backend(backend_url: str, **kwargs) -> Backend:
    """
    Create a backend instance from a URL.

    Supported formats:
    - "memory" or "memory://" - In-memory storage
    - "sqlite:///path/to/db.sqlite" - SQLite database
    - "postgresql://user:pass@host/db" - PostgreSQL database
    - "redis://host:port/db" - Redis
    - "dynamodb://table-name" - DynamoDB

    Args:
        backend_url: Backend connection URL
        **kwargs: Additional backend-specific options

    Returns:
        Backend instance
    """
    if backend_url in ("memory", "memory://"):
        return MemoryBackend(**kwargs)

    if backend_url.startswith("sqlite:"):
        from llm_cost_guard.backends.sqlite import SQLiteBackend

        return SQLiteBackend(backend_url, **kwargs)

    if backend_url.startswith("postgresql://") or backend_url.startswith("postgres://"):
        from llm_cost_guard.backends.postgres import PostgresBackend

        return PostgresBackend(backend_url, **kwargs)

    if backend_url.startswith("redis://"):
        from llm_cost_guard.backends.redis_backend import RedisBackend

        return RedisBackend(backend_url, **kwargs)

    if backend_url.startswith("dynamodb://"):
        from llm_cost_guard.backends.dynamodb import DynamoDBBackend

        return DynamoDBBackend(backend_url, **kwargs)

    raise ValueError(f"Unsupported backend URL: {backend_url}")
