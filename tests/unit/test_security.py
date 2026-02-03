"""
Security-focused tests for LLM Cost Guard.

Tests for:
- Path traversal prevention
- Input validation
- SQL injection prevention
- Key injection prevention
"""

import pytest
import os
import tempfile

from llm_cost_guard import CostTracker
from llm_cost_guard.secrets import FileSecretsProvider
from llm_cost_guard.backends.sqlite import SQLiteBackend


class TestPathTraversalPrevention:
    """Tests for path traversal prevention in FileSecretsProvider."""

    def test_rejects_double_dot(self):
        """Test rejection of .. in paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretsProvider(base_path=tmpdir)
            
            # Various path traversal attempts
            assert provider.get_secret("..") is None
            assert provider.get_secret("../secret") is None
            assert provider.get_secret("foo/../bar") is None
            assert provider.get_secret("../../etc/passwd") is None

    def test_rejects_absolute_paths(self):
        """Test rejection of absolute paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretsProvider(base_path=tmpdir)
            
            assert provider.get_secret("/etc/passwd") is None
            assert provider.get_secret("/var/secrets/key") is None

    def test_rejects_path_separators(self):
        """Test rejection of path separators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretsProvider(base_path=tmpdir)
            
            assert provider.get_secret("subdir/secret") is None
            assert provider.get_secret("a\\b") is None

    def test_rejects_null_bytes(self):
        """Test rejection of null bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretsProvider(base_path=tmpdir)
            
            assert provider.get_secret("secret\x00.txt") is None
            assert provider.get_secret("\x00") is None

    def test_rejects_empty_keys(self):
        """Test rejection of empty keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretsProvider(base_path=tmpdir)
            
            assert provider.get_secret("") is None
            assert provider.get_secret("   ") is None


class TestSQLInjectionPrevention:
    """Tests for SQL injection prevention in SQLite backend."""

    def test_rejects_invalid_tag_keys(self):
        """Test that invalid tag keys are rejected in queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(f"sqlite:///{db_path}")
            
            # SQL injection attempts in tag keys
            with pytest.raises(ValueError, match="Invalid tag key"):
                backend._build_where_clause(tags={"key'; DROP TABLE--": "value"})
            
            with pytest.raises(ValueError, match="Invalid tag key"):
                backend._build_where_clause(tags={"key)--": "value"})
            
            backend.close()

    def test_allows_valid_tag_keys(self):
        """Test that valid tag keys are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(f"sqlite:///{db_path}")
            
            # These should work
            where, params = backend._build_where_clause(
                tags={"team": "search", "env": "prod"}
            )
            assert "team" in where
            assert "env" in where
            
            backend.close()


class TestInputValidation:
    """Tests for input validation in CostTracker."""

    def test_rejects_empty_provider(self):
        """Test rejection of empty provider name."""
        tracker = CostTracker()
        
        with pytest.raises(ValueError, match="provider"):
            tracker.record(
                provider="",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        
        tracker.close()

    def test_rejects_empty_model(self):
        """Test rejection of empty model name."""
        tracker = CostTracker()
        
        with pytest.raises(ValueError, match="model"):
            tracker.record(
                provider="openai",
                model="",
                input_tokens=100,
                output_tokens=50,
            )
        
        tracker.close()

    def test_rejects_negative_tokens(self):
        """Test rejection of negative token counts."""
        tracker = CostTracker()
        
        with pytest.raises(ValueError, match="input_tokens"):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=-1,
                output_tokens=50,
            )
        
        with pytest.raises(ValueError, match="output_tokens"):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=-1,
            )
        
        tracker.close()

    def test_accepts_valid_model_names(self):
        """Test acceptance of valid model names with special chars."""
        tracker = CostTracker()
        
        # Test that model names with various chars are accepted
        # Note: We use openai models that exist in pricing data
        valid_models = [
            ("openai", "gpt-4o"),
            ("openai", "gpt-4-0613"),
            ("openai", "gpt-4-turbo"),
            ("anthropic", "claude-3-sonnet-20240229"),
        ]
        
        for provider, model in valid_models:
            record = tracker.record(
                provider=provider,
                model=model,
                input_tokens=100,
                output_tokens=50,
            )
            assert record.model == model
        
        tracker.close()


class TestHashingSecurity:
    """Tests for hashing security in HashedEncryption."""

    def test_hmac_produces_consistent_output(self):
        """Test that HMAC produces consistent hashes."""
        from llm_cost_guard.encryption import HashedEncryption
        
        provider = HashedEncryption(salt="secure-test-salt!")
        
        hash1 = provider.encrypt(b"user@example.com")
        hash2 = provider.encrypt(b"user@example.com")
        
        assert hash1 == hash2

    def test_different_salts_produce_different_hashes(self):
        """Test that different salts produce different hashes."""
        from llm_cost_guard.encryption import HashedEncryption
        
        provider1 = HashedEncryption(salt="salt-one-secure!")
        provider2 = HashedEncryption(salt="salt-two-secure!")
        
        hash1 = provider1.encrypt(b"same data")
        hash2 = provider2.encrypt(b"same data")
        
        assert hash1 != hash2

    def test_hash_length_is_sha256(self):
        """Test that hash length matches SHA-256 output."""
        from llm_cost_guard.encryption import HashedEncryption
        
        provider = HashedEncryption(salt="test-salt-16char")
        hash_result = provider.encrypt(b"test data")
        
        # SHA-256 hex digest is 64 characters
        assert len(hash_result) == 64
