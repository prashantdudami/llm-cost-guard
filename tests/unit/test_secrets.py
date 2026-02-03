"""
Unit tests for secrets providers.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from llm_cost_guard.secrets import (
    SecretsProvider,
    EnvironmentSecretsProvider,
    FileSecretsProvider,
    CompositeSecretsProvider,
    get_secrets_provider,
)


class TestEnvironmentSecretsProvider:
    """Tests for EnvironmentSecretsProvider."""

    def test_get_secret_exists(self):
        """Test getting existing secret."""
        with patch.dict(os.environ, {"TEST_SECRET": "secret_value"}):
            provider = EnvironmentSecretsProvider()
            
            value = provider.get_secret("TEST_SECRET")
            
            assert value == "secret_value"

    def test_get_secret_not_exists(self):
        """Test getting non-existent secret."""
        provider = EnvironmentSecretsProvider()
        
        value = provider.get_secret("NONEXISTENT_SECRET_12345")
        
        assert value is None

    def test_get_secret_with_prefix(self):
        """Test getting secret with prefix."""
        with patch.dict(os.environ, {"APP_DATABASE_URL": "postgres://..."}):
            provider = EnvironmentSecretsProvider(prefix="APP_")
            
            value = provider.get_secret("DATABASE_URL")
            
            assert value == "postgres://..."

    def test_get_secret_required_exists(self):
        """Test get_secret_required with existing secret."""
        with patch.dict(os.environ, {"REQUIRED_SECRET": "value"}):
            provider = EnvironmentSecretsProvider()
            
            value = provider.get_secret_required("REQUIRED_SECRET")
            
            assert value == "value"

    def test_get_secret_required_not_exists(self):
        """Test get_secret_required raises for missing secret."""
        provider = EnvironmentSecretsProvider()
        
        with pytest.raises(ValueError, match="Required secret not found"):
            provider.get_secret_required("NONEXISTENT_SECRET_12345")

    def test_get_json_secret(self):
        """Test getting JSON secret."""
        json_value = json.dumps({"host": "localhost", "port": 5432})
        with patch.dict(os.environ, {"JSON_SECRET": json_value}):
            provider = EnvironmentSecretsProvider()
            
            value = provider.get_json_secret("JSON_SECRET")
            
            assert value == {"host": "localhost", "port": 5432}

    def test_get_json_secret_not_exists(self):
        """Test getting non-existent JSON secret."""
        provider = EnvironmentSecretsProvider()
        
        value = provider.get_json_secret("NONEXISTENT_JSON_SECRET")
        
        assert value is None


class TestFileSecretsProvider:
    """Tests for FileSecretsProvider."""

    def test_get_secret_from_file(self):
        """Test getting secret from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create secret file
            secret_path = os.path.join(tmpdir, "db-password")
            with open(secret_path, "w") as f:
                f.write("super-secret-password\n")
            
            provider = FileSecretsProvider(base_path=tmpdir)
            
            value = provider.get_secret("db-password")
            
            assert value == "super-secret-password"

    def test_get_secret_file_not_exists(self):
        """Test getting secret when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretsProvider(base_path=tmpdir)
            
            value = provider.get_secret("nonexistent")
            
            assert value is None

    def test_get_secret_strips_whitespace(self):
        """Test that whitespace is stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = os.path.join(tmpdir, "secret")
            with open(secret_path, "w") as f:
                f.write("  value with spaces  \n\n")
            
            provider = FileSecretsProvider(base_path=tmpdir)
            
            value = provider.get_secret("secret")
            
            assert value == "value with spaces"


class TestCompositeSecretsProvider:
    """Tests for CompositeSecretsProvider."""

    def test_first_provider_wins(self):
        """Test that first provider with secret wins."""
        with patch.dict(os.environ, {"SHARED_SECRET": "from_env"}):
            env_provider = EnvironmentSecretsProvider()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                secret_path = os.path.join(tmpdir, "SHARED_SECRET")
                with open(secret_path, "w") as f:
                    f.write("from_file")
                file_provider = FileSecretsProvider(base_path=tmpdir)
                
                # Env provider first
                composite = CompositeSecretsProvider([env_provider, file_provider])
                
                value = composite.get_secret("SHARED_SECRET")
                
                assert value == "from_env"

    def test_fallback_to_second_provider(self):
        """Test fallback to second provider."""
        env_provider = EnvironmentSecretsProvider()  # Won't have this secret
        
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = os.path.join(tmpdir, "FILE_ONLY_SECRET")
            with open(secret_path, "w") as f:
                f.write("from_file")
            file_provider = FileSecretsProvider(base_path=tmpdir)
            
            composite = CompositeSecretsProvider([env_provider, file_provider])
            
            value = composite.get_secret("FILE_ONLY_SECRET")
            
            assert value == "from_file"

    def test_none_if_no_provider_has_secret(self):
        """Test None if no provider has secret."""
        env_provider = EnvironmentSecretsProvider()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_provider = FileSecretsProvider(base_path=tmpdir)
            
            composite = CompositeSecretsProvider([env_provider, file_provider])
            
            value = composite.get_secret("NONEXISTENT_12345")
            
            assert value is None

    def test_continues_on_provider_error(self):
        """Test that errors in one provider don't stop others."""
        # Create a failing provider
        failing_provider = MagicMock(spec=SecretsProvider)
        failing_provider.get_secret.side_effect = Exception("Provider error")
        
        with patch.dict(os.environ, {"FALLBACK_SECRET": "value"}):
            env_provider = EnvironmentSecretsProvider()
            
            composite = CompositeSecretsProvider([failing_provider, env_provider])
            
            value = composite.get_secret("FALLBACK_SECRET")
            
            assert value == "value"


class TestGetSecretsProvider:
    """Tests for get_secrets_provider factory."""

    def test_get_env_provider(self):
        """Test getting environment provider."""
        provider = get_secrets_provider("env")
        
        assert isinstance(provider, EnvironmentSecretsProvider)

    def test_get_env_provider_with_prefix(self):
        """Test getting environment provider with prefix."""
        provider = get_secrets_provider("env", prefix="APP_")
        
        assert provider._prefix == "APP_"

    def test_get_file_provider(self):
        """Test getting file provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = get_secrets_provider("file", base_path=tmpdir)
            
            assert isinstance(provider, FileSecretsProvider)

    def test_get_file_provider_without_path_raises(self):
        """Test that file provider without path raises."""
        with pytest.raises(ValueError, match="requires 'base_path'"):
            get_secrets_provider("file")

    def test_unknown_provider_raises(self):
        """Test that unknown provider raises."""
        with pytest.raises(ValueError, match="Unknown secrets provider"):
            get_secrets_provider("unknown")
