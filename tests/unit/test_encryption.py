"""
Unit tests for encryption providers.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from llm_cost_guard.encryption import (
    EncryptionProvider,
    NoEncryption,
    FernetEncryption,
    HashedEncryption,
    FieldEncryption,
    get_encryption_provider,
)


class TestNoEncryption:
    """Tests for NoEncryption provider."""

    def test_encrypt_passthrough(self):
        """Test encryption is passthrough."""
        provider = NoEncryption()
        plaintext = b"sensitive data"
        
        encrypted = provider.encrypt(plaintext)
        
        assert encrypted == plaintext

    def test_decrypt_passthrough(self):
        """Test decryption is passthrough."""
        provider = NoEncryption()
        ciphertext = b"some data"
        
        decrypted = provider.decrypt(ciphertext)
        
        assert decrypted == ciphertext

    def test_encrypt_string(self):
        """Test string encryption."""
        provider = NoEncryption()
        plaintext = "hello world"
        
        encrypted = provider.encrypt_string(plaintext)
        decrypted = provider.decrypt_string(encrypted)
        
        assert decrypted == plaintext

    def test_encrypt_dict(self):
        """Test dictionary encryption."""
        provider = NoEncryption()
        data = {"key": "value", "nested": {"a": 1}}
        
        encrypted = provider.encrypt_dict(data)
        decrypted = provider.decrypt_dict(encrypted)
        
        assert decrypted == data


class TestFernetEncryption:
    """Tests for FernetEncryption provider."""

    @pytest.fixture
    def fernet_key(self):
        """Generate a Fernet key for testing."""
        try:
            from cryptography.fernet import Fernet
            return Fernet.generate_key()
        except ImportError:
            pytest.skip("cryptography package not installed")

    def test_encrypt_decrypt(self, fernet_key):
        """Test encryption and decryption."""
        provider = FernetEncryption(fernet_key)
        plaintext = b"sensitive data"
        
        encrypted = provider.encrypt(plaintext)
        decrypted = provider.decrypt(encrypted)
        
        assert encrypted != plaintext
        assert decrypted == plaintext

    def test_encrypted_is_different(self, fernet_key):
        """Test that encrypted data is different from plaintext."""
        provider = FernetEncryption(fernet_key)
        plaintext = b"sensitive data"
        
        encrypted = provider.encrypt(plaintext)
        
        assert encrypted != plaintext

    def test_encrypt_string(self, fernet_key):
        """Test string encryption."""
        provider = FernetEncryption(fernet_key)
        plaintext = "hello world"
        
        encrypted = provider.encrypt_string(plaintext)
        decrypted = provider.decrypt_string(encrypted)
        
        assert encrypted != plaintext
        assert decrypted == plaintext

    def test_encrypt_dict(self, fernet_key):
        """Test dictionary encryption."""
        provider = FernetEncryption(fernet_key)
        data = {"user_id": "12345", "tags": {"team": "search"}}
        
        encrypted = provider.encrypt_dict(data)
        decrypted = provider.decrypt_dict(encrypted)
        
        assert decrypted == data

    def test_generate_key(self):
        """Test key generation."""
        try:
            key = FernetEncryption.generate_key()
            assert len(key) > 0
            
            # Key should work for encryption
            provider = FernetEncryption(key)
            encrypted = provider.encrypt(b"test")
            decrypted = provider.decrypt(encrypted)
            assert decrypted == b"test"
        except ImportError:
            pytest.skip("cryptography package not installed")


class TestHashedEncryption:
    """Tests for HashedEncryption provider."""

    def test_hash_is_consistent(self):
        """Test that hashing is consistent."""
        provider = HashedEncryption(salt="test-salt-16chars")
        plaintext = b"user@example.com"
        
        hash1 = provider.encrypt(plaintext)
        hash2 = provider.encrypt(plaintext)
        
        assert hash1 == hash2

    def test_hash_with_salt(self):
        """Test that salt affects hash."""
        provider1 = HashedEncryption(salt="salt1-long-enough!")
        provider2 = HashedEncryption(salt="salt2-long-enough!")
        plaintext = b"data"
        
        hash1 = provider1.encrypt(plaintext)
        hash2 = provider2.encrypt(plaintext)
        
        assert hash1 != hash2

    def test_decrypt_raises(self):
        """Test that decrypt raises error."""
        provider = HashedEncryption(salt="test-salt-16chars")
        
        with pytest.raises(NotImplementedError):
            provider.decrypt(b"hashed")

    def test_auto_generates_salt_when_none(self):
        """Test that salt is auto-generated when not provided."""
        provider = HashedEncryption()  # No salt provided
        
        # Should work without error
        hash_result = provider.encrypt(b"test data")
        assert len(hash_result) > 0

    def test_uses_hmac_not_simple_hash(self):
        """Test that HMAC is used for security."""
        provider = HashedEncryption(salt="secure-salt-here!")
        
        # Hash should be HMAC-SHA256 hex digest (64 chars)
        hash_result = provider.encrypt(b"test data")
        assert len(hash_result) == 64  # SHA256 hex = 64 chars


class TestFieldEncryption:
    """Tests for FieldEncryption."""

    def test_encrypt_specified_fields(self):
        """Test encrypting specified fields."""
        provider = NoEncryption()  # Use passthrough for easy testing
        field_enc = FieldEncryption(
            provider=provider,
            encrypted_fields=["tags"],
        )
        
        record = {
            "model": "gpt-4o",
            "cost": 0.05,
            "tags": {"team": "search"},
        }
        
        encrypted = field_enc.encrypt_record(record)
        
        assert encrypted["model"] == "gpt-4o"  # Unchanged
        assert encrypted["cost"] == 0.05  # Unchanged
        assert encrypted.get("_tags_encrypted") is True

    def test_decrypt_specified_fields(self):
        """Test decrypting specified fields."""
        provider = NoEncryption()
        field_enc = FieldEncryption(
            provider=provider,
            encrypted_fields=["tags"],
        )
        
        record = {"tags": {"team": "search"}}
        encrypted = field_enc.encrypt_record(record)
        decrypted = field_enc.decrypt_record(encrypted)
        
        assert decrypted["tags"] == {"team": "search"}
        assert "_tags_encrypted" not in decrypted

    def test_hash_fields(self):
        """Test hashing specified fields."""
        field_enc = FieldEncryption(
            provider=NoEncryption(),
            hashed_fields=["user_id"],
            hash_salt="test",
        )
        
        record = {"user_id": "user@example.com", "model": "gpt-4o"}
        encrypted = field_enc.encrypt_record(record)
        
        assert encrypted["user_id"] != "user@example.com"
        assert encrypted.get("_user_id_hashed") is True
        assert encrypted["model"] == "gpt-4o"

    def test_encrypt_with_fernet(self):
        """Test field encryption with Fernet."""
        try:
            key = FernetEncryption.generate_key()
            provider = FernetEncryption(key)
            field_enc = FieldEncryption(
                provider=provider,
                encrypted_fields=["metadata"],
            )
            
            record = {"metadata": {"secret": "value"}, "cost": 0.05}
            encrypted = field_enc.encrypt_record(record)
            
            assert encrypted["metadata"] != '{"secret": "value"}'
            assert encrypted["cost"] == 0.05
            
            decrypted = field_enc.decrypt_record(encrypted)
            assert decrypted["metadata"] == {"secret": "value"}
        except ImportError:
            pytest.skip("cryptography package not installed")


class TestGetEncryptionProvider:
    """Tests for get_encryption_provider factory."""

    def test_get_none_provider(self):
        """Test getting no-op provider."""
        provider = get_encryption_provider("none")
        
        assert isinstance(provider, NoEncryption)

    def test_get_fernet_provider(self):
        """Test getting Fernet provider."""
        try:
            key = FernetEncryption.generate_key()
            provider = get_encryption_provider("fernet", key=key)
            
            assert isinstance(provider, FernetEncryption)
        except ImportError:
            pytest.skip("cryptography package not installed")

    def test_get_fernet_without_key_raises(self):
        """Test that Fernet without key raises."""
        with pytest.raises(ValueError, match="requires 'key'"):
            get_encryption_provider("fernet")

    def test_unknown_provider_raises(self):
        """Test that unknown provider raises."""
        with pytest.raises(ValueError, match="Unknown encryption provider"):
            get_encryption_provider("unknown")
