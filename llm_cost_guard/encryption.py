"""
Pluggable encryption providers for LLM Cost Guard.

Supports any encryption backend - local (Fernet) or cloud KMS.
"""

import base64
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EncryptionProvider(ABC):
    """
    Abstract encryption provider.
    
    Implement this interface for any encryption backend:
    - Local: Fernet (default)
    - AWS: KMS
    - GCP: Cloud KMS
    - Azure: Key Vault
    - HashiCorp: Vault Transit
    """

    @abstractmethod
    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt plaintext bytes."""
        pass

    @abstractmethod
    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext bytes."""
        pass

    def encrypt_string(self, plaintext: str) -> str:
        """Encrypt a string and return base64-encoded ciphertext."""
        encrypted = self.encrypt(plaintext.encode("utf-8"))
        return base64.b64encode(encrypted).decode("utf-8")

    def decrypt_string(self, ciphertext: str) -> str:
        """Decrypt a base64-encoded ciphertext string."""
        encrypted = base64.b64decode(ciphertext.encode("utf-8"))
        return self.decrypt(encrypted).decode("utf-8")

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt a dictionary as JSON."""
        json_str = json.dumps(data, default=str)
        return self.encrypt_string(json_str)

    def decrypt_dict(self, ciphertext: str) -> Dict[str, Any]:
        """Decrypt a dictionary from encrypted JSON."""
        json_str = self.decrypt_string(ciphertext)
        return json.loads(json_str)


class NoEncryption(EncryptionProvider):
    """
    No encryption (passthrough).
    
    Use for development/testing or when encryption is handled
    at the storage layer (encrypted volumes, etc.).
    """

    def encrypt(self, plaintext: bytes) -> bytes:
        return plaintext

    def decrypt(self, ciphertext: bytes) -> bytes:
        return ciphertext


class FernetEncryption(EncryptionProvider):
    """
    Local encryption using Fernet (symmetric encryption).
    
    Cloud-agnostic - works anywhere without external dependencies.
    
    Usage:
        # Generate a key (do this once, store securely)
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        
        # Use the key
        provider = FernetEncryption(key)
        encrypted = provider.encrypt(b"sensitive data")
    """

    def __init__(self, key: bytes):
        """
        Initialize with a Fernet key.
        
        Args:
            key: 32-byte URL-safe base64-encoded key.
                 Generate with: Fernet.generate_key()
        """
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError(
                "cryptography package required for FernetEncryption. "
                "Install with: pip install llm-cost-guard[encryption]"
            )
        
        self._fernet = Fernet(key)

    def encrypt(self, plaintext: bytes) -> bytes:
        return self._fernet.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self._fernet.decrypt(ciphertext)

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a new Fernet key."""
        from cryptography.fernet import Fernet
        return Fernet.generate_key()


class HashedEncryption(EncryptionProvider):
    """
    One-way hashing for PII fields (not reversible).
    
    Use for fields like user_id where you need consistency
    but don't need to recover the original value.
    """

    def __init__(self, salt: str = ""):
        """
        Initialize with optional salt.
        
        Args:
            salt: Salt to add before hashing (recommended for security)
        """
        self._salt = salt.encode("utf-8")

    def encrypt(self, plaintext: bytes) -> bytes:
        """Hash the plaintext (one-way, not reversible)."""
        return hashlib.sha256(self._salt + plaintext).hexdigest().encode("utf-8")

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Cannot decrypt hashed data."""
        raise NotImplementedError("HashedEncryption is one-way and cannot be decrypted")


class FieldEncryption:
    """
    Field-level encryption for CostRecord fields.
    
    Encrypts specified fields while leaving others in plaintext.
    
    Usage:
        field_enc = FieldEncryption(
            provider=FernetEncryption(key),
            encrypted_fields=["tags", "metadata"],
            hashed_fields=["user_id"],
        )
    """

    def __init__(
        self,
        provider: EncryptionProvider,
        encrypted_fields: Optional[list] = None,
        hashed_fields: Optional[list] = None,
        hash_salt: str = "",
    ):
        """
        Initialize field encryption.
        
        Args:
            provider: Encryption provider for reversible encryption
            encrypted_fields: List of field names to encrypt
            hashed_fields: List of field names to hash (one-way)
            hash_salt: Salt for hashing
        """
        self._provider = provider
        self._encrypted_fields = set(encrypted_fields or [])
        self._hashed_fields = set(hashed_fields or [])
        self._hasher = HashedEncryption(salt=hash_salt)

    def encrypt_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt specified fields in a record."""
        result = dict(record)
        
        for field in self._encrypted_fields:
            if field in result and result[field] is not None:
                value = result[field]
                if isinstance(value, dict):
                    result[field] = self._provider.encrypt_dict(value)
                elif isinstance(value, str):
                    result[field] = self._provider.encrypt_string(value)
                else:
                    result[field] = self._provider.encrypt_string(str(value))
                result[f"_{field}_encrypted"] = True
        
        for field in self._hashed_fields:
            if field in result and result[field] is not None:
                value = str(result[field])
                result[field] = self._hasher.encrypt(value.encode()).decode()
                result[f"_{field}_hashed"] = True
        
        return result

    def decrypt_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt specified fields in a record."""
        result = dict(record)
        
        for field in self._encrypted_fields:
            if result.get(f"_{field}_encrypted") and field in result:
                try:
                    # Try to decrypt as dict first
                    result[field] = self._provider.decrypt_dict(result[field])
                except json.JSONDecodeError:
                    # Fall back to string
                    result[field] = self._provider.decrypt_string(result[field])
                del result[f"_{field}_encrypted"]
        
        # Hashed fields cannot be decrypted
        for field in self._hashed_fields:
            if f"_{field}_hashed" in result:
                del result[f"_{field}_hashed"]
        
        return result


def get_encryption_provider(
    provider_type: str = "none",
    **kwargs: Any,
) -> EncryptionProvider:
    """
    Factory function to create encryption providers.
    
    Args:
        provider_type: One of "none", "fernet", "aws_kms", "gcp_kms", "azure_kv"
        **kwargs: Provider-specific configuration
        
    Returns:
        EncryptionProvider instance
    """
    if provider_type == "none":
        return NoEncryption()
    
    if provider_type == "fernet":
        key = kwargs.get("key")
        if not key:
            raise ValueError("FernetEncryption requires 'key' parameter")
        return FernetEncryption(key)
    
    if provider_type == "aws_kms":
        # Lazy import to avoid dependency
        try:
            from llm_cost_guard.encryption_aws import AWSKMSEncryption
            return AWSKMSEncryption(**kwargs)
        except ImportError:
            raise ImportError(
                "AWS KMS encryption requires boto3. "
                "Install with: pip install llm-cost-guard[aws]"
            )
    
    if provider_type == "gcp_kms":
        try:
            from llm_cost_guard.encryption_gcp import GCPKMSEncryption
            return GCPKMSEncryption(**kwargs)
        except ImportError:
            raise ImportError(
                "GCP KMS encryption requires google-cloud-kms. "
                "Install with: pip install llm-cost-guard[gcp]"
            )
    
    if provider_type == "azure_kv":
        try:
            from llm_cost_guard.encryption_azure import AzureKeyVaultEncryption
            return AzureKeyVaultEncryption(**kwargs)
        except ImportError:
            raise ImportError(
                "Azure Key Vault encryption requires azure-keyvault-keys. "
                "Install with: pip install llm-cost-guard[azure]"
            )
    
    raise ValueError(f"Unknown encryption provider: {provider_type}")
