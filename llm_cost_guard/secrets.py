"""
Pluggable secrets providers for LLM Cost Guard.

Supports any secrets backend - environment variables, Vault, or cloud providers.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SecretsProvider(ABC):
    """
    Abstract secrets provider.
    
    Implement this interface for any secrets backend:
    - Local: Environment variables (default)
    - HashiCorp: Vault
    - AWS: Secrets Manager
    - GCP: Secret Manager
    - Azure: Key Vault
    """

    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """
        Retrieve a secret value.
        
        Args:
            key: Secret key/name
            
        Returns:
            Secret value or None if not found
        """
        pass

    def get_secret_required(self, key: str) -> str:
        """
        Retrieve a secret value, raising if not found.
        
        Args:
            key: Secret key/name
            
        Returns:
            Secret value
            
        Raises:
            ValueError: If secret is not found
        """
        value = self.get_secret(key)
        if value is None:
            raise ValueError(f"Required secret not found: {key}")
        return value

    def get_json_secret(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and parse a JSON secret.
        
        Args:
            key: Secret key/name
            
        Returns:
            Parsed JSON as dict, or None if not found
        """
        import json
        value = self.get_secret(key)
        if value is None:
            return None
        return json.loads(value)


class EnvironmentSecretsProvider(SecretsProvider):
    """
    Read secrets from environment variables.
    
    This is the default provider - works everywhere without dependencies.
    
    Usage:
        provider = EnvironmentSecretsProvider(prefix="LLM_COST_GUARD_")
        redis_url = provider.get_secret("REDIS_URL")
        # Reads from LLM_COST_GUARD_REDIS_URL
    """

    def __init__(self, prefix: str = ""):
        """
        Initialize with optional prefix.
        
        Args:
            prefix: Prefix to add to all secret keys
        """
        self._prefix = prefix

    def get_secret(self, key: str) -> Optional[str]:
        """Read secret from environment variable."""
        env_key = f"{self._prefix}{key}"
        return os.environ.get(env_key)


class FileSecretsProvider(SecretsProvider):
    """
    Read secrets from files (useful for Kubernetes secrets mounted as volumes).
    
    Usage:
        provider = FileSecretsProvider(base_path="/var/secrets")
        redis_url = provider.get_secret("redis-url")
        # Reads from /var/secrets/redis-url
    
    Security:
        - Path traversal protection: Keys containing '..' or absolute paths are rejected
        - Only files within base_path can be read
    """

    def __init__(self, base_path: str):
        """
        Initialize with base path.
        
        Args:
            base_path: Directory containing secret files
        """
        self._base_path = os.path.abspath(base_path)

    def get_secret(self, key: str) -> Optional[str]:
        """
        Read secret from file.
        
        Security: Validates key to prevent path traversal attacks.
        """
        # Security: Validate key to prevent path traversal
        if not self._is_safe_key(key):
            logger.warning(f"Rejected unsafe secret key: {key}")
            return None
        
        file_path = os.path.join(self._base_path, key)
        
        # Security: Double-check resolved path is within base_path
        resolved_path = os.path.abspath(file_path)
        if not resolved_path.startswith(self._base_path + os.sep) and resolved_path != self._base_path:
            logger.warning(f"Path traversal attempt detected: {key}")
            return None
        
        try:
            with open(resolved_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
        except PermissionError:
            logger.warning(f"Permission denied reading secret: {key}")
            return None
        except IOError as e:
            logger.warning(f"Error reading secret {key}: {e}")
            return None

    @staticmethod
    def _is_safe_key(key: str) -> bool:
        """Validate that a key is safe (no path traversal)."""
        # Reject empty keys
        if not key or not key.strip():
            return False
        
        # Reject absolute paths
        if os.path.isabs(key):
            return False
        
        # Reject path traversal sequences
        if ".." in key:
            return False
        
        # Reject keys with path separators (be strict)
        if os.sep in key or "/" in key or "\\" in key:
            return False
        
        # Reject null bytes
        if "\x00" in key:
            return False
        
        return True


class VaultSecretsProvider(SecretsProvider):
    """
    HashiCorp Vault secrets provider.
    
    Cloud-agnostic - works on any infrastructure.
    
    Usage:
        provider = VaultSecretsProvider(
            url="https://vault.example.com",
            token="hvs.xxx",  # Or use VAULT_TOKEN env var
            path="secret/data/llm-cost-guard",
        )
        redis_url = provider.get_secret("redis_url")
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        path: str = "secret/data/llm-cost-guard",
        namespace: Optional[str] = None,
    ):
        """
        Initialize Vault client.
        
        Args:
            url: Vault URL (or VAULT_ADDR env var)
            token: Vault token (or VAULT_TOKEN env var)
            path: KV secrets path
            namespace: Vault namespace (Enterprise)
        """
        try:
            import hvac
        except ImportError:
            raise ImportError(
                "hvac package required for VaultSecretsProvider. "
                "Install with: pip install llm-cost-guard[vault]"
            )

        self._url = url or os.environ.get("VAULT_ADDR")
        self._token = token or os.environ.get("VAULT_TOKEN")
        self._path = path
        self._namespace = namespace

        if not self._url:
            raise ValueError("Vault URL required (url parameter or VAULT_ADDR env var)")

        self._client = hvac.Client(
            url=self._url,
            token=self._token,
            namespace=self._namespace,
        )
        
        # Cache secrets to reduce API calls
        self._cache: Optional[Dict[str, Any]] = None

    def _load_secrets(self) -> Dict[str, Any]:
        """Load all secrets from Vault path."""
        if self._cache is None:
            try:
                response = self._client.secrets.kv.v2.read_secret_version(
                    path=self._path.replace("secret/data/", "")
                )
                self._cache = response.get("data", {}).get("data", {})
            except Exception as e:
                logger.warning(f"Failed to load secrets from Vault: {e}")
                self._cache = {}
        return self._cache

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from Vault."""
        secrets = self._load_secrets()
        value = secrets.get(key)
        return str(value) if value is not None else None

    def refresh(self) -> None:
        """Clear cache to reload secrets."""
        self._cache = None


class CompositeSecretsProvider(SecretsProvider):
    """
    Chain multiple secrets providers with fallback.
    
    Tries each provider in order until a secret is found.
    
    Usage:
        provider = CompositeSecretsProvider([
            VaultSecretsProvider(...),  # Try Vault first
            EnvironmentSecretsProvider(),  # Fall back to env vars
        ])
    """

    def __init__(self, providers: list):
        """
        Initialize with list of providers.
        
        Args:
            providers: List of SecretsProvider instances (tried in order)
        """
        self._providers = providers

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from first provider that has it."""
        for provider in self._providers:
            try:
                value = provider.get_secret(key)
                if value is not None:
                    return value
            except Exception as e:
                logger.warning(f"Provider {type(provider).__name__} failed: {e}")
        return None


def get_secrets_provider(
    provider_type: str = "env",
    **kwargs: Any,
) -> SecretsProvider:
    """
    Factory function to create secrets providers.
    
    Args:
        provider_type: One of "env", "file", "vault", "aws", "gcp", "azure"
        **kwargs: Provider-specific configuration
        
    Returns:
        SecretsProvider instance
    """
    if provider_type == "env":
        return EnvironmentSecretsProvider(
            prefix=kwargs.get("prefix", "")
        )
    
    if provider_type == "file":
        base_path = kwargs.get("base_path")
        if not base_path:
            raise ValueError("FileSecretsProvider requires 'base_path' parameter")
        return FileSecretsProvider(base_path=base_path)
    
    if provider_type == "vault":
        return VaultSecretsProvider(**kwargs)
    
    if provider_type == "aws":
        try:
            from llm_cost_guard.secrets_aws import AWSSecretsProvider
            return AWSSecretsProvider(**kwargs)
        except ImportError:
            raise ImportError(
                "AWS Secrets Manager requires boto3. "
                "Install with: pip install llm-cost-guard[aws]"
            )
    
    if provider_type == "gcp":
        try:
            from llm_cost_guard.secrets_gcp import GCPSecretsProvider
            return GCPSecretsProvider(**kwargs)
        except ImportError:
            raise ImportError(
                "GCP Secret Manager requires google-cloud-secret-manager. "
                "Install with: pip install llm-cost-guard[gcp]"
            )
    
    if provider_type == "azure":
        try:
            from llm_cost_guard.secrets_azure import AzureSecretsProvider
            return AzureSecretsProvider(**kwargs)
        except ImportError:
            raise ImportError(
                "Azure Key Vault requires azure-keyvault-secrets. "
                "Install with: pip install llm-cost-guard[azure]"
            )
    
    raise ValueError(f"Unknown secrets provider: {provider_type}")
