#!/usr/bin/env python3
"""
Secrets loader utility for easy API key management
Supports multiple environments and secure key access
"""

import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SecretsManager:
    """Centralized secrets management"""
    
    SECRETS_DIR = Path(__file__).parent
    SUPPORTED_ENVIRONMENTS = ['production', 'development', 'testing']
    
    def __init__(self, environment: str = 'development'):
        self.environment = environment
        self._secrets_cache = {}
        
    def load_secrets(self, environment: Optional[str] = None) -> Dict[str, str]:
        """Load secrets from environment file"""
        env = environment or self.environment
        
        if env in self._secrets_cache:
            return self._secrets_cache[env]
            
        if env not in self.SUPPORTED_ENVIRONMENTS:
            raise ValueError(f"Unsupported environment: {env}. Use one of: {self.SUPPORTED_ENVIRONMENTS}")
        
        env_file = self.SECRETS_DIR / f"{env}.env"
        
        if not env_file.exists():
            logger.warning(f"Secrets file not found: {env_file}")
            return {}
            
        secrets = {}
        
        try:
            with open(env_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#') or line.startswith('='):
                        continue
                        
                    if '=' not in line:
                        logger.warning(f"Invalid line {line_num} in {env_file}: {line}")
                        continue
                        
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                        
                    secrets[key] = value
                    
            self._secrets_cache[env] = secrets
            logger.info(f"Loaded {len(secrets)} secrets from {env} environment")
            return secrets
            
        except Exception as e:
            logger.error(f"Error loading secrets from {env_file}: {e}")
            return {}
    
    def get_secret(self, key: str, environment: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
        """Get a specific secret by key"""
        secrets = self.load_secrets(environment)
        return secrets.get(key, default)
    
    def get_news_api_key(self, service: str = 'newsapi', environment: Optional[str] = None) -> Optional[str]:
        """Get news service API key"""
        key_mapping = {
            'newsapi': 'NEWSAPI_KEY',
            'gnews': 'GNEWS_KEY'
        }
        
        key_name = key_mapping.get(service.lower())
        if not key_name:
            raise ValueError(f"Unknown news service: {service}. Supported: {list(key_mapping.keys())}")
            
        return self.get_secret(key_name, environment)
    
    def get_financial_api_key(self, service: str, environment: Optional[str] = None) -> Optional[str]:
        """Get financial data service API key"""
        key_mapping = {
            'finnhub': 'FINNHUB_KEY',
            'twelvedata': 'TWELVEDATA_KEY', 
            'tiingo': 'TIINGO_KEY',
            'polygon': 'POLYGON_KEY',
            'marketstack': 'MARKETSTACK_KEY',
            'eodhd': 'EODHD_KEY',
            'nasdaq': 'NASDAQ_DATALINK_KEY',
            'alpaca_key': 'ALPACA_API_KEY',
            'alpaca_secret': 'ALPACA_SECRET_KEY'
        }
        
        key_name = key_mapping.get(service.lower())
        if not key_name:
            raise ValueError(f"Unknown financial service: {service}. Supported: {list(key_mapping.keys())}")
            
        return self.get_secret(key_name, environment)
    
    def get_government_api_key(self, service: str, environment: Optional[str] = None) -> Optional[str]:
        """Get government data service API key"""
        key_mapping = {
            'fred': 'FRED_KEY',
            'bea': 'BEA_KEY', 
            'bls': 'BLS_KEY'
        }
        
        key_name = key_mapping.get(service.lower())
        if not key_name:
            raise ValueError(f"Unknown government service: {service}. Supported: {list(key_mapping.keys())}")
            
        return self.get_secret(key_name, environment)
    
    def get_alpaca_credentials(self, environment: Optional[str] = None) -> Dict[str, str]:
        """Get Alpaca trading credentials (both key and secret)"""
        secrets = self.load_secrets(environment)
        return {
            'api_key': secrets.get('ALPACA_API_KEY'),
            'secret_key': secrets.get('ALPACA_SECRET_KEY')
        }
    
    def list_available_keys(self, environment: Optional[str] = None) -> list:
        """List all available API keys for debugging"""
        secrets = self.load_secrets(environment)
        return sorted(secrets.keys())


# Global instance for easy access
_default_manager = SecretsManager()

# Convenience functions
def load_secrets(environment: str = 'development') -> Dict[str, str]:
    """Load secrets from specified environment"""
    return _default_manager.load_secrets(environment)

def get_secret(key: str, environment: str = 'development', default: Optional[str] = None) -> Optional[str]:
    """Get a specific secret"""
    return _default_manager.get_secret(key, environment, default)

def get_news_api_key(service: str = 'newsapi', environment: str = 'development') -> Optional[str]:
    """Get news API key"""
    return _default_manager.get_news_api_key(service, environment)

def get_financial_api_key(service: str, environment: str = 'development') -> Optional[str]:
    """Get financial API key"""
    return _default_manager.get_financial_api_key(service, environment)

def get_government_api_key(service: str, environment: str = 'development') -> Optional[str]:
    """Get government API key"""
    return _default_manager.get_government_api_key(service, environment)

def get_alpaca_credentials(environment: str = 'development') -> Dict[str, str]:
    """Get Alpaca credentials"""
    return _default_manager.get_alpaca_credentials(environment)


if __name__ == "__main__":
    # Demo usage
    print("🔐 Secrets Manager Demo")
    print("=" * 50)
    
    manager = SecretsManager('development')
    
    # List available keys
    print(f"Available keys: {manager.list_available_keys()}")
    
    # Test specific services
    print(f"NewsAPI key: {manager.get_news_api_key('newsapi')}")
    print(f"Finnhub key: {manager.get_financial_api_key('finnhub')}")
    print(f"FRED key: {manager.get_government_api_key('fred')}")
    
    # Test Alpaca credentials
    alpaca_creds = manager.get_alpaca_credentials()
    print(f"Alpaca API key: {alpaca_creds['api_key']}")
    print(f"Alpaca secret: {'*' * len(alpaca_creds['secret_key'] or '')}")  # Mask secret