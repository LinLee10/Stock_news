#!/usr/bin/env python3
"""
API module for Stonk News system
Provides REST endpoints when API features are enabled
"""
from config.feature_flags import feature_flags

if feature_flags.is_enabled('enable_api_endpoints'):
    from .app import create_app
    from .health import health_blueprint
    
    __all__ = ['create_app', 'health_blueprint']
else:
    # API disabled - provide no-op exports
    def create_app():
        raise RuntimeError("API endpoints are disabled. Set ENABLE_API_ENDPOINTS=true")
    
    __all__ = ['create_app']