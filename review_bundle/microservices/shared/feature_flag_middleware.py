"""
Feature Flag Middleware for Flask/FastAPI Integration
"""

import json
import time
from typing import Dict, Any, Optional, Callable, Awaitable
from functools import wraps

import structlog
from flask import Flask, request, g
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware

from .feature_flags import FeatureFlagManager, User, UserSegment

logger = structlog.get_logger(__name__)


class FeatureFlagFlaskMiddleware:
    """Flask middleware for feature flag integration"""
    
    def __init__(self, app: Flask, flag_manager: FeatureFlagManager):
        self.app = app
        self.flag_manager = flag_manager
        self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize Flask app with feature flag middleware"""
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.teardown_appcontext(self._teardown)
    
    def _before_request(self):
        """Process request and set up feature flag context"""
        g.feature_flags = {}
        g.user_context = self._extract_user_context()
        g.request_start_time = time.time()
    
    def _after_request(self, response):
        """Process response and track feature flag events"""
        if hasattr(g, 'feature_flags') and g.feature_flags:
            response.headers['X-Feature-Flags'] = json.dumps(list(g.feature_flags.keys()))
        return response
    
    def _teardown(self, exception):
        """Clean up feature flag context"""
        pass
    
    def _extract_user_context(self) -> Optional[User]:
        """Extract user context from request"""
        try:
            # Try to get user from JWT token, session, or headers
            user_id = request.headers.get('X-User-ID')
            if not user_id:
                return None
            
            # Extract user attributes from headers or decode from JWT
            risk_profile = UserSegment(request.headers.get('X-Risk-Profile', 'moderate'))
            account_type = UserSegment(request.headers.get('X-Account-Type', 'free'))
            subscription_tier = request.headers.get('X-Subscription-Tier', 'basic')
            region = request.headers.get('X-User-Region', 'us')
            
            return User(
                user_id=user_id,
                risk_profile=risk_profile,
                account_type=account_type,
                subscription_tier=subscription_tier,
                region=region,
                signup_date=None,  # Would need to fetch from database
                portfolio_value=float(request.headers.get('X-Portfolio-Value', '0')),
                trading_frequency=request.headers.get('X-Trading-Frequency', 'low'),
                beta_tester=request.headers.get('X-Beta-Tester', 'false').lower() == 'true'
            )
        
        except Exception as e:
            logger.warning("Failed to extract user context", error=str(e))
            return None


class FeatureFlagFastAPIMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for feature flag integration"""
    
    def __init__(self, app: FastAPI, flag_manager: FeatureFlagManager):
        super().__init__(app)
        self.flag_manager = flag_manager
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable]):
        """Process request with feature flag context"""
        start_time = time.time()
        
        # Extract user context
        user_context = await self._extract_user_context(request)
        request.state.user_context = user_context
        request.state.feature_flags = {}
        request.state.request_start_time = start_time
        
        # Process request
        response = await call_next(request)
        
        # Add feature flag headers
        if hasattr(request.state, 'feature_flags') and request.state.feature_flags:
            response.headers['X-Feature-Flags'] = json.dumps(list(request.state.feature_flags.keys()))
        
        return response
    
    async def _extract_user_context(self, request: Request) -> Optional[User]:
        """Extract user context from request"""
        try:
            user_id = request.headers.get('x-user-id')
            if not user_id:
                return None
            
            risk_profile = UserSegment(request.headers.get('x-risk-profile', 'moderate'))
            account_type = UserSegment(request.headers.get('x-account-type', 'free'))
            subscription_tier = request.headers.get('x-subscription-tier', 'basic')
            region = request.headers.get('x-user-region', 'us')
            
            return User(
                user_id=user_id,
                risk_profile=risk_profile,
                account_type=account_type,
                subscription_tier=subscription_tier,
                region=region,
                signup_date=None,
                portfolio_value=float(request.headers.get('x-portfolio-value', '0')),
                trading_frequency=request.headers.get('x-trading-frequency', 'low'),
                beta_tester=request.headers.get('x-beta-tester', 'false').lower() == 'true'
            )
        
        except Exception as e:
            logger.warning("Failed to extract user context", error=str(e))
            return None


# Decorator functions for feature flag checks

def feature_flag_required(flag_name: str, default_behavior: str = "block"):
    """
    Decorator to require a feature flag to be enabled
    
    Args:
        flag_name: Name of the feature flag
        default_behavior: What to do if flag is disabled ("block", "fallback", "log")
    """
    def decorator(func):
        if hasattr(func, '__self__'):  # Method
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get flag manager from instance or global context
                flag_manager = getattr(args[0], 'flag_manager', None)
                if not flag_manager:
                    return await func(*args, **kwargs)
                
                user = await _get_current_user()
                if not user:
                    if default_behavior == "block":
                        raise HTTPException(status_code=403, detail="Feature not available")
                    return await func(*args, **kwargs)
                
                enabled = await flag_manager.is_enabled(flag_name, user)
                if not enabled:
                    if default_behavior == "block":
                        raise HTTPException(status_code=403, detail="Feature not available")
                    elif default_behavior == "log":
                        logger.info("Feature flag disabled", flag=flag_name, user=user.user_id)
                
                # Track feature usage
                await flag_manager.track_custom_event(flag_name, user, "feature_accessed")
                return await func(*args, **kwargs)
            
            return async_wrapper
        else:  # Function
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, would need different implementation
                return func(*args, **kwargs)
            
            return sync_wrapper
    
    return decorator


def ab_test_variation(flag_name: str, variations: Dict[str, Callable]):
    """
    Decorator to run different code paths based on A/B test variation
    
    Args:
        flag_name: Name of the A/B test flag
        variations: Dict mapping variation names to functions
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            flag_manager = await _get_flag_manager()
            user = await _get_current_user()
            
            if not flag_manager or not user:
                return await func(*args, **kwargs)
            
            variation = await flag_manager.get_variation(flag_name, user)
            
            if variation and variation in variations:
                # Track variation usage
                await flag_manager.track_custom_event(flag_name, user, "variation_used")
                return await variations[variation](*args, **kwargs)
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Helper functions

async def _get_current_user() -> Optional[User]:
    """Get current user from request context"""
    try:
        # Try Flask first
        from flask import g
        if hasattr(g, 'user_context'):
            return g.user_context
    except ImportError:
        pass
    
    try:
        # Try FastAPI
        from fastapi import Request
        # This would need to be passed through context or dependency injection
        pass
    except ImportError:
        pass
    
    return None


async def _get_flag_manager() -> Optional[FeatureFlagManager]:
    """Get feature flag manager from context"""
    # This would need to be implemented based on your DI/context system
    return None


class FeatureFlagDecorators:
    """Class-based decorators with access to flag manager"""
    
    def __init__(self, flag_manager: FeatureFlagManager):
        self.flag_manager = flag_manager
    
    def require_flag(self, flag_name: str, default_behavior: str = "block"):
        """Require feature flag to be enabled"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get user from request context
                user = await self._extract_user_from_args(args, kwargs)
                
                if not user:
                    if default_behavior == "block":
                        raise HTTPException(status_code=403, detail="Authentication required")
                    return await func(*args, **kwargs)
                
                enabled = await self.flag_manager.is_enabled(flag_name, user)
                if not enabled:
                    if default_behavior == "block":
                        raise HTTPException(status_code=403, detail="Feature not available")
                    elif default_behavior == "log":
                        logger.info("Feature flag disabled", flag=flag_name, user=user.user_id)
                
                await self.flag_manager.track_custom_event(flag_name, user, "feature_accessed")
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def ab_test(self, flag_name: str):
        """Enable A/B testing for function"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = await self._extract_user_from_args(args, kwargs)
                
                if user:
                    # Get variation and configuration
                    variation = await self.flag_manager.get_variation(flag_name, user)
                    config = await self.flag_manager.get_config(flag_name, user)
                    
                    # Add variation info to function context
                    kwargs['_ab_test_variation'] = variation
                    kwargs['_ab_test_config'] = config
                    
                    # Track usage
                    await self.flag_manager.track_custom_event(flag_name, user, "function_called")
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    async def _extract_user_from_args(self, args, kwargs) -> Optional[User]:
        """Extract user from function arguments"""
        # Look for user in kwargs
        if 'user' in kwargs:
            return kwargs['user']
        
        # Look for request object with user context
        if 'request' in kwargs:
            request = kwargs['request']
            if hasattr(request, 'state') and hasattr(request.state, 'user_context'):
                return request.state.user_context
        
        # Look for user_id and build user object
        if 'user_id' in kwargs:
            # Would need to fetch user details from database
            pass
        
        return None


# Context managers for feature flag scopes

class FeatureFlagScope:
    """Context manager for feature flag scope"""
    
    def __init__(self, flag_manager: FeatureFlagManager, user: User):
        self.flag_manager = flag_manager
        self.user = user
        self.active_flags = set()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Track scope completion
        for flag_name in self.active_flags:
            await self.flag_manager.track_custom_event(flag_name, self.user, "scope_completed")
    
    async def is_enabled(self, flag_name: str) -> bool:
        """Check if flag is enabled and track usage"""
        enabled = await self.flag_manager.is_enabled(flag_name, self.user)
        if enabled:
            self.active_flags.add(flag_name)
            await self.flag_manager.track_custom_event(flag_name, self.user, "flag_checked")
        return enabled
    
    async def get_config(self, flag_name: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get flag configuration and track usage"""
        config = await self.flag_manager.get_config(flag_name, self.user, default or {})
        if config:
            self.active_flags.add(flag_name)
            await self.flag_manager.track_custom_event(flag_name, self.user, "config_retrieved")
        return config
    
    async def track_conversion(self, flag_name: str, value: float = 1.0):
        """Track conversion for A/B test"""
        await self.flag_manager.track_conversion(flag_name, self.user, value)