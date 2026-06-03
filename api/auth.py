#!/usr/bin/env python3
"""
API Authentication middleware for Stonk News API
Implements API key-based authentication for secure endpoints
"""
from functools import wraps
from flask import request, jsonify, current_app
from config.config import API_KEY
import logging

logger = logging.getLogger(__name__)

def require_api_key(f):
    """
    Decorator to require X-API-Key header for endpoint access
    Returns 401 if key is missing, 403 if key is invalid
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if API key is configured
        if not API_KEY:
            logger.error("API_KEY not configured but authentication required")
            return jsonify({
                'error': 'Server configuration error',
                'code': 'MISSING_API_CONFIG'
            }), 500
        
        # Extract API key from headers
        provided_key = request.headers.get('X-API-Key')
        
        if not provided_key:
            logger.warning(f"API access attempt without key from {request.remote_addr}")
            return jsonify({
                'error': 'API key required',
                'details': 'Include X-API-Key header with valid API key',
                'code': 'MISSING_API_KEY'
            }), 401
        
        if provided_key != API_KEY:
            logger.warning(f"Invalid API key attempt from {request.remote_addr}")
            return jsonify({
                'error': 'Invalid API key',
                'code': 'INVALID_API_KEY'
            }), 403
        
        # Log successful authentication
        logger.debug(f"Authenticated API request to {request.endpoint}")
        return f(*args, **kwargs)
    
    return decorated_function

def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key against configured key
    Used for testing and validation
    """
    return api_key == API_KEY if API_KEY else False