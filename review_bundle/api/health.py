#!/usr/bin/env python3
"""
Health and configuration debug endpoints
"""
from flask import Blueprint, jsonify
from config.feature_flags import feature_flags
from config.config import ALPHA_VANTAGE_KEY
from datetime import datetime
import os
import sys

health_blueprint = Blueprint('health', __name__)

@health_blueprint.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'service': 'stonk-news'
    })

@health_blueprint.route('/config', methods=['GET'])
def config_debug():
    """Debug endpoint to show current configuration and feature flags"""
    
    # Only show in debug mode or with explicit flag
    if not (feature_flags.is_enabled('enable_debug_mode') or 
            os.getenv('SHOW_CONFIG_DEBUG', 'false').lower() == 'true'):
        return jsonify({'error': 'Config debug not enabled'}), 403
    
    config_info = {
        'feature_flags': feature_flags.get_all_flags(),
        'api_config': {
            'host': os.getenv('API_HOST', 'localhost'),
            'port': os.getenv('API_PORT', '8000'),
            'debug': os.getenv('API_DEBUG', 'false')
        },
        'data_sources': {
            'alpha_vantage_configured': bool(ALPHA_VANTAGE_KEY),
            'alpha_vantage_key_length': len(ALPHA_VANTAGE_KEY) if ALPHA_VANTAGE_KEY else 0
        },
        'environment': {
            'python_version': sys.version,
            'working_directory': os.getcwd()
        },
        'timestamp': datetime.utcnow().isoformat()
    }
    
    return jsonify(config_info)

@health_blueprint.route('/flags', methods=['GET'])
def feature_flags_status():
    """Endpoint specifically for feature flag status"""
    return jsonify({
        'flags': feature_flags.get_all_flags(),
        'timestamp': datetime.utcnow().isoformat()
    })