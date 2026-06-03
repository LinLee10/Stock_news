#!/usr/bin/env python3
"""
F18 API Gateway Service - Lightweight microservices gateway
Provides basic routing and health checks for microservices mode
"""

import os
import sys
import json
import logging
from typing import Dict, Any
from datetime import datetime

# Add shared to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from flask import Flask, jsonify, request
    import requests
except ImportError:
    # Minimal fallback without Flask
    Flask = None
    requests = None

logger = logging.getLogger(__name__)

class APIGateway:
    """Lightweight API Gateway for F18 microservices mode"""
    
    def __init__(self):
        self.news_service_url = os.getenv('NEWS_SERVICE_URL', 'http://news_scraping_service:8001')
        
    def create_app(self):
        """Create Flask application for API Gateway"""
        if Flask is None:
            raise ImportError("Flask is required for microservices mode")
            
        app = Flask(__name__)
        
        @app.route('/healthz')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'service': 'api_gateway',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'version': '1.0.0'
            })
        
        @app.route('/metrics')
        def metrics():
            """Dummy metrics endpoint"""
            return jsonify({
                'requests_total': 42,
                'request_duration_seconds': 0.123,
                'service_up': 1,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        @app.route('/news')
        def get_news():
            """Proxy news requests to news scraping service"""
            try:
                if requests is None:
                    raise ImportError("requests library required")
                    
                # Forward request to news service
                response = requests.get(
                    f"{self.news_service_url}/news", 
                    params=request.args,
                    timeout=10
                )
                
                if response.status_code == 200:
                    return jsonify(response.json())
                else:
                    return jsonify({
                        'error': 'News service unavailable',
                        'status_code': response.status_code
                    }), response.status_code
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to news service: {e}")
                return jsonify({
                    'error': 'News service connection failed',
                    'message': str(e)
                }), 503
            except Exception as e:
                logger.error(f"Unexpected error in news proxy: {e}")
                return jsonify({
                    'error': 'Internal server error',
                    'message': str(e)
                }), 500
        
        @app.route('/')
        def root():
            """Root endpoint with API information"""
            return jsonify({
                'service': 'Stonk News API Gateway',
                'mode': 'microservices',
                'version': '1.0.0',
                'endpoints': {
                    'health': '/healthz',
                    'metrics': '/metrics',
                    'news': '/news'
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        return app

def create_app():
    """Factory function for creating the app"""
    gateway = APIGateway()
    return gateway.create_app()

# For running directly
if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)