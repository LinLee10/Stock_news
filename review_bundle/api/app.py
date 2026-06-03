#!/usr/bin/env python3
"""
Flask application factory for Stonk News API
"""
from flask import Flask, jsonify
from config.feature_flags import feature_flags
from config.config import API_DEBUG
import logging

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config['DEBUG'] = API_DEBUG
    
    # BEGIN F15_API_DISCLAIMER
    # Add root endpoint with disclaimer and basic information
    @app.route('/')
    def root():
        """Root endpoint with disclaimer and API information"""
        import os
        from datetime import datetime
        
        disclaimer_text = os.getenv('DISCLAIMER_TEXT', 'This API provides research and educational data only. Not financial advice.')
        company_name = os.getenv('COMPANY_NAME', 'Stonk News')
        
        return jsonify({
            'service': 'Stonk News API',
            'version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'disclaimer': disclaimer_text,
            'company': company_name,
            'endpoints': {
                'health': '/api/v1/health',
                'docs': '/api/v1/docs',
                'openapi': '/api/v1/openapi.json'
            },
            'status': 'operational',
            'notice': 'All data provided is for research purposes. Consult a financial advisor for investment decisions.'
        })
    # END F15_API_DISCLAIMER
    
    # Register blueprints based on feature flags
    if feature_flags.is_enabled('enable_api_endpoints'):
        from .health import health_blueprint
        app.register_blueprint(health_blueprint, url_prefix='/api/v1')
        
        # Admin endpoints (always available when API is enabled)
        from .admin import admin_blueprint
        app.register_blueprint(admin_blueprint, url_prefix='/api/v1/admin')
        
        if feature_flags.is_enabled('enable_symbol_intake'):
            from .symbols import symbols_blueprint
            app.register_blueprint(symbols_blueprint, url_prefix='/api/v1')
        
        if feature_flags.is_enabled('enable_recos'):
            from .recommendations import recommendations_blueprint
            app.register_blueprint(recommendations_blueprint, url_prefix='/api/v1')
        
        if feature_flags.is_enabled('enable_earnings_reads'):
            from .earnings import earnings_blueprint
            app.register_blueprint(earnings_blueprint, url_prefix='/api/v1')
        
        # OpenAPI documentation (always available when API is enabled)
        from .openapi import openapi_blueprint
        app.register_blueprint(openapi_blueprint, url_prefix='/api/v1')
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logging.exception("Internal server error")
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    if not feature_flags.is_enabled('enable_api_endpoints'):
        print("API endpoints are disabled. Set ENABLE_API_ENDPOINTS=true")
        exit(1)
    
    from config.config import API_HOST, API_PORT
    app = create_app()
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)