#!/usr/bin/env python3
"""
Integration tests for F08 REST API hardening
Tests full API endpoint flows with authentication
"""
import pytest
import os
import json
from unittest.mock import patch, MagicMock
from flask import Flask

@pytest.fixture(autouse=True)
def setup_integration_env():
    """Setup integration test environment"""
    os.environ.update({
        'ENABLE_API_ENDPOINTS': 'true',
        'ENABLE_RECOS': 'true',
        'ENABLE_SYMBOL_INTAKE': 'true',
        'ENABLE_EARNINGS_READS': 'true',
        'API_KEY': 'integration-test-key-xyz',
        'ALPHA_VANTAGE_KEY': 'test-av-key',
        'API_HOST': 'localhost',
        'API_PORT': '8000',
        'API_DEBUG': 'true'
    })
    yield
    # Cleanup
    for key in ['ENABLE_API_ENDPOINTS', 'ENABLE_RECOS', 'ENABLE_SYMBOL_INTAKE', 
               'ENABLE_EARNINGS_READS', 'API_KEY', 'ALPHA_VANTAGE_KEY',
               'API_HOST', 'API_PORT', 'API_DEBUG']:
        os.environ.pop(key, None)

@pytest.fixture
def app():
    """Create Flask app for integration testing"""
    from api.app import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def authenticated_client(app):
    """Create authenticated test client"""
    client = app.test_client()
    # Add default headers for authentication
    client.environ_base['HTTP_X_API_KEY'] = 'integration-test-key-xyz'
    return client

class TestAPIWorkflow:
    """Test complete API workflows"""
    
    def test_symbol_intake_workflow(self, authenticated_client):
        """Test complete symbol intake workflow"""
        # Mock services
        mock_intake_service = MagicMock()
        mock_job_queue = MagicMock()
        mock_intake_service.get_symbol_by_ticker.return_value = None
        mock_job_queue.get_queue_size.return_value = 5
        
        with patch('services.symbol_intake.SymbolIntakeService', return_value=mock_intake_service), \
             patch('services.job_queue.JobQueue', return_value=mock_job_queue):
            
            # Step 1: Submit symbol for intake
            intake_data = {
                'ticker': 'TSLA',
                'company_name': 'Tesla Inc',
                'priority': 'high',
                'force_refresh': False
            }
            
            response = authenticated_client.post('/api/v1/symbols/intake', json=intake_data)
            assert response.status_code == 202
            
            response_data = response.get_json()
            assert 'job_id' in response_data
            assert 'symbol_id' in response_data
            assert response_data['status'] == 'queued'
            
            job_id = response_data['job_id']
            
            # Step 2: Check job status
            mock_job_queue.get_job_status.return_value = {
                'status': 'processing',
                'created_at': '2025-08-30T10:00:00Z',
                'updated_at': '2025-08-30T10:05:00Z'
            }
            
            response = authenticated_client.get(f'/api/v1/symbols/jobs/{job_id}')
            assert response.status_code == 200
            
            job_status = response.get_json()
            assert job_status['status'] == 'processing'
            
            # Step 3: Check symbol-specific intake status
            response = authenticated_client.get(f'/api/v1/symbols/TSLA/intake_status?job_id={job_id}')
            assert response.status_code == 200
            
            status_data = response.get_json()
            assert status_data['symbol'] == 'TSLA'
            assert status_data['job_id'] == job_id
    
    def test_recommendations_workflow(self, authenticated_client):
        """Test recommendations API workflow"""
        # Mock recommendations service
        mock_recs_service = MagicMock()
        mock_recommendations = [
            {
                'symbol': 'AAPL',
                'action': 'buy',
                'confidence': 0.85,
                'reasoning': 'Strong fundamentals'
            },
            {
                'symbol': 'GOOGL', 
                'action': 'hold',
                'confidence': 0.72,
                'reasoning': 'Market volatility'
            }
        ]
        mock_recs_service.generate_watchlist_recommendations.return_value = mock_recommendations
        
        with patch('services.recommendations_service.RecommendationsService', return_value=mock_recs_service):
            # Test watchlist recommendations
            response = authenticated_client.get('/api/v1/recs?scope=watchlist&include_details=true')
            assert response.status_code == 200
            
            data = response.get_json()
            assert data['scope'] == 'watchlist'
            assert len(data['recommendations']) == 2
            assert data['summary']['total'] == 2
            assert 'generated_at' in data
            assert 'disclaimer' in data['meta']
            
            # Test portfolio recommendations
            mock_recs_service.generate_portfolio_recommendations.return_value = mock_recommendations
            response = authenticated_client.get('/api/v1/recs?scope=portfolio&max_age_hours=48')
            assert response.status_code == 200
            
            data = response.get_json()
            assert data['scope'] == 'portfolio'
    
    def test_earnings_workflow(self, authenticated_client):
        """Test earnings API workflow"""
        # Mock earnings service
        mock_earnings_service = MagicMock()
        mock_upcoming = [
            {
                'symbol': 'AAPL',
                'date': '2025-09-15',
                'confirmed': True,
                'analysis': {
                    'risk_level': 'medium',
                    'implied_move_pct': 5.2
                }
            }
        ]
        mock_earnings_service.get_upcoming_earnings.return_value = mock_upcoming
        mock_earnings_service._get_current_timestamp.return_value = '2025-08-30T10:00:00Z'
        
        with patch('services.earnings_service.EarningsAnalysisService', return_value=mock_earnings_service):
            # Test upcoming earnings
            response = authenticated_client.get('/api/v1/earnings/upcoming?days=30')
            assert response.status_code == 200
            
            data = response.get_json()
            assert data['query']['days_ahead'] == 30
            assert len(data['upcoming_earnings']) == 1
            assert data['summary']['total_earnings'] == 1
            
            # Test earnings explanation
            mock_explanation = {
                'symbol': 'AAPL',
                'analysis_type': 'earnings_proximity',
                'explanation': 'Detailed analysis...'
            }
            mock_earnings_service.explain_earnings_analysis.return_value = mock_explanation
            
            response = authenticated_client.get('/api/v1/earnings/AAPL/explain')
            assert response.status_code == 200
            
            data = response.get_json()
            assert data['symbol'] == 'AAPL'
    
    def test_admin_logs_workflow(self, authenticated_client):
        """Test admin logs API workflow"""
        # Mock audit logger
        mock_logs = [
            {
                'log_id': 'op-123',
                'timestamp': '2025-08-30T10:00:00Z',
                'feature': 'recommendations',
                'operation': 'generate_recs',
                'status': 'completed',
                'symbol': 'AAPL',
                'step': 'completed'
            },
            {
                'log_id': 'op-124',
                'timestamp': '2025-08-30T10:05:00Z', 
                'feature': 'symbol_intake',
                'operation': 'intake_symbol',
                'status': 'processing',
                'symbol': 'TSLA',
                'step': 'processing'
            }
        ]
        
        mock_summary = {
            'total_operations': 2,
            'by_status': {'completed': 1, 'processing': 1},
            'by_feature': {'recommendations': 1, 'symbol_intake': 1}
        }
        
        with patch('services.audit_logger.audit_logger') as mock_audit:
            mock_audit.get_recent_logs.return_value = mock_logs
            mock_audit.get_operation_summary.return_value = mock_summary
            
            # Test recent logs
            response = authenticated_client.get('/api/v1/admin/logs?hours=24&feature=all')
            assert response.status_code == 200
            
            data = response.get_json()
            assert data['query']['hours'] == 24
            assert len(data['logs']) == 2
            assert data['total_entries'] == 2
            
            # Test logs summary
            response = authenticated_client.get('/api/v1/admin/logs/summary?hours=12')
            assert response.status_code == 200
            
            data = response.get_json()
            assert data['query']['hours'] == 12
            assert 'summary' in data
            
            # Test specific operation logs
            response = authenticated_client.get('/api/v1/admin/logs/operation/op-123')
            assert response.status_code == 200
            
            data = response.get_json()
            assert data['operation_id'] == 'op-123'

class TestAuthenticationFlows:
    """Test authentication edge cases and flows"""
    
    def test_missing_api_key_flow(self, app):
        """Test complete flow when API key is missing"""
        client = app.test_client()
        
        # All protected endpoints should return 401
        protected_endpoints = [
            '/api/v1/recs?scope=watchlist',
            '/api/v1/symbols/intake',
            '/api/v1/earnings/upcoming',
            '/api/v1/admin/logs'
        ]
        
        for endpoint in protected_endpoints:
            if 'intake' in endpoint:
                response = client.post(endpoint, json={'ticker': 'AAPL', 'company_name': 'Apple'})
            else:
                response = client.get(endpoint)
            
            assert response.status_code == 401
            assert 'API key required' in response.get_json()['error']
    
    def test_invalid_api_key_flow(self, app):
        """Test flow with invalid API key"""
        client = app.test_client()
        headers = {'X-API-Key': 'invalid-key-123'}
        
        # Mock services to avoid service errors
        with patch('services.recommendations_service.RecommendationsService'), \
             patch('config.feature_flags.is_recos_enabled', return_value=True):
            
            response = client.get('/api/v1/recs?scope=watchlist', headers=headers)
            assert response.status_code == 403
            assert 'Invalid API key' in response.get_json()['error']
    
    def test_api_key_rotation_scenario(self, app):
        """Test scenario where API key changes (rotation)"""
        # Test with old key first
        client = app.test_client()
        old_headers = {'X-API-Key': 'old-api-key'}
        
        # Should fail with old key
        response = client.get('/api/v1/health')  # Health should work without key
        assert response.status_code == 200
        
        # Protected endpoint should fail with any invalid key
        with patch('services.recommendations_service.RecommendationsService'), \
             patch('config.feature_flags.is_recos_enabled', return_value=True):
            response = client.get('/api/v1/recs?scope=watchlist', headers=old_headers)
            assert response.status_code == 403

class TestOpenAPIIntegration:
    """Test OpenAPI documentation integration"""
    
    def test_openapi_spec_completeness(self, authenticated_client):
        """Test that OpenAPI spec covers all endpoints"""
        response = authenticated_client.get('/api/v1/openapi.json')
        assert response.status_code == 200
        
        spec = response.get_json()
        paths = spec['paths']
        
        # Verify key endpoints are documented
        expected_paths = [
            '/health',
            '/flags',
            '/config',
            '/recs',
            '/symbols/intake',
            '/symbols/jobs/{job_id}',
            '/symbols/{symbol}/intake_status',
            '/symbols/{symbol}/snapshot',
            '/earnings/upcoming',
            '/earnings/{symbol}/explain',
            '/admin/logs'
        ]
        
        for path in expected_paths:
            assert path in paths, f"Path {path} missing from OpenAPI spec"
        
        # Verify security schemes
        assert 'ApiKeyAuth' in spec['components']['securitySchemes']
        security_scheme = spec['components']['securitySchemes']['ApiKeyAuth']
        assert security_scheme['type'] == 'apiKey'
        assert security_scheme['in'] == 'header'
        assert security_scheme['name'] == 'X-API-Key'
    
    def test_swagger_ui_functionality(self, authenticated_client):
        """Test Swagger UI serves correctly"""
        response = authenticated_client.get('/api/v1/docs')
        assert response.status_code == 200
        
        content = response.data.decode()
        assert 'SwaggerUIBundle' in content
        assert '/api/v1/openapi.json' in content
        assert 'Stonk News API Documentation' in content

class TestFeatureFlagDependencies:
    """Test feature flag dependencies and interactions"""
    
    def test_cascading_feature_flag_effects(self, app):
        """Test how feature flags interact with each other"""
        client = app.test_client()
        headers = {'X-API-Key': 'integration-test-key-xyz'}
        
        # When API endpoints are disabled, nothing should work
        with patch.dict(os.environ, {'ENABLE_API_ENDPOINTS': 'false'}):
            # Create new app instance with disabled flags
            from api.app import create_app
            disabled_app = create_app()
            disabled_client = disabled_app.test_client()
            
            # Even OpenAPI should be disabled
            response = disabled_client.get('/api/v1/openapi.json')
            assert response.status_code in [404, 503]  # Endpoint not found or disabled
    
    def test_granular_feature_control(self, authenticated_client):
        """Test granular feature flag control"""
        # Test each feature can be independently disabled
        features_and_endpoints = [
            ('ENABLE_RECOS', '/api/v1/recs?scope=watchlist'),
            ('ENABLE_SYMBOL_INTAKE', '/api/v1/symbols/intake'),
            ('ENABLE_EARNINGS_READS', '/api/v1/earnings/upcoming')
        ]
        
        for flag, endpoint in features_and_endpoints:
            with patch.dict(os.environ, {flag: 'false'}):
                # Need to reload config
                from config.feature_flags import FeatureFlags
                flags = FeatureFlags()
                
                if 'intake' in endpoint:
                    response = authenticated_client.post(endpoint, 
                                                      json={'ticker': 'AAPL', 'company_name': 'Apple'})
                else:
                    response = authenticated_client.get(endpoint)
                
                # Should be service disabled, not auth error
                assert response.status_code == 503
                assert 'disabled' in response.get_json()['error'].lower()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])