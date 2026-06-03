#!/usr/bin/env python3
"""
Tests for F08 REST API hardening and authentication
"""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from flask import Flask

# Setup test environment
@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ['ENABLE_API_ENDPOINTS'] = 'true'
    os.environ['API_KEY'] = 'test-api-key-12345'
    os.environ['ALPHA_VANTAGE_KEY'] = 'test-av-key'
    yield
    # Cleanup
    for key in ['ENABLE_API_ENDPOINTS', 'API_KEY', 'ALPHA_VANTAGE_KEY']:
        os.environ.pop(key, None)

@pytest.fixture
def app():
    """Create test Flask app"""
    from api.app import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

class TestAPIAuthentication:
    """Test API key authentication middleware"""
    
    def test_auth_middleware_validates_key(self):
        """Test that auth middleware validates API keys correctly"""
        from api.auth import validate_api_key
        
        # Valid key
        assert validate_api_key('test-api-key-12345') is True
        
        # Invalid key
        assert validate_api_key('wrong-key') is False
        
        # None key
        assert validate_api_key(None) is False
    
    def test_require_api_key_decorator_with_valid_key(self, client):
        """Test that require_api_key decorator allows valid keys"""
        # Health endpoint should not require auth
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        
        # Feature flags endpoint should not require auth
        response = client.get('/api/v1/flags')
        assert response.status_code == 200
    
    def test_require_api_key_decorator_with_invalid_key(self, client):
        """Test that require_api_key decorator rejects invalid keys"""
        # Mock a protected endpoint that would normally exist
        with patch('config.feature_flags.is_recos_enabled', return_value=True):
            # No API key
            response = client.get('/api/v1/recs?scope=watchlist')
            assert response.status_code == 401
            assert 'API key required' in response.get_json()['error']
            
            # Invalid API key
            response = client.get('/api/v1/recs?scope=watchlist', 
                                headers={'X-API-Key': 'invalid-key'})
            assert response.status_code == 403
            assert 'Invalid API key' in response.get_json()['error']
            
            # Valid API key should work (mocking the service)
            with patch('services.recommendations_service.RecommendationsService'):
                response = client.get('/api/v1/recs?scope=watchlist', 
                                    headers={'X-API-Key': 'test-api-key-12345'})
                # Should not be 401/403 (might be 500 due to mocked service, but not auth error)
                assert response.status_code not in [401, 403]
    
    def test_api_key_missing_configuration(self):
        """Test behavior when API_KEY is not configured"""
        # Temporarily remove API key
        original_key = os.environ.get('API_KEY')
        if 'API_KEY' in os.environ:
            del os.environ['API_KEY']
        
        try:
            from api.auth import validate_api_key
            # Should return False when no key configured
            assert validate_api_key('any-key') is False
        finally:
            # Restore original key
            if original_key:
                os.environ['API_KEY'] = original_key

class TestPublicEndpoints:
    """Test that public endpoints remain accessible"""
    
    def test_health_endpoint_public(self, client):
        """Test health endpoint doesn't require authentication"""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'service' in data
    
    def test_flags_endpoint_public(self, client):
        """Test feature flags endpoint doesn't require authentication"""
        response = client.get('/api/v1/flags')
        assert response.status_code == 200
        data = response.get_json()
        assert 'flags' in data
        assert 'timestamp' in data
    
    def test_config_endpoint_depends_on_debug_flag(self, client):
        """Test config endpoint access based on debug settings"""
        # Without debug flag, should be forbidden
        response = client.get('/api/v1/config')
        assert response.status_code == 403
        
        # With debug flag enabled
        with patch.dict(os.environ, {'ENABLE_DEBUG_MODE': 'true'}):
            # Need to reload feature flags to pick up change
            from config.feature_flags import feature_flags
            feature_flags.set_flag('enable_debug_mode', True)
            
            response = client.get('/api/v1/config')
            # Should be allowed (might be 200 or 500 depending on config state)
            assert response.status_code != 403

class TestProtectedEndpoints:
    """Test that protected endpoints require authentication"""
    
    @pytest.mark.parametrize("endpoint,method,data", [
        ('/api/v1/recs?scope=watchlist', 'GET', None),
        ('/api/v1/symbols/intake', 'POST', {'ticker': 'AAPL', 'company_name': 'Apple Inc'}),
        ('/api/v1/symbols/jobs/test-id', 'GET', None),
        ('/api/v1/symbols/AAPL/intake_status', 'GET', None),
        ('/api/v1/symbols/AAPL/snapshot', 'GET', None),
        ('/api/v1/earnings/upcoming', 'GET', None),
        ('/api/v1/earnings/AAPL/explain', 'GET', None),
        ('/api/v1/admin/logs', 'GET', None),
        ('/api/v1/admin/logs/summary', 'GET', None),
        ('/api/v1/admin/logs/operation/test-id', 'GET', None),
        ('/api/v1/admin/logs/export', 'GET', None)
    ])
    def test_protected_endpoints_require_auth(self, client, endpoint, method, data):
        """Test all protected endpoints require authentication"""
        # Enable all features for testing
        with patch.dict(os.environ, {
            'ENABLE_RECOS': 'true',
            'ENABLE_SYMBOL_INTAKE': 'true', 
            'ENABLE_EARNINGS_READS': 'true',
            'ENABLE_API_ENDPOINTS': 'true'
        }):
            # Mock services to avoid actual API calls
            with patch('services.recommendations_service.RecommendationsService'), \
                 patch('services.symbol_intake.SymbolIntakeService'), \
                 patch('services.job_queue.JobQueue'), \
                 patch('services.earnings_service.EarningsAnalysisService'), \
                 patch('services.symbol_snapshot.SymbolSnapshotService'), \
                 patch('services.audit_logger.audit_logger'):
                
                # Test without API key
                if method == 'GET':
                    response = client.get(endpoint)
                else:
                    response = client.post(endpoint, json=data)
                
                assert response.status_code == 401, f"Endpoint {endpoint} should require auth"
                
                # Test with valid API key (should not be auth error)
                headers = {'X-API-Key': 'test-api-key-12345'}
                if method == 'GET':
                    response = client.get(endpoint, headers=headers)
                else:
                    response = client.post(endpoint, json=data, headers=headers)
                
                assert response.status_code not in [401, 403], f"Endpoint {endpoint} should accept valid key"

class TestOpenAPIDocumentation:
    """Test OpenAPI documentation endpoints"""
    
    def test_openapi_spec_available(self, client):
        """Test OpenAPI specification is available"""
        response = client.get('/api/v1/openapi.json')
        assert response.status_code == 200
        
        spec = response.get_json()
        assert spec['openapi'] == '3.0.0'
        assert spec['info']['title'] == 'Stonk News API'
        assert 'paths' in spec
        assert 'components' in spec
        assert 'securitySchemes' in spec['components']
    
    def test_swagger_ui_available(self, client):
        """Test Swagger UI is available"""
        response = client.get('/api/v1/docs')
        assert response.status_code == 200
        assert b'swagger-ui' in response.data
        assert b'Stonk News API Documentation' in response.data
    
    def test_openapi_disabled_when_api_disabled(self):
        """Test OpenAPI endpoints disabled when API endpoints disabled"""
        with patch.dict(os.environ, {'ENABLE_API_ENDPOINTS': 'false'}):
            from api.app import create_app
            app = create_app()
            client = app.test_client()
            
            response = client.get('/api/v1/openapi.json')
            assert response.status_code == 503
            
            response = client.get('/api/v1/docs')
            assert response.status_code == 503

class TestFeatureFlagIntegration:
    """Test feature flag integration with API endpoints"""
    
    def test_api_endpoints_disabled_by_feature_flag(self):
        """Test that disabling API endpoints feature flag prevents access"""
        with patch.dict(os.environ, {'ENABLE_API_ENDPOINTS': 'false'}):
            from api.app import create_app
            app = create_app()
            
            # Should have minimal blueprints when API disabled
            blueprint_names = [bp.name for bp in app.blueprints.values()]
            # Only health and basic endpoints should be registered
            assert len(blueprint_names) == 0  # No blueprints when API disabled
    
    def test_individual_feature_flags_control_endpoints(self, client):
        """Test individual feature flags control their respective endpoints"""
        # Test with recommendations disabled
        with patch('config.feature_flags.is_recos_enabled', return_value=False):
            headers = {'X-API-Key': 'test-api-key-12345'}
            response = client.get('/api/v1/recs?scope=watchlist', headers=headers)
            assert response.status_code == 503
            assert 'disabled' in response.get_json()['error'].lower()
        
        # Test with symbol intake disabled
        with patch('config.feature_flags.is_symbol_intake_enabled', return_value=False):
            headers = {'X-API-Key': 'test-api-key-12345'}
            data = {'ticker': 'AAPL', 'company_name': 'Apple Inc'}
            response = client.post('/api/v1/symbols/intake', json=data, headers=headers)
            assert response.status_code == 503
            assert 'disabled' in response.get_json()['error'].lower()

class TestErrorHandling:
    """Test API error handling"""
    
    def test_404_error_handler(self, client):
        """Test 404 error handling"""
        response = client.get('/api/v1/nonexistent')
        assert response.status_code == 404
        assert 'Endpoint not found' in response.get_json()['error']
    
    def test_authentication_error_logging(self, client):
        """Test that authentication failures are logged appropriately"""
        with patch('api.auth.logger') as mock_logger:
            # Test invalid API key
            headers = {'X-API-Key': 'invalid-key'}
            client.get('/api/v1/recs?scope=watchlist', headers=headers)
            
            # Should log warning about invalid key
            mock_logger.warning.assert_called()
            assert 'Invalid API key attempt' in str(mock_logger.warning.call_args)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])