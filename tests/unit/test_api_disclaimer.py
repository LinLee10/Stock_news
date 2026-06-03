#!/usr/bin/env python3
"""
Unit tests for API disclaimer functionality
Tests that disclaimer appears in API root endpoint response
"""

import os
import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check if API module exists
try:
    from api.app import create_app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


@pytest.mark.skipif(not API_AVAILABLE, reason="API module not present")
class TestAPIDisclaimer:
    """Test disclaimer inclusion in API responses"""
    
    def test_root_endpoint_contains_disclaimer(self):
        """Test that root endpoint returns disclaimer information"""
        with patch.dict(os.environ, {}, clear=True):  # Clear env vars for default values
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                
                assert response.status_code == 200
                assert response.content_type == 'application/json'
                
                data = response.get_json()
                
                # Check required fields
                assert 'disclaimer' in data
                assert 'service' in data
                assert 'version' in data
                assert 'timestamp' in data
                assert 'status' in data
                
                # Check default disclaimer text
                assert 'research and educational data only' in data['disclaimer']
                assert 'Not financial advice' in data['disclaimer']
                
                # Check service information
                assert data['service'] == 'Stonk News API'
                assert data['status'] == 'operational'
    
    def test_custom_disclaimer_in_api(self):
        """Test that custom disclaimer from environment appears in API"""
        custom_disclaimer = "Custom API disclaimer for testing purposes only."
        custom_company = "Test API Company"
        
        with patch.dict(os.environ, {
            'DISCLAIMER_TEXT': custom_disclaimer,
            'COMPANY_NAME': custom_company
        }):
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                data = response.get_json()
                
                assert response.status_code == 200
                assert data['disclaimer'] == custom_disclaimer
                assert data['company'] == custom_company
    
    def test_api_endpoints_information(self):
        """Test that root endpoint provides endpoint information"""
        with patch.dict(os.environ, {}, clear=True):
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                data = response.get_json()
                
                assert response.status_code == 200
                assert 'endpoints' in data
                
                endpoints = data['endpoints']
                assert 'health' in endpoints
                assert 'docs' in endpoints
                assert 'openapi' in endpoints
                
                # Check endpoint paths are correct
                assert endpoints['health'] == '/api/v1/health'
                assert endpoints['docs'] == '/api/v1/docs'
                assert endpoints['openapi'] == '/api/v1/openapi.json'
    
    def test_api_timestamp_format(self):
        """Test that API timestamp is in correct ISO format"""
        with patch.dict(os.environ, {}, clear=True):
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                data = response.get_json()
                
                assert response.status_code == 200
                assert 'timestamp' in data
                
                timestamp = data['timestamp']
                assert timestamp.endswith('Z')  # UTC format
                assert 'T' in timestamp  # ISO format separator
                
                # Should be parseable as ISO format
                from datetime import datetime
                parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                assert parsed_time is not None
    
    def test_api_notice_field(self):
        """Test that API includes compliance notice"""
        with patch.dict(os.environ, {}, clear=True):
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                data = response.get_json()
                
                assert response.status_code == 200
                assert 'notice' in data
                
                notice = data['notice']
                assert 'research purposes' in notice
                assert 'financial advisor' in notice
    
    def test_api_version_present(self):
        """Test that API version information is present"""
        with patch.dict(os.environ, {}, clear=True):
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                data = response.get_json()
                
                assert response.status_code == 200
                assert 'version' in data
                assert data['version'] == '1.0.0'
    
    def test_api_response_json_structure(self):
        """Test that API response has proper JSON structure"""
        with patch.dict(os.environ, {}, clear=True):
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                
                assert response.status_code == 200
                assert response.content_type == 'application/json'
                
                # Should be valid JSON
                data = response.get_json()
                assert isinstance(data, dict)
                
                # Check all expected top-level keys
                expected_keys = {
                    'service', 'version', 'timestamp', 'disclaimer', 
                    'company', 'endpoints', 'status', 'notice'
                }
                assert set(data.keys()) == expected_keys
    
    def test_api_disclaimer_safety(self):
        """Test that API disclaimer handles special characters safely"""
        special_disclaimer = "API tëst with spéciâl çharåcters & JSON-safe symbols"
        
        with patch.dict(os.environ, {
            'DISCLAIMER_TEXT': special_disclaimer
        }):
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                data = response.get_json()
                
                assert response.status_code == 200
                assert data['disclaimer'] == special_disclaimer
                
                # Should be valid JSON even with special characters
                json_str = json.dumps(data)
                parsed_back = json.loads(json_str)
                assert parsed_back['disclaimer'] == special_disclaimer
    
    def test_api_root_always_available(self):
        """Test that root endpoint is always available regardless of feature flags"""
        with patch.dict(os.environ, {
            'ENABLE_API_ENDPOINTS': 'false',
            'ENABLE_SYMBOL_INTAKE': 'false',
            'ENABLE_RECOS': 'false'
        }):
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                
                # Root should always work even when other endpoints are disabled
                assert response.status_code == 200
                data = response.get_json()
                assert 'disclaimer' in data
                assert 'status' in data
    
    def test_api_cors_headers(self):
        """Test that API response includes appropriate headers"""
        with patch.dict(os.environ, {}, clear=True):
            app = create_app()
            
            with app.test_client() as client:
                response = client.get('/')
                
                assert response.status_code == 200
                # Should have JSON content type
                assert 'application/json' in response.content_type


@pytest.mark.skipif(API_AVAILABLE, reason="Testing behavior when API is not available")
class TestAPINotAvailable:
    """Test behavior when API module is not present"""
    
    def test_api_not_present_skip(self):
        """Test that tests are skipped when API is not available"""
        # This test should only run when API_AVAILABLE is False
        assert not API_AVAILABLE
        # If we get here, the skip condition is working correctly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])