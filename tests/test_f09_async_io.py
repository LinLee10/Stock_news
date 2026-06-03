#!/usr/bin/env python3
"""
Tests for F09 Async I/O + caching + retries functionality
"""
import pytest
import asyncio
import aiohttp
import os
import time
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# Setup test environment
@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ.update({
        'ENABLE_ASYNC_IO': 'true',
        'MAX_CONCURRENCY': '3',
        'BACKOFF_MAX': '10',
        'CIRCUIT_FAILURE_THRESHOLD': '2',
        'CIRCUIT_RECOVERY_TIMEOUT': '5',
        'CACHE_TTL': '300',
        'REQUEST_TIMEOUT': '10'
    })
    yield
    # Cleanup
    for key in ['ENABLE_ASYNC_IO', 'MAX_CONCURRENCY', 'BACKOFF_MAX', 
               'CIRCUIT_FAILURE_THRESHOLD', 'CIRCUIT_RECOVERY_TIMEOUT', 
               'CACHE_TTL', 'REQUEST_TIMEOUT']:
        os.environ.pop(key, None)

@pytest.fixture
def async_config():
    """Provide async configuration for tests"""
    from services.retry_policies import load_async_config
    return load_async_config()

@pytest.fixture
async def reset_async_state():
    """Reset global async state before each test"""
    from services.retry_policies import reset_async_state
    await reset_async_state()
    yield
    await reset_async_state()

class TestAsyncRetryPolicies:
    """Test F09 async retry policies and circuit breakers"""
    
    @pytest.mark.asyncio
    async def test_retry_async_success_first_attempt(self, reset_async_state):
        """Test successful async operation on first attempt"""
        from services.retry_policies import retry_async, AsyncRetryConfig
        
        async def mock_success():
            return "success"
        
        config = AsyncRetryConfig(max_retries=3, base_delay=0.1)
        result = await retry_async(mock_success, retry_config=config)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_async_exponential_backoff(self, reset_async_state):
        """Test exponential backoff with jitter"""
        from services.retry_policies import retry_async, AsyncRetryConfig
        
        attempt_count = 0
        async def mock_failing_then_success():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise aiohttp.ClientError("Temporary failure")
            return "success"
        
        config = AsyncRetryConfig(
            max_retries=3, 
            base_delay=0.1, 
            exponential_base=2.0,
            backoff_strategy="exponential",
            jitter=True
        )
        
        start_time = time.time()
        result = await retry_async(mock_failing_then_success, retry_config=config)
        end_time = time.time()
        
        assert result == "success"
        assert attempt_count == 3
        # Should have some delay from backoff
        assert end_time - start_time > 0.1
    
    @pytest.mark.asyncio
    async def test_retry_async_all_attempts_fail(self, reset_async_state):
        """Test failure after all retry attempts exhausted"""
        from services.retry_policies import retry_async, AsyncRetryConfig
        
        async def mock_always_fail():
            raise ValueError("Persistent failure")
        
        config = AsyncRetryConfig(max_retries=2, base_delay=0.01)
        
        with pytest.raises(ValueError, match="Persistent failure"):
            await retry_async(mock_always_fail, retry_config=config)
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker_open_close_cycle(self, reset_async_state):
        """Test circuit breaker state transitions"""
        from services.retry_policies import AsyncCircuitBreaker, AsyncCircuitBreakerConfig, AsyncCircuitBreakerOpenError
        
        config = AsyncCircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=2
        )
        breaker = AsyncCircuitBreaker(config, name="test")
        
        # Should start CLOSED
        assert breaker.state == "CLOSED"
        
        # Fail enough times to open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await breaker.call(self._mock_failing_async)
        
        # Circuit should now be OPEN
        assert breaker.state == "OPEN"
        assert breaker.failure_count == 2
        
        # Next call should fail fast
        with pytest.raises(AsyncCircuitBreakerOpenError):
            await breaker.call(self._mock_success_async)
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Should transition to HALF_OPEN and then CLOSED after successes
        result1 = await breaker.call(self._mock_success_async)
        assert result1 == "success"
        assert breaker.state == "HALF_OPEN"
        
        result2 = await breaker.call(self._mock_success_async)
        assert result2 == "success"
        assert breaker.state == "CLOSED"
    
    async def _mock_failing_async(self):
        """Helper mock function that always fails"""
        raise ValueError("Mock failure")
    
    async def _mock_success_async(self):
        """Helper mock function that always succeeds"""
        return "success"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_stats(self, reset_async_state):
        """Test circuit breaker statistics"""
        from services.retry_policies import AsyncCircuitBreaker, AsyncCircuitBreakerConfig
        
        config = AsyncCircuitBreakerConfig(failure_threshold=3)
        breaker = AsyncCircuitBreaker(config, name="stats_test")
        
        # Record a few failures
        for i in range(2):
            with pytest.raises(ValueError):
                await breaker.call(self._mock_failing_async)
        
        stats = breaker.get_stats()
        assert stats['name'] == "stats_test"
        assert stats['state'] == "CLOSED"  # Not enough failures to open
        assert stats['failure_count'] == 2
        assert stats['last_failure_time'] is not None

class TestAsyncCache:
    """Test F09 async local cache functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_hit_miss(self, reset_async_state):
        """Test cache hit and miss scenarios"""
        from services.retry_policies import AsyncLocalCache, CacheConfig
        
        config = CacheConfig(enabled=True, ttl_seconds=1, max_entries=10)
        cache = AsyncLocalCache(config)
        
        url = "http://example.com/api"
        params = {"param1": "value1"}
        test_data = {"result": "cached_data"}
        
        # Should be cache miss initially
        result = await cache.get(url, params)
        assert result is None
        
        # Store data in cache
        await cache.set(url, test_data, params)
        
        # Should now be cache hit
        result = await cache.get(url, params)
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, reset_async_state):
        """Test cache TTL expiration"""
        from services.retry_policies import AsyncLocalCache, CacheConfig
        
        config = CacheConfig(enabled=True, ttl_seconds=0.1, max_entries=10)
        cache = AsyncLocalCache(config)
        
        url = "http://example.com/api"
        test_data = {"result": "will_expire"}
        
        await cache.set(url, test_data)
        
        # Should hit immediately
        result = await cache.get(url)
        assert result == test_data
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should miss after expiration
        result = await cache.get(url)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, reset_async_state):
        """Test LRU cache eviction when max_entries reached"""
        from services.retry_policies import AsyncLocalCache, CacheConfig
        
        config = CacheConfig(enabled=True, ttl_seconds=60, max_entries=2)
        cache = AsyncLocalCache(config)
        
        # Fill cache to capacity
        await cache.set("url1", "data1")
        await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
        await cache.set("url2", "data2")
        
        # Add one more - should evict oldest (url1)
        await cache.set("url3", "data3")
        
        # url1 should be evicted, url2 and url3 should remain
        assert await cache.get("url1") is None
        assert await cache.get("url2") == "data2"
        assert await cache.get("url3") == "data3"
    
    @pytest.mark.asyncio
    async def test_cache_key_generation_security(self, reset_async_state):
        """Test cache key generation excludes sensitive headers"""
        from services.retry_policies import AsyncLocalCache, CacheConfig
        
        config = CacheConfig(enabled=True, ttl_seconds=60)
        cache = AsyncLocalCache(config)
        
        url = "http://api.example.com"
        params = {"query": "test"}
        headers_with_auth = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
        headers_without_auth = {"Content-Type": "application/json"}
        
        # Both should generate the same key (auth header excluded)
        key1 = cache._generate_key(url, params, headers_with_auth)
        key2 = cache._generate_key(url, params, headers_without_auth)
        
        assert key1 == key2

class TestAsyncHTTPClient:
    """Test F09 async HTTP client with resilience features"""
    
    @pytest.mark.asyncio
    async def test_http_client_get_success(self, reset_async_state):
        """Test successful GET request"""
        from services.retry_policies import AsyncHTTPClient, AsyncRetryConfig
        
        # Mock aiohttp session
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.content_type = "application/json"
        mock_response.headers = {}
        mock_response.url = "http://example.com"
        
        mock_session = MagicMock()
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)
        
        config = AsyncRetryConfig(max_retries=1)
        
        async with AsyncHTTPClient(retry_config=config, semaphore_limit=1) as client:
            client.session = mock_session
            
            response = await client.get("http://example.com/api")
            
            assert response['status'] == 200
            assert response['data'] == {"success": True}
            assert response['from_cache'] is False
    
    @pytest.mark.asyncio
    async def test_http_client_cache_behavior(self, reset_async_state):
        """Test HTTP client caching behavior"""
        from services.retry_policies import AsyncHTTPClient, CacheConfig
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"cached": True})
        mock_response.content_type = "application/json"
        mock_response.headers = {}
        mock_response.url = "http://example.com"
        
        mock_session = MagicMock()
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)
        
        cache_config = CacheConfig(enabled=True, ttl_seconds=60)
        
        async with AsyncHTTPClient(cache_config=cache_config, semaphore_limit=1) as client:
            client.session = mock_session
            
            # First request should hit network and cache
            response1 = await client.get("http://example.com/api")
            assert response1['from_cache'] is False
            
            # Second identical request should hit cache
            response2 = await client.get("http://example.com/api")
            assert response2['from_cache'] is True
            assert response2['data'] == {"cached": True}
    
    @pytest.mark.asyncio
    async def test_http_client_server_error_retry(self, reset_async_state):
        """Test HTTP client retry on server errors"""
        from services.retry_policies import AsyncHTTPClient, AsyncRetryConfig
        
        attempt_count = 0
        
        async def mock_request_with_retry(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            
            mock_response = MagicMock()
            if attempt_count < 3:
                # First two attempts fail with 500
                mock_response.status = 500
                mock_response.request_info = None
                mock_response.history = None
                raise aiohttp.ClientResponseError(
                    None, None, status=500, message="Server error"
                )
            else:
                # Third attempt succeeds
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={"retry_success": True})
                mock_response.content_type = "application/json"
                mock_response.headers = {}
                mock_response.url = "http://example.com"
                return mock_response
        
        mock_session = MagicMock()
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = mock_request_with_retry
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)
        
        retry_config = AsyncRetryConfig(max_retries=3, base_delay=0.01)
        
        async with AsyncHTTPClient(retry_config=retry_config, semaphore_limit=1) as client:
            client.session = mock_session
            
            response = await client.get("http://example.com/api")
            
            assert response['status'] == 200
            assert response['data'] == {"retry_success": True}
            assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_http_client_performance_stats(self, reset_async_state):
        """Test HTTP client performance statistics"""
        from services.retry_policies import AsyncHTTPClient
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="success")
        mock_response.content_type = "text/plain"
        mock_response.headers = {}
        mock_response.url = "http://example.com"
        
        mock_session = MagicMock()
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)
        
        async with AsyncHTTPClient(semaphore_limit=1) as client:
            client.session = mock_session
            
            # Make a few requests
            await client.get("http://example.com/api1")
            await client.get("http://example.com/api2")
            
            stats = client.get_performance_stats()
            
            assert stats['client_stats']['requests'] == 2
            assert stats['client_stats']['average_latency'] > 0
            assert 'cache' in stats
            assert 'circuit_breakers' in stats

class TestMakeResilientRequests:
    """Test F09 asyncio.gather with semaphore limits"""
    
    @pytest.mark.asyncio
    async def test_make_resilient_requests_success(self, reset_async_state):
        """Test concurrent resilient requests succeed"""
        from services.retry_policies import make_resilient_requests
        
        # Mock responses for different URLs
        def mock_get_response(url, **kwargs):
            mock_response = AsyncMock()
            mock_response.return_value = {
                'status': 200,
                'data': f'data_for_{url}',
                'from_cache': False,
                'url': url
            }
            return mock_response()
        
        urls = [
            "http://api1.example.com",
            "http://api2.example.com", 
            "http://api3.example.com"
        ]
        
        with patch('services.retry_policies.AsyncHTTPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            results = await make_resilient_requests(urls, method='GET', max_concurrency=2)
            
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result['status'] == 200
                assert f'api{i+1}.example.com' in result['data']
    
    @pytest.mark.asyncio
    async def test_make_resilient_requests_partial_failure(self, reset_async_state):
        """Test resilient requests with some failures (AC1: failures logged but do not abort)"""
        from services.retry_policies import make_resilient_requests
        
        def mock_get_response(url, **kwargs):
            if 'fail' in url:
                raise aiohttp.ClientError("Simulated failure")
            
            mock_response = AsyncMock()
            mock_response.return_value = {
                'status': 200,
                'data': f'data_for_{url}',
                'from_cache': False,
                'url': url
            }
            return mock_response()
        
        urls = [
            "http://success.example.com",
            "http://fail.example.com",
            "http://success2.example.com"
        ]
        
        with patch('services.retry_policies.AsyncHTTPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            results = await make_resilient_requests(urls, method='GET', max_concurrency=2)
            
            assert len(results) == 3
            
            # First request should succeed
            assert results[0]['status'] == 200
            assert 'success.example.com' in results[0]['data']
            
            # Second request should fail gracefully
            assert results[1]['status'] == 0
            assert results[1]['failed'] is True
            assert 'error' in results[1]
            
            # Third request should succeed
            assert results[2]['status'] == 200
            assert 'success2.example.com' in results[2]['data']

class TestGlobalStatsAndConfig:
    """Test F09 global statistics and configuration"""
    
    @pytest.mark.asyncio
    async def test_load_async_config(self, async_config, reset_async_state):
        """Test loading async configuration from environment"""
        assert async_config['max_concurrency'] == 3
        assert async_config['backoff_max'] == 10
        assert async_config['circuit_failure_threshold'] == 2
        assert async_config['circuit_recovery_timeout'] == 5
        assert async_config['cache_ttl'] == 300
        assert async_config['request_timeout'] == 10
    
    @pytest.mark.asyncio
    async def test_get_async_stats(self, reset_async_state):
        """Test getting global async statistics"""
        from services.retry_policies import get_async_stats
        
        stats = get_async_stats()
        
        assert 'circuit_breakers' in stats
        assert 'cache' in stats
        assert 'timestamp' in stats
        assert stats['feature'] == 'F09_async_io'
        assert isinstance(stats['circuit_breakers'], dict)
        assert isinstance(stats['cache'], dict)
    
    @pytest.mark.asyncio
    async def test_reset_async_state(self, reset_async_state):
        """Test resetting global async state"""
        from services.retry_policies import (
            get_async_circuit_breaker, _async_cache, get_async_stats
        )
        
        # Create some state
        breaker = get_async_circuit_breaker("test.example.com")
        await _async_cache.set("test_url", "test_data")
        
        # Verify state exists
        initial_stats = get_async_stats()
        assert len(initial_stats['circuit_breakers']) > 0
        assert initial_stats['cache']['entries'] > 0
        
        # Reset should clear state
        # (already done by fixture, verify it's clean)
        final_stats = get_async_stats()
        assert len(final_stats['circuit_breakers']) == 0
        assert final_stats['cache']['entries'] == 0

class TestFeatureFlagIntegration:
    """Test F09 feature flag integration"""
    
    def test_async_io_disabled_fallback(self):
        """Test fallback to sync when async I/O is disabled"""
        with patch.dict(os.environ, {'ENABLE_ASYNC_IO': 'false'}):
            from config.feature_flags import is_async_io_enabled
            assert not is_async_io_enabled()
    
    def test_async_io_enabled(self):
        """Test async I/O feature flag enabled"""
        with patch.dict(os.environ, {'ENABLE_ASYNC_IO': 'true'}):
            from config.feature_flags import is_async_io_enabled
            assert is_async_io_enabled()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])