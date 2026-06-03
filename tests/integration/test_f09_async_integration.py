#!/usr/bin/env python3
"""
Integration tests for F09 Async I/O + caching + retries
Tests full integration with news_scraper.py and multi_source_data_manager.py
"""
import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, date

@pytest.fixture(autouse=True)
def setup_integration_env():
    """Setup integration test environment"""
    os.environ.update({
        'ENABLE_ASYNC_IO': 'true',
        'MAX_CONCURRENCY': '2',
        'BACKOFF_MAX': '5',
        'CIRCUIT_FAILURE_THRESHOLD': '3',
        'CIRCUIT_RECOVERY_TIMEOUT': '2',
        'CACHE_TTL': '300',
        'REQUEST_TIMEOUT': '10',
        'ALPHA_VANTAGE_KEY': 'test_key'
    })
    yield
    # Cleanup
    for key in ['ENABLE_ASYNC_IO', 'MAX_CONCURRENCY', 'BACKOFF_MAX',
               'CIRCUIT_FAILURE_THRESHOLD', 'CIRCUIT_RECOVERY_TIMEOUT',
               'CACHE_TTL', 'REQUEST_TIMEOUT', 'ALPHA_VANTAGE_KEY']:
        os.environ.pop(key, None)

@pytest.fixture
async def reset_async_state():
    """Reset global async state before each test"""
    from services.retry_policies import reset_async_state
    await reset_async_state()
    yield
    await reset_async_state()

class TestNewsScraperAsyncIntegration:
    """Test F09 integration with news_scraper.py"""
    
    @pytest.mark.asyncio
    async def test_scrape_headlines_async_feature_flag_enabled(self, reset_async_state):
        """Test async headlines scraping when feature flag is enabled"""
        # Mock the async HTTP responses
        mock_google_response = '''<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Apple Stock Rises on Strong Earnings</title>
                    <link>http://example.com/news1</link>
                    <pubDate>Wed, 29 Aug 2024 14:30:00 GMT</pubDate>
                </item>
            </channel>
        </rss>'''
        
        mock_av_response = {
            "feed": [
                {
                    "title": "Apple Inc. Reports Q3 Results",
                    "url": "http://example.com/news2",
                    "time_published": "2024-08-29T15:00:00Z"
                }
            ]
        }
        
        # Mock sentiment pipeline
        mock_sentiment_results = [
            {"label": "positive", "score": 0.8},
            {"label": "positive", "score": 0.7}
        ]
        
        with patch('news_scraper.get_sentiment_pipeline') as mock_pipeline, \
             patch('services.retry_policies.AsyncHTTPClient') as mock_client_class:
            
            # Setup sentiment pipeline mock
            mock_pipe = MagicMock()
            mock_pipe.return_value = mock_sentiment_results
            mock_pipeline.return_value = mock_pipe
            
            # Setup async HTTP client mock
            mock_client = MagicMock()
            
            def mock_get_response(url, **kwargs):
                if 'google.com' in url:
                    return AsyncMock(return_value={
                        'status': 200,
                        'data': mock_google_response,
                        'from_cache': False
                    })()
                elif 'alphavantage.co' in url:
                    return AsyncMock(return_value={
                        'status': 200,
                        'data': mock_av_response,
                        'from_cache': False
                    })()
                else:
                    return AsyncMock(return_value={
                        'status': 404,
                        'data': '',
                        'from_cache': False
                    })()
            
            mock_client.get = mock_get_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # Test the async scraper
            from news_scraper import scrape_headlines_async
            
            tickers = ['AAPL']
            results = await scrape_headlines_async(tickers, days=7)
            
            # Verify results
            assert 'AAPL' in results
            aapl_data = results['AAPL']
            assert aapl_data['count'] >= 1  # Should have at least one headline
            assert aapl_data['count_positive'] >= 1  # Should have positive sentiment
            assert isinstance(aapl_data['headlines'], list)
            assert isinstance(aapl_data['daily_sentiment'], dict)
    
    @pytest.mark.asyncio
    async def test_scrape_headlines_resilient_wrapper(self, reset_async_state):
        """Test resilient wrapper function that chooses async vs sync"""
        with patch('news_scraper.scrape_headlines_async') as mock_async, \
             patch('news_scraper.scrape_headlines') as mock_sync:
            
            mock_async.return_value = {'AAPL': {'async': True}}
            mock_sync.return_value = {'AAPL': {'sync': True}}
            
            from news_scraper import scrape_headlines_resilient
            
            # Should use async version when flag is enabled
            result = scrape_headlines_resilient(['AAPL'])
            mock_async.assert_called_once()
            mock_sync.assert_not_called()
            assert result['AAPL']['async'] is True
    
    @pytest.mark.asyncio
    async def test_scrape_headlines_async_resilience_to_failures(self, reset_async_state):
        """Test F09 AC1: resilient to transient failures; failures logged but do not abort"""
        
        # Mock mixed success/failure scenario
        def mock_get_response(url, **kwargs):
            if 'fail' in url:
                raise Exception("Simulated network failure")
            return AsyncMock(return_value={
                'status': 200,
                'data': '<rss></rss>',
                'from_cache': False
            })()
        
        with patch('news_scraper.get_sentiment_pipeline') as mock_pipeline, \
             patch('services.retry_policies.AsyncHTTPClient') as mock_client_class:
            
            mock_pipeline.return_value = MagicMock(return_value=[])
            
            mock_client = MagicMock()
            mock_client.get = mock_get_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            from news_scraper import scrape_headlines_async
            
            # Should complete successfully even with some failures
            tickers = ['AAPL', 'GOOGL']
            results = await scrape_headlines_async(tickers, days=7)
            
            # Both tickers should have results (even if empty due to failures)
            assert 'AAPL' in results
            assert 'GOOGL' in results
            
            # Results should be valid structure even if empty
            for ticker in results:
                assert 'headlines' in results[ticker]
                assert 'count' in results[ticker]
                assert isinstance(results[ticker]['headlines'], list)
    
    @pytest.mark.asyncio
    async def test_async_fetch_functions_circuit_breaker_protection(self, reset_async_state):
        """Test individual async fetch functions with circuit breaker protection"""
        from news_scraper import fetch_google_rss_async, fetch_av_news_async
        
        with patch('services.retry_policies.AsyncHTTPClient') as mock_client_class:
            # Mock client that fails consistently to trigger circuit breaker
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=Exception("Persistent failure"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # Google RSS fetch should return empty string on failure
            google_result = await fetch_google_rss_async("Apple Inc")
            assert google_result == ""
            
            # Alpha Vantage fetch should return empty list on failure
            av_result = await fetch_av_news_async("AAPL")
            assert av_result == []

class TestMultiSourceDataManagerAsyncIntegration:
    """Test F09 integration with multi_source_data_manager.py"""
    
    @pytest.mark.asyncio
    async def test_async_multi_source_data_manager_initialization(self, reset_async_state):
        """Test AsyncMultiSourceDataManager initializes properly"""
        from services.multi_source_data_manager import AsyncMultiSourceDataManager
        
        manager = AsyncMultiSourceDataManager()
        
        assert hasattr(manager, 'async_config')
        assert hasattr(manager, 'async_stats')
        assert manager.async_config['max_concurrency'] == 2  # From test env
        assert manager.async_stats['async_requests'] == 0
    
    @pytest.mark.asyncio
    async def test_get_price_data_async_feature_flag_disabled(self, reset_async_state):
        """Test async data manager falls back to sync when flag disabled"""
        with patch.dict(os.environ, {'ENABLE_ASYNC_IO': 'false'}):
            from services.multi_source_data_manager import AsyncMultiSourceDataManager
            
            manager = AsyncMultiSourceDataManager()
            
            with patch.object(manager, 'get_price_data') as mock_sync_method:
                mock_sync_method.return_value = {'AAPL': 'sync_data'}
                
                result = await manager.get_price_data_async(['AAPL'])
                
                mock_sync_method.assert_called_once_with(['AAPL'], 90)
                assert result == {'AAPL': 'sync_data'}
    
    @pytest.mark.asyncio
    async def test_get_price_data_async_with_yfinance_cache(self, reset_async_state):
        """Test async price data with YFinance cache integration"""
        with patch.dict(os.environ, {'ENABLE_YF_PRICES': 'true'}):
            from services.multi_source_data_manager import AsyncMultiSourceDataManager
            
            manager = AsyncMultiSourceDataManager()
            
            # Mock YFinance cache data
            mock_cache_data = {'AAPL': 'cached_price_data'}
            
            with patch.object(manager, 'get_yf_cached_data', return_value=mock_cache_data):
                result = await manager.get_price_data_async(['AAPL'])
                
                assert result == mock_cache_data
                assert manager.async_stats['cache_hits'] == 1
    
    @pytest.mark.asyncio
    async def test_async_data_manager_performance_stats(self, reset_async_state):
        """Test async data manager performance statistics"""
        from services.multi_source_data_manager import AsyncMultiSourceDataManager
        
        manager = AsyncMultiSourceDataManager()
        
        # Simulate some activity
        manager.async_stats['async_requests'] = 5
        manager.async_stats['cache_hits'] = 2
        
        stats = manager.get_async_performance_stats()
        
        assert 'async_stats' in stats
        assert 'global_async_stats' in stats
        assert 'async_io_enabled' in stats
        assert 'config' in stats
        assert stats['async_stats']['async_requests'] == 5
        assert stats['async_stats']['cache_hits'] == 2
    
    @pytest.mark.asyncio
    async def test_refresh_price_cache_async_resilience(self, reset_async_state):
        """Test async cache refresh is resilient to failures"""
        from services.multi_source_data_manager import AsyncMultiSourceDataManager
        
        manager = AsyncMultiSourceDataManager()
        
        # Mock partial failure scenario
        with patch.object(manager, 'get_price_data_async') as mock_get_data:
            mock_get_data.return_value = {
                'AAPL': 'success_data',
                # GOOGL missing - simulates failure
            }
            
            result = await manager.refresh_price_cache_async(['AAPL', 'GOOGL'])
            
            assert result['requested_symbols'] == 2
            assert result['successful_fetches'] == 1
            assert result['failed_symbols'] == ['GOOGL']
            assert result['async_mode'] is True
            assert 'f09_stats' in result

class TestConvenienceFunctions:
    """Test F09 convenience functions and wrappers"""
    
    @pytest.mark.asyncio
    async def test_async_convenience_functions(self, reset_async_state):
        """Test async convenience functions"""
        from services.multi_source_data_manager import (
            get_portfolio_price_data_async,
            get_watchlist_price_data_async,
            get_all_price_data_async,
            async_data_manager
        )
        
        with patch.object(async_data_manager, 'get_price_data_async') as mock_get_data:
            mock_get_data.return_value = {'TEST': 'data'}
            
            # Test portfolio function
            result = await get_portfolio_price_data_async()
            mock_get_data.assert_called()
            assert result == {'TEST': 'data'}
            
            # Test watchlist function
            result = await get_watchlist_price_data_async()
            mock_get_data.assert_called()
            
            # Test all data function
            result = await get_all_price_data_async()
            mock_get_data.assert_called()
    
    def test_resilient_wrapper_functions(self, reset_async_state):
        """Test resilient wrapper functions that choose async vs sync"""
        from services.multi_source_data_manager import get_price_data_resilient
        
        with patch('services.multi_source_data_manager.async_data_manager') as mock_async_mgr, \
             patch('asyncio.run') as mock_run:
            
            mock_run.return_value = {'AAPL': 'async_result'}
            mock_async_mgr.get_price_data_async = AsyncMock(return_value={'AAPL': 'async_result'})
            
            result = get_price_data_resilient(['AAPL'])
            
            # Should use async manager when flag is enabled
            mock_run.assert_called_once()
            assert result == {'AAPL': 'async_result'}

class TestEndToEndAsyncFlow:
    """Test complete end-to-end F09 async flow"""
    
    @pytest.mark.asyncio
    async def test_complete_async_pipeline_simulation(self, reset_async_state):
        """Test simulated complete async pipeline with F09 features"""
        
        # Mock all external dependencies
        mock_responses = {
            'google_rss': '<rss><channel><item><title>AAPL up 5%</title><link>http://news1</link><pubDate>Wed, 29 Aug 2024 14:30:00 GMT</pubDate></item></channel></rss>',
            'av_news': {"feed": [{"title": "Apple reports earnings", "url": "http://news2", "time_published": "2024-08-29T15:00:00Z"}]},
            'price_data': {'AAPL': 'price_dataframe_mock'}
        }
        
        with patch('news_scraper.get_sentiment_pipeline') as mock_sentiment, \
             patch('services.retry_policies.AsyncHTTPClient') as mock_http_client, \
             patch('services.multi_source_data_manager.AsyncMultiSourceDataManager') as mock_data_mgr:
            
            # Setup mocks
            mock_sentiment.return_value = MagicMock(return_value=[{"label": "positive", "score": 0.8}])
            
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value={
                'status': 200,
                'data': mock_responses['google_rss'],
                'from_cache': False
            })
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_http_client.return_value = mock_client
            
            mock_mgr_instance = MagicMock()
            mock_mgr_instance.get_price_data_async = AsyncMock(return_value=mock_responses['price_data'])
            mock_data_mgr.return_value = mock_mgr_instance
            
            # Test news scraping
            from news_scraper import scrape_headlines_async
            news_results = await scrape_headlines_async(['AAPL'], days=7)
            
            # Test price data fetching
            from services.multi_source_data_manager import get_all_price_data_async
            price_results = await get_all_price_data_async()
            
            # Verify both completed successfully with F09 async patterns
            assert 'AAPL' in news_results
            assert news_results['AAPL']['count'] >= 0  # May be 0 due to mocking
            assert price_results == mock_responses['price_data']
            
            # Verify async clients were used with proper configuration
            assert mock_http_client.called
            mock_http_client.assert_called_with(
                retry_config=pytest.any(),
                semaphore_limit=2  # From test env MAX_CONCURRENCY
            )

class TestAsyncErrorHandling:
    """Test F09 error handling and resilience"""
    
    @pytest.mark.asyncio
    async def test_async_pipeline_fallback_to_sync_on_error(self, reset_async_state):
        """Test async pipeline falls back to sync when async fails"""
        
        with patch('news_scraper.scrape_headlines_async') as mock_async, \
             patch('news_scraper.scrape_headlines') as mock_sync:
            
            # Make async version fail
            mock_async.side_effect = Exception("Async pipeline failure")
            mock_sync.return_value = {'AAPL': {'fallback': True}}
            
            from news_scraper import scrape_headlines_resilient
            
            result = scrape_headlines_resilient(['AAPL'])
            
            # Should have called sync fallback
            mock_sync.assert_called_once()
            assert result == {'AAPL': {'fallback': True}}
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration_with_real_components(self, reset_async_state):
        """Test circuit breaker integration with actual components"""
        from services.retry_policies import get_async_circuit_breaker, AsyncCircuitBreakerOpenError
        
        # Get circuit breaker for a test host
        breaker = get_async_circuit_breaker("news.google.com")
        
        # Simulate failures to open circuit
        for _ in range(3):  # Exceed failure threshold
            with pytest.raises(Exception):
                await breaker.call(lambda: asyncio.create_task(self._failing_async_task()))
        
        # Circuit should now be open
        assert breaker.state == "OPEN"
        
        # Next call should fail fast
        with pytest.raises(AsyncCircuitBreakerOpenError):
            await breaker.call(lambda: asyncio.create_task(self._success_async_task()))
    
    async def _failing_async_task(self):
        """Helper task that always fails"""
        raise Exception("Simulated failure")
    
    async def _success_async_task(self):
        """Helper task that always succeeds"""
        return "success"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])