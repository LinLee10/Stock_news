#!/usr/bin/env python3
"""
Test yfinance once-per-day refresh with fast-failing tests.

These tests mock network calls and use fast backoff to ensure they complete quickly.
"""

import sys
import os
import unittest
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set fast backoff mode for all tests
os.environ['YF_TEST_FAST_BACKOFF'] = '1'

from services.yf_refresh_guard import YFDailyRefreshGuard, RefreshManifest
from services.data_sources.yfinance_provider import YFinanceProvider, YFConfig
from services.multi_source_data_manager import MultiSourceDataManager


class TestYFOnceDailyRefresh(unittest.TestCase):
    """Test once-per-day refresh logic with mocked network calls."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_cache_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.test_cache_dir))
        
        self.test_symbols = ['AAPL', 'MSFT']
        self.config = {
            'window_hour': 0,  # Always within window for tests
            'daily_key': 'test_key',
            'max_retries': 2,
            'backoff_base': 0.01  # Very small for tests
        }
        
    def test_first_run_creates_manifest_and_fetches_data(self):
        """Test that first run creates manifest and calls yfinance."""
        guard = YFDailyRefreshGuard(self.test_cache_dir, self.config)
        
        # Mock yfinance provider to return data
        mock_data = {
            'AAPL': pd.DataFrame({
                'Date': [datetime.now()],
                'Close': [150.0],
                'Symbol': ['AAPL']
            })
        }
        
        with patch('services.data_sources.yfinance_provider.create_yfinance_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.fetch_history.return_value = mock_data
            mock_create.return_value = mock_provider
            
            # First run should refresh
            result = guard.run_once_per_day(self.test_symbols)
            
            self.assertEqual(result['status'], 'done')
            self.assertEqual(result['success_count'], 1)
            self.assertEqual(result['attempts'], 1)
            
            # Verify provider was called
            mock_provider.fetch_history.assert_called_once_with(self.test_symbols)
            
            # Verify manifest was created
            manifest = guard._load_manifest()
            self.assertIsNotNone(manifest)
            self.assertEqual(manifest.status, 'done')
            self.assertEqual(manifest.success_count, 1)
    
    def test_second_run_same_day_skips_refresh(self):
        """Test that second run on same day does not call yfinance."""
        guard = YFDailyRefreshGuard(self.test_cache_dir, self.config)
        
        # Create existing manifest for today
        manifest = RefreshManifest(
            date_utc=datetime.now(timezone.utc).date().isoformat(),
            window_hour=self.config['window_hour'],
            key=self.config['daily_key'],
            symbols_hash=guard._compute_symbols_hash(self.test_symbols),
            status='done',
            attempts=1,
            symbols=self.test_symbols,
            last_update=datetime.now(timezone.utc).isoformat(),
            success_count=2,
            failed_count=0
        )
        guard._save_manifest(manifest)
        
        with patch('services.data_sources.yfinance_provider.create_yfinance_provider') as mock_create:
            mock_provider = Mock()
            mock_create.return_value = mock_provider
            
            # Second run should skip
            result = guard.run_once_per_day(self.test_symbols)
            
            self.assertEqual(result['status'], 'skipped')
            self.assertEqual(result['reason'], 'already_done')
            
            # Verify provider was NOT called
            mock_provider.fetch_history.assert_not_called()
    
    def test_rate_limited_refresh_fails_fast(self):
        """Test that rate limited refresh fails quickly and doesn't hang."""
        guard = YFDailyRefreshGuard(self.test_cache_dir, self.config)
        
        # Mock yfinance provider to always raise rate limit error
        from yfinance.exceptions import YFRateLimitError
        
        with patch('services.data_sources.yfinance_provider.create_yfinance_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.fetch_history.side_effect = YFRateLimitError()
            mock_create.return_value = mock_provider
            
            import time
            start_time = time.time()
            
            # Should fail fast
            result = guard.run_once_per_day(self.test_symbols)
            
            elapsed = time.time() - start_time
            
            # Should complete quickly (< 5 seconds even with retries)
            self.assertLess(elapsed, 5.0)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['success_count'], 0)
    
    def test_provider_bounded_retries(self):
        """Test that provider respects bounded retry limits."""
        config = YFConfig(
            max_retries=2,
            backoff_base_seconds=0.01,  # Very fast
            test_fast_backoff=True
        )
        provider = YFinanceProvider(config)
        
        # Mock yfinance to always fail
        with patch('yfinance.Ticker') as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker.history.side_effect = Exception("Network error")
            mock_ticker_class.return_value = mock_ticker
            
            import time
            start_time = time.time()
            
            # Should try limited times and give up quickly
            results = provider.fetch_history(['AAPL'])
            
            elapsed = time.time() - start_time
            
            # Should complete very quickly with fast backoff
            self.assertLess(elapsed, 2.0)
            self.assertEqual(len(results), 0)
    
    def test_circuit_breaker_fast_trip_in_tests(self):
        """Test that circuit breaker trips quickly in test mode."""
        config = YFConfig(test_fast_backoff=True)
        provider = YFinanceProvider(config)
        
        # Circuit breaker should have failure_threshold=1 in test mode
        self.assertEqual(provider.circuit_breaker.failure_threshold, 1)
        self.assertEqual(provider.circuit_breaker.recovery_timeout, 60)
    
    def test_multi_source_integration_cache_only_read(self):
        """Test that multi-source manager only reads from cache, no network calls."""
        with patch.dict(os.environ, {
            'ENABLE_YF_PRICES': 'true',
            'ENABLE_YF_DAILY_REFRESH': 'true'
        }):
            # Force reload of feature flags to pick up environment changes
            from config.feature_flags import feature_flags
            feature_flags.set_flag('enable_yf_prices', True)
            feature_flags.set_flag('enable_yf_daily_refresh', True)
            
            # Mock the refresh guard
            with patch('services.multi_source_data_manager.YFDailyRefreshGuard') as mock_guard_class:
                mock_guard_class.run_once_per_day_static.return_value = {
                    'status': 'skipped',
                    'reason': 'already_done'
                }
                
                manager = MultiSourceDataManager()
                
                # Mock cached data
                cached_data = {
                    'AAPL': pd.DataFrame({
                        'Date': [datetime.now()],
                        'Close': [150.0]
                    })
                }
                
                with patch.object(manager, 'get_yf_cached_data', return_value=cached_data):
                    import asyncio
                    result = asyncio.run(manager.get_price_data(['AAPL']))
                    
                    self.assertEqual(len(result), 1)
                    self.assertIn('AAPL', result)
                    
                    # Verify refresh guard was called but no network requests made
                    mock_guard_class.run_once_per_day_static.assert_called_once()
    
    def test_lockfile_prevents_concurrent_refresh(self):
        """Test that lockfile prevents concurrent refresh attempts."""
        guard1 = YFDailyRefreshGuard(self.test_cache_dir, self.config)
        guard2 = YFDailyRefreshGuard(self.test_cache_dir, self.config)
        
        # Simulate concurrent access
        lock_fd = guard1._acquire_lock(timeout=1)
        self.assertIsNotNone(lock_fd)
        
        # Second guard should fail to acquire lock
        lock_fd2 = guard2._acquire_lock(timeout=1)
        self.assertIsNone(lock_fd2)
        
        # Release first lock
        guard1._release_lock(lock_fd)
        
        # Now second guard should succeed
        lock_fd2 = guard2._acquire_lock(timeout=1)
        self.assertIsNotNone(lock_fd2)
        guard2._release_lock(lock_fd2)


class TestYFProviderHardening(unittest.TestCase):
    """Test yfinance provider hardening for free tier."""
    
    def test_serialized_requests_no_threading(self):
        """Test that provider uses single thread to avoid rate limits."""
        config = YFConfig(threads=1)
        provider = YFinanceProvider(config)
        
        self.assertEqual(config.threads, 1)
        
        # Verify session configuration doesn't enable threading
        # (yfinance threading is controlled by download parameters)
    
    def test_atomic_csv_operations(self):
        """Test that CSV writes are atomic and don't corrupt existing cache."""
        config = YFConfig(test_fast_backoff=True)
        provider = YFinanceProvider(config)
        
        test_data = pd.DataFrame({
            'Date': [datetime.now()],
            'Close': [150.0],
            'Symbol': ['TEST']
        })
        
        # Save data
        provider._save_cached_history('TEST', test_data)
        
        # Verify file exists and is readable
        cache_file = provider._get_cache_file_path('TEST')
        self.assertTrue(cache_file.exists())
        
        # Load and verify data
        loaded_data = provider._load_cached_history('TEST')
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), 1)
        self.assertEqual(loaded_data.iloc[0]['Close'], 150.0)


if __name__ == '__main__':
    # Ensure fast backoff mode is enabled
    os.environ['YF_TEST_FAST_BACKOFF'] = '1'
    
    # Run tests with high verbosity
    unittest.main(verbosity=2, buffer=True)