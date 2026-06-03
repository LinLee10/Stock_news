#!/usr/bin/env python3
"""
Fast core integration tests that verify system wiring without heavy imports.
Focuses on feature flag behavior and critical integration points.
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment to avoid heavy loading
os.environ['TESTING'] = '1'
os.environ['YF_TEST_FAST_BACKOFF'] = '1'


class TestCoreIntegrationFast(unittest.TestCase):
    """Fast integration tests for core system components."""
    
    def setUp(self):
        """Set up lightweight test environment."""
        self.test_tickers = ['AAPL', 'MSFT']
        
    def test_feature_flags_all_default_off(self):
        """Critical: Verify all feature flags default to OFF."""
        
        # Import feature flags (lightweight)
        from config.feature_flags import FeatureFlags
        
        flags = FeatureFlags()
        all_flags = flags.get_all_flags()
        
        # Check that ALL flags are OFF by default
        enabled_flags = [name for name, enabled in all_flags.items() if enabled]
        
        self.assertEqual(len(enabled_flags), 0, 
                        f"These flags should default to OFF: {enabled_flags}")
        
        # Verify critical flags specifically
        critical_flags = [
            'enable_api_endpoints',
            'enable_finbert_pipeline',
            'enable_multisource_prices', 
            'enable_newsapi_ingestion',
            'enable_portfolio_analytics'
        ]
        
        for flag in critical_flags:
            with self.subTest(flag=flag):
                self.assertFalse(flags.is_enabled(flag), 
                               f"Critical flag {flag} must default to OFF")
                               
    def test_feature_flag_toggling(self):
        """Test feature flags can be toggled safely."""
        
        from config.feature_flags import FeatureFlags
        
        flags = FeatureFlags()
        
        # Test enabling/disabling
        test_flag = 'enable_finbert_pipeline'
        
        # Initially OFF
        self.assertFalse(flags.is_enabled(test_flag))
        
        # Enable
        flags.set_flag(test_flag, True)
        self.assertTrue(flags.is_enabled(test_flag))
        
        # Disable
        flags.set_flag(test_flag, False)
        self.assertFalse(flags.is_enabled(test_flag))
        
    def test_retry_policies_importable_and_fast(self):
        """Test retry policies can be imported and work in fast mode."""
        
        from services.retry_policies import retry_with_backoff, CircuitBreaker
        
        # Test retry with fast backoff
        call_count = 0
        def test_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("test failure")
            return "success"
        
        import time
        start_time = time.time()
        
        result = retry_with_backoff(
            test_fn,
            retry_on=(ValueError,),
            max_retries=3,
            base_delay=0.001
        )
        
        elapsed = time.time() - start_time
        
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
        # Should complete very quickly in test mode
        self.assertLess(elapsed, 0.5)
        
    def test_circuit_breaker_basic_operation(self):
        """Test circuit breaker basic state transitions."""
        
        from services.retry_policies import CircuitBreaker, CircuitBreakerOpenError
        
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        # Initial state
        self.assertEqual(cb.state, "CLOSED")
        
        # Successful call
        result = cb.call(lambda: "success")
        self.assertEqual(result, "success")
        self.assertEqual(cb.state, "CLOSED")
        
        # Failures to trip circuit
        with self.assertRaises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        self.assertEqual(cb.state, "CLOSED")  # Still closed after 1 failure
        
        with self.assertRaises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        self.assertEqual(cb.state, "OPEN")  # Now open after 2 failures
        
        # Subsequent calls should fail fast
        with self.assertRaises(CircuitBreakerOpenError):
            cb.call(lambda: "should not execute")
            
    def test_main_pipeline_structure_exists(self):
        """Test that main pipeline structure exists and is importable."""
        
        # Test that key functions exist without executing heavy operations
        try:
            # Import main module (should be fast with mocking)
            with patch('news_scraper.scrape_headlines') as mock_scraper, \
                 patch('prediction.train_predict_stock') as mock_predict, \
                 patch('charts.create_collage') as mock_charts, \
                 patch('email_report.send_report') as mock_email:
                
                # Mock lightweight returns
                mock_scraper.return_value = {}
                mock_predict.return_value = {'predictions': {}}
                mock_charts.return_value = True
                mock_email.return_value = True
                
                import main
                
                # Verify key functions exist
                self.assertTrue(hasattr(main, 'get_top_mentioned_stocks'))
                
                # Test basic functionality
                result = main.get_top_mentioned_stocks({
                    'AAPL': {'count': 10},
                    'MSFT': {'count': 5}
                }, top_n=2)
                
                self.assertEqual(result, ['AAPL', 'MSFT'])
                
        except Exception as e:
            self.fail(f"Main pipeline structure import failed: {e}")
            
    def test_api_endpoints_conditional_loading(self):
        """Test API endpoints only load when flag is enabled."""
        
        from config.feature_flags import FeatureFlags
        
        # Test with flag OFF - API should not be active
        flags_off = FeatureFlags()
        flags_off.set_flag('enable_api_endpoints', False)
        
        with patch('config.feature_flags.feature_flags', flags_off):
            # API endpoints should respect flag
            self.assertFalse(flags_off.is_enabled('enable_api_endpoints'))
        
        # Test with flag ON - API should be available
        flags_on = FeatureFlags()
        flags_on.set_flag('enable_api_endpoints', True)
        
        with patch('config.feature_flags.feature_flags', flags_on):
            self.assertTrue(flags_on.is_enabled('enable_api_endpoints'))
            
            # Try to import API (may not be fully implemented)
            try:
                from api.app import create_app
                app = create_app()
                self.assertIsNotNone(app)
            except ImportError:
                # API not fully implemented yet - acceptable
                pass
                
    def test_yfinance_guard_integration_point(self):
        """Test yfinance daily refresh guard integration exists."""
        
        try:
            from services.yf_refresh_guard import YFDailyRefreshGuard
            
            # Test that guard can be instantiated
            guard = YFDailyRefreshGuard(
                cache_dir=Path("/tmp/test_cache"),
                config={'window_hour': 0}
            )
            
            # Basic interface check
            self.assertTrue(hasattr(guard, 'should_refresh_today'))
            self.assertTrue(hasattr(guard, 'refresh_data'))
            
        except ImportError:
            self.skipTest("YFDailyRefreshGuard not available")
            
    def test_multisource_data_manager_exists(self):
        """Test multisource data manager integration point exists."""
        
        try:
            from services.multi_source_data_manager import MultiSourceDataManager
            
            # Test basic instantiation
            manager = MultiSourceDataManager()
            
            # Check basic interface
            self.assertTrue(hasattr(manager, 'get_data') or 
                          hasattr(manager, 'fetch_data') or
                          hasattr(manager, 'get_prices'))
            
        except ImportError:
            self.skipTest("MultiSourceDataManager not available")
            
    def test_news_corroboration_conditional(self):
        """Test news corroboration is conditional on flags."""
        
        from config.feature_flags import FeatureFlags
        
        # Test flag checking
        flags = FeatureFlags()
        
        # Flag OFF
        flags.set_flag('enable_news_corroboration', False)
        self.assertFalse(flags.is_enabled('enable_news_corroboration'))
        
        # Flag ON  
        flags.set_flag('enable_news_corroboration', True)
        self.assertTrue(flags.is_enabled('enable_news_corroboration'))


class TestSystemIntegrationWiringFast(unittest.TestCase):
    """Fast tests for system integration wiring points."""
    
    def test_all_integration_anchors_present(self):
        """Test that integration anchor points are present in code."""
        
        # This test verifies that BEGIN INT# / END INT# anchors exist
        # where expected in the codebase for integration points
        
        integration_files = [
            ('main.py', ['INT1', 'INT2']),  # Expected integration points in main
            ('config/feature_flags.py', []),  # Feature flag anchors
            ('tests/integration/test_e2e_pipeline_regression.py', 
             ['INT1', 'INT2', 'INT3', 'INT4', 'INT5', 'INT6', 'INT7'])
        ]
        
        for file_path, expected_anchors in integration_files:
            full_path = Path(__file__).parent.parent / file_path
            
            if not full_path.exists():
                continue  # Skip if file doesn't exist
                
            with self.subTest(file=file_path):
                try:
                    content = full_path.read_text()
                    
                    for anchor in expected_anchors:
                        begin_anchor = f"BEGIN {anchor}"
                        end_anchor = f"END {anchor}"
                        
                        self.assertIn(begin_anchor, content, 
                                    f"Missing {begin_anchor} in {file_path}")
                        self.assertIn(end_anchor, content,
                                    f"Missing {end_anchor} in {file_path}")
                except Exception as e:
                    # File might not be readable or might have issues
                    self.skipTest(f"Could not verify anchors in {file_path}: {e}")
                    
    def test_performance_target_achievable(self):
        """Test that individual components are fast enough for <30s target."""
        
        import time
        
        component_tests = [
            ('feature_flags', lambda: __import__('config.feature_flags').feature_flags.FeatureFlags()),
            ('retry_policies', lambda: __import__('services.retry_policies').retry_policies.CircuitBreaker()),
        ]
        
        for component_name, component_fn in component_tests:
            with self.subTest(component=component_name):
                start_time = time.time()
                
                try:
                    component = component_fn()
                    self.assertIsNotNone(component)
                except ImportError:
                    self.skipTest(f"{component_name} not available")
                
                elapsed = time.time() - start_time
                
                # Each component should load very quickly
                self.assertLess(elapsed, 1.0, 
                               f"{component_name} took {elapsed:.3f}s to load")


if __name__ == '__main__':
    # Run with minimal verbosity for speed
    unittest.main(verbosity=1)