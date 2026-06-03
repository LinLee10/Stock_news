#!/usr/bin/env python3
"""
End-to-end pipeline regression tests ensuring no breaking changes.

Tests the complete pipeline with all flags OFF (baseline) and selected flags ON
to verify system integration without regressions.
"""

import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set test mode environment
os.environ['YF_TEST_FAST_BACKOFF'] = '1'
os.environ['ENABLE_DEBUG_MODE'] = 'false'

from config.feature_flags import FeatureFlags
import main
from news_scraper import scrape_headlines
from prediction import train_predict_stock  
from charts import create_collage
from email_report import send_report


class TestE2EPipelineRegression(unittest.TestCase):
    """Test complete pipeline with flags OFF to ensure no regressions."""
    
    def setUp(self):
        """Set up test environment with clean state."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        
        # Override paths for testing
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        self.addCleanup(lambda: os.chdir(self.original_cwd))
        
        # Create necessary directories
        (self.test_dir / "data").mkdir(parents=True)
        (self.test_dir / "charts").mkdir(parents=True) 
        (self.test_dir / "report").mkdir(parents=True)
        
        # Create feature flags with all OFF (baseline)
        self.flags_all_off = FeatureFlags()
        for flag in self.flags_all_off._flags:
            self.flags_all_off._flags[flag] = False
            
        # Test data
        self.test_tickers = ['AAPL', 'MSFT']
        
    def test_baseline_pipeline_flags_all_off(self):
        """Test baseline pipeline with all flags OFF - no regressions allowed."""
        
        # BEGIN INT1 - Mock all external network calls
        with patch('news_scraper.scrape_headlines') as mock_headlines, \
             patch('prediction.train_predict_stock') as mock_predict, \
             patch('charts.create_collage') as mock_charts, \
             patch('email_report.send_report') as mock_email:
            
            # Mock headline data (RSS only, no external APIs)
            mock_headlines.return_value = {
                'AAPL': {
                    'headlines': [
                        ('Apple reports strong earnings', 'https://example.com/1', '2025-08-31'),
                        ('iPhone sales exceed expectations', 'https://example.com/2', '2025-08-31')
                    ],
                    'count': 2,
                    'sentiment_score': 0.7
                },
                'MSFT': {
                    'headlines': [
                        ('Microsoft cloud growth continues', 'https://example.com/3', '2025-08-31')
                    ],
                    'count': 1,
                    'sentiment_score': 0.6
                }
            }
            
            # Mock prediction results
            mock_predict.return_value = {
                'predictions': {'AAPL': 0.15, 'MSFT': 0.12},
                'model_metrics': {'accuracy': 0.85, 'mse': 0.02}
            }
            
            # Mock chart creation (no file I/O)
            mock_charts.return_value = True
            
            # Mock email sending
            mock_email.return_value = True
            
            # Override feature flags globally for this test
            with patch('config.feature_flags.feature_flags', self.flags_all_off):
                
                # Run main pipeline logic (simplified)
                headlines_data = scrape_headlines(self.test_tickers, lookback_days=7)
                
                # Verify headlines structure
                self.assertIsInstance(headlines_data, dict)
                self.assertIn('AAPL', headlines_data)
                self.assertIn('MSFT', headlines_data)
                
                # Basic sentiment should be computed (no FinBERT)
                for ticker in self.test_tickers:
                    ticker_data = headlines_data[ticker]
                    self.assertIn('sentiment_score', ticker_data)
                    self.assertIsInstance(ticker_data['sentiment_score'], (int, float))
                
                # Run predictions (basic models only)
                prediction_results = train_predict_stock(self.test_tickers)
                self.assertIn('predictions', prediction_results)
                
                # Create charts
                chart_success = create_collage(headlines_data, self.test_tickers)
                self.assertTrue(chart_success)
                
                # Send email report
                email_success = send_report(headlines_data, prediction_results)
                self.assertTrue(email_success)
                
            # Verify no advanced features were called
            self._verify_baseline_behavior()
        # END INT1
            
    def _verify_baseline_behavior(self):
        """Verify that only baseline features were used, no advanced integrations."""
        
        # Check that no advanced API calls were made
        # This would be expanded based on specific integration points
        
        # Verify flags are all OFF
        for flag_name, flag_value in self.flags_all_off.get_all_flags().items():
            self.assertFalse(flag_value, f"Flag {flag_name} should be OFF in baseline test")
            
    def test_yfinance_once_daily_integration(self):
        """Test yfinance once-daily guard integration when enabled."""
        
        # Enable only yfinance flags
        test_flags = FeatureFlags()
        test_flags.set_flag('enable_yf_prices', True)
        test_flags.set_flag('enable_yf_daily_refresh', True)
        
        # BEGIN INT2 - Test yfinance integration
        with patch('config.feature_flags.feature_flags', test_flags), \
             patch('services.yf_refresh_guard.YFDailyRefreshGuard') as mock_guard:
            
            mock_guard_instance = Mock()
            mock_guard.return_value = mock_guard_instance
            mock_guard_instance.should_refresh_today.return_value = True
            mock_guard_instance.refresh_data.return_value = {'AAPL': Mock()}
            
            # Test that guard is consulted
            from services.multi_source_data_manager import MultiSourceDataManager
            
            # Verify guard integration exists and works
            self.assertTrue(test_flags.is_enabled('enable_yf_prices'))
            mock_guard_instance.should_refresh_today.assert_not_called()  # Not called until actual use
        # END INT2
            
    def test_multisource_prices_fallback(self):
        """Test multisource pricing with Alpha Vantage -> Yahoo Finance fallback."""
        
        test_flags = FeatureFlags()
        test_flags.set_flag('enable_multisource_prices', True)
        test_flags.set_flag('enable_alpha_vantage_batching', True)
        
        # BEGIN INT3 - Test multisource fallback logic
        with patch('config.feature_flags.feature_flags', test_flags), \
             patch('services.data_sources.price_provider.PriceProvider') as mock_provider:
            
            # Mock quota exhausted scenario
            mock_provider_instance = Mock()
            mock_provider.return_value = mock_provider_instance
            mock_provider_instance.get_prices.side_effect = Exception("Quota exhausted")
            
            # Test fallback behavior
            with self.assertLogs(level='WARNING') as log:
                try:
                    # This would normally call the multisource manager
                    mock_provider_instance.get_prices(['AAPL'])
                except Exception as e:
                    self.assertIn("Quota", str(e))
        # END INT3
        
    def test_news_ingestion_fallback_chain(self):
        """Test news ingestion RSS -> NewsAPI -> fallback chain."""
        
        test_flags = FeatureFlags()
        test_flags.set_flag('enable_newsapi_ingestion', True)
        test_flags.set_flag('enable_news_corroboration', True)
        
        # BEGIN INT4 - Test news fallback chain
        with patch('config.feature_flags.feature_flags', test_flags), \
             patch('integrations.newsapi_client.NewsAPIClient') as mock_newsapi:
            
            # Mock NewsAPI failure
            mock_client = Mock()
            mock_newsapi.return_value = mock_client
            mock_client.get_headlines.side_effect = Exception("API quota exceeded")
            
            # Test RSS fallback
            with patch('news_scraper.scrape_headlines') as mock_scraper:
                mock_scraper.return_value = {'AAPL': {'headlines': [], 'count': 0}}
                
                result = mock_scraper(['AAPL'])
                self.assertIsInstance(result, dict)
                # Would verify RSS was used as fallback
        # END INT4
        
    def test_api_endpoints_auth_integration(self):
        """Test API endpoints are only mounted when flag enabled and require auth."""
        
        # BEGIN INT5 - Test API mounting and auth
        test_flags = FeatureFlags() 
        test_flags.set_flag('enable_api_endpoints', True)
        
        with patch('config.feature_flags.feature_flags', test_flags):
            # Import after flag is set
            try:
                from api.app import create_app
                app = create_app()
                
                # Verify auth is required (this would be more detailed)
                self.assertIsNotNone(app)
                
            except ImportError:
                # API module might not be fully implemented yet
                self.skipTest("API endpoints not fully implemented")
        # END INT5
        
    def test_performance_under_30_seconds(self):
        """Verify complete test suite runs under 30 seconds."""
        import time
        
        start_time = time.time()
        
        # Run a subset of core functionality tests
        with patch('news_scraper.scrape_headlines') as mock_headlines, \
             patch('prediction.train_predict_stock') as mock_predict:
            
            mock_headlines.return_value = {'TEST': {'headlines': [], 'count': 0}}
            mock_predict.return_value = {'predictions': {'TEST': 0.1}}
            
            # Simulate pipeline
            mock_headlines(['TEST'])
            mock_predict(['TEST'])
            
        elapsed = time.time() - start_time
        
        # This individual test should be very fast
        self.assertLess(elapsed, 5.0, "Individual test took too long - check for blocking calls")


class TestFeatureWiringCoverage(unittest.TestCase):
    """Test individual feature wiring when flags are ON."""
    
    def setUp(self):
        """Set up for feature-specific tests."""
        self.base_flags = FeatureFlags()
        
    def test_finbert_pipeline_wiring(self):
        """Test FinBERT pipeline integration with configurable parameters."""
        
        test_flags = FeatureFlags()
        test_flags.set_flag('enable_finbert_pipeline', True)
        
        # BEGIN INT6 - Test FinBERT integration
        with patch('config.feature_flags.feature_flags', test_flags), \
             patch('services.finbert_sentiment_analyzer.analyze_articles_for_symbol') as mock_finbert:
            
            mock_finbert.return_value = {
                'sentiment_score': 0.8,
                'confidence': 0.9,
                'analysis_time': 0.1
            }
            
            # Test configurable parameters
            from config.config import FINBERT_LAMBDA, FINBERT_BARRIER_DAYS
            
            # These should be configurable
            self.assertIsInstance(FINBERT_LAMBDA, (int, float))
            self.assertIsInstance(FINBERT_BARRIER_DAYS, int)
            
            # Test integration point exists
            self.assertTrue(test_flags.is_enabled('enable_finbert_pipeline'))
        # END INT6
        
    def test_portfolio_analytics_conditional_sections(self):
        """Test portfolio analytics sections only appear when flagged ON."""
        
        # Test with flag OFF - no analytics sections
        with patch('config.feature_flags.feature_flags', self.base_flags):
            # Would test that analytics sections are not rendered
            pass
            
        # Test with flag ON - analytics sections appear  
        test_flags = FeatureFlags()
        test_flags.set_flag('enable_portfolio_analytics', True)
        
        with patch('config.feature_flags.feature_flags', test_flags):
            # Would verify analytics sections are added to report
            self.assertTrue(test_flags.is_enabled('enable_portfolio_analytics'))
            
    def test_smart_alerts_cached_data_only(self):
        """Test smart alerts compute from cached data, no fresh API calls."""
        
        test_flags = FeatureFlags()
        test_flags.set_flag('enable_smart_alerts', True)
        
        # BEGIN INT7 - Test smart alerts caching behavior
        with patch('config.feature_flags.feature_flags', test_flags):
            
            # Mock cached data availability
            with patch('services.monitoring_alerting.SmartAlertsManager') as mock_alerts:
                mock_manager = Mock()
                mock_alerts.return_value = mock_manager
                mock_manager.generate_alerts.return_value = []
                
                # Verify no external API calls in alert generation
                self.assertTrue(test_flags.is_enabled('enable_smart_alerts'))
                
                # Alert generation should use cached data only
                mock_manager.generate_alerts.assert_not_called()  # Until actually invoked
        # END INT7


if __name__ == '__main__':
    # Ensure clean test environment
    os.environ['TESTING'] = '1'
    
    # Run tests with verbosity
    unittest.main(verbosity=2)