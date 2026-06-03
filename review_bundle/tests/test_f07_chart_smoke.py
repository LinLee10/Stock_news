#!/usr/bin/env python3
"""
F07 Chart Visual Smoke Tests

Visual smoke tests to ensure chart generation works correctly with F07 upgrades.
"""

import sys
import os
import unittest
import tempfile
import shutil
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from charts import create_collage, create_enhanced_collage
from config.feature_flags import feature_flags


class TestF07ChartSmoke(unittest.TestCase):
    """Visual smoke tests for F07 chart upgrades."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.test_dir))
        
        # Create realistic test data
        self.tickers = ['AAPL', 'MSFT', 'GOOGL']
        self.price_data = self._create_test_price_data()
        self.forecast_data = self._create_test_forecast_data()
        
        # Save original feature flag states
        self.original_flags = {}
        for flag in ['enable_portfolio_analytics', 'enable_smart_alerts', 'enable_finbert_pipeline']:
            self.original_flags[flag] = feature_flags.is_enabled(flag)
    
    def tearDown(self):
        """Reset feature flags."""
        for flag, state in self.original_flags.items():
            feature_flags.set_flag(flag, state)
    
    def _create_test_price_data(self):
        """Create realistic price data for testing."""
        price_data = {}
        base_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0}
        
        for ticker in self.tickers:
            dates = pd.date_range('2024-12-01', periods=30, freq='D')
            base_price = base_prices[ticker]
            
            # Create realistic price movements
            prices = []
            current_price = base_price
            for i in range(30):
                # Random walk with slight upward trend
                change = (i * 0.5 + (i % 7) * 2 - 3) * 0.01  # Realistic daily changes
                current_price *= (1 + change)
                prices.append(current_price)
            
            price_data[ticker] = pd.DataFrame({
                'Date': dates,
                'Close': prices
            })
        
        return price_data
    
    def _create_test_forecast_data(self):
        """Create realistic forecast data for testing."""
        forecast_data = {}
        
        for ticker in self.tickers:
            last_price = self.price_data[ticker]['Close'].iloc[-1]
            dates = pd.date_range('2024-12-31', periods=3, freq='D')
            
            # Create forecasts with slight upward trend
            forecasts = [last_price * (1 + i * 0.01) for i in range(1, 4)]
            
            forecast_data[ticker] = pd.DataFrame({
                'Date': dates,
                'Forecast_Close': forecasts
            })
        
        return forecast_data
    
    def _create_test_portfolio_analytics(self):
        """Create test portfolio analytics data."""
        return {
            'sector_allocation': {
                'Technology': 0.65,
                'Healthcare': 0.20,
                'Financial': 0.15
            },
            'beta_stats': {
                'AAPL': {'beta': 1.15, 'r_squared': 0.82, 'ticker_volatility': 24.5},
                'MSFT': {'beta': 0.95, 'r_squared': 0.78, 'ticker_volatility': 22.1},
                'GOOGL': {'beta': 1.25, 'r_squared': 0.85, 'ticker_volatility': 28.3}
            },
            'benchmark_performance': {
                '^GSPC': {'1M': 2.3, '3M': 5.7, '1Y': 12.4},
                '^IXIC': {'1M': 3.1, '3M': 7.2, '1Y': 16.8},
                '^DJI': {'1M': 1.8, '3M': 4.9, '1Y': 10.2}
            },
            'portfolio_performance': {'1M': 4.2, '3M': 8.1, '1Y': 18.7}
        }
    
    def _create_test_smart_alerts(self):
        """Create test smart alerts data."""
        from services.monitoring_alerting import SmartAlert
        
        return [
            SmartAlert(
                alert_id='test1', symbol='AAPL', alert_type='price_move',
                severity='HIGH', title='Price Alert: AAPL up 8.2%',
                description='AAPL moved up 8.2% to $163.45',
                timestamp=datetime.now(timezone.utc), current_value=163.45,
                previous_value=151.12, change_percent=8.2, guidance='Monitor momentum',
                metadata={'direction': 'up'}
            ),
            SmartAlert(
                alert_id='test2', symbol='MSFT', alert_type='sentiment_swing',
                severity='MEDIUM', title='Sentiment Alert: MSFT sentiment improved',
                description='MSFT sentiment changed from -0.1 to 0.4 (+0.5)',
                timestamp=datetime.now(timezone.utc), current_value=0.4,
                previous_value=-0.1, change_percent=None, guidance='Review news coverage',
                metadata={'direction': 'improved'}
            ),
            SmartAlert(
                alert_id='test3', symbol='GOOGL', alert_type='earnings_proximity',
                severity='LOW', title='Earnings Alert: GOOGL reports in 7 days',
                description='GOOGL earnings expected in 7 days',
                timestamp=datetime.now(timezone.utc), current_value=7,
                previous_value=None, change_percent=None, guidance='Review estimates',
                metadata={'days_until': 7}
            )
        ]
    
    def test_legacy_chart_creation_smoke(self):
        """Smoke test: legacy chart creation works without errors."""
        chart_path = self.test_dir / "legacy_smoke_test.png"
        
        # Should not raise any exceptions
        result = create_collage(
            self.tickers, self.price_data, self.forecast_data,
            "Legacy Smoke Test Chart", str(chart_path)
        )
        
        # Verify chart file was created
        self.assertTrue(chart_path.exists())
        self.assertEqual(result, str(chart_path))
        
        # Verify file is not empty
        self.assertGreater(chart_path.stat().st_size, 0)
    
    def test_enhanced_chart_baseline_smoke(self):
        """Smoke test: enhanced chart with no optional features."""
        # Disable all features
        feature_flags.set_flag('enable_portfolio_analytics', False)
        feature_flags.set_flag('enable_smart_alerts', False)
        feature_flags.set_flag('enable_finbert_pipeline', False)
        
        chart_path = self.test_dir / "enhanced_baseline_smoke.png"
        
        # Should work like legacy chart when no features enabled
        result = create_enhanced_collage(
            self.tickers, self.price_data, self.forecast_data,
            "Enhanced Baseline Smoke Test", str(chart_path)
        )
        
        self.assertTrue(chart_path.exists())
        self.assertEqual(result, str(chart_path))
        self.assertGreater(chart_path.stat().st_size, 0)
    
    def test_enhanced_chart_portfolio_analytics_smoke(self):
        """Smoke test: enhanced chart with portfolio analytics pane."""
        # Enable only portfolio analytics
        feature_flags.set_flag('enable_portfolio_analytics', True)
        feature_flags.set_flag('enable_smart_alerts', False)
        feature_flags.set_flag('enable_finbert_pipeline', False)
        
        chart_path = self.test_dir / "enhanced_portfolio_smoke.png"
        portfolio_analytics = self._create_test_portfolio_analytics()
        
        result = create_enhanced_collage(
            self.tickers, self.price_data, self.forecast_data,
            "Enhanced Portfolio Analytics Smoke Test", str(chart_path),
            portfolio_analytics=portfolio_analytics
        )
        
        self.assertTrue(chart_path.exists())
        self.assertEqual(result, str(chart_path))
        self.assertGreater(chart_path.stat().st_size, 0)
    
    def test_enhanced_chart_smart_alerts_smoke(self):
        """Smoke test: enhanced chart with smart alerts pane."""
        # Enable only smart alerts
        feature_flags.set_flag('enable_portfolio_analytics', False)
        feature_flags.set_flag('enable_smart_alerts', True)
        feature_flags.set_flag('enable_finbert_pipeline', False)
        
        chart_path = self.test_dir / "enhanced_alerts_smoke.png"
        smart_alerts = self._create_test_smart_alerts()
        
        result = create_enhanced_collage(
            self.tickers, self.price_data, self.forecast_data,
            "Enhanced Smart Alerts Smoke Test", str(chart_path),
            smart_alerts=smart_alerts
        )
        
        self.assertTrue(chart_path.exists())
        self.assertEqual(result, str(chart_path))
        self.assertGreater(chart_path.stat().st_size, 0)
    
    def test_enhanced_chart_all_features_smoke(self):
        """Smoke test: enhanced chart with all features enabled."""
        # Enable all features
        feature_flags.set_flag('enable_portfolio_analytics', True)
        feature_flags.set_flag('enable_smart_alerts', True)
        feature_flags.set_flag('enable_finbert_pipeline', True)
        
        chart_path = self.test_dir / "enhanced_full_smoke.png"
        portfolio_analytics = self._create_test_portfolio_analytics()
        smart_alerts = self._create_test_smart_alerts()
        benchmark_data = portfolio_analytics['benchmark_performance']
        
        result = create_enhanced_collage(
            self.tickers, self.price_data, self.forecast_data,
            "Enhanced Full Features Smoke Test", str(chart_path),
            portfolio_analytics=portfolio_analytics,
            benchmark_data=benchmark_data,
            smart_alerts=smart_alerts,
            finbert_results={'AAPL': {'recommendation': None}}  # Mock FinBERT
        )
        
        self.assertTrue(chart_path.exists())
        self.assertEqual(result, str(chart_path))
        self.assertGreater(chart_path.stat().st_size, 0)
    
    def test_chart_handles_empty_data_gracefully(self):
        """Smoke test: charts handle empty/missing data gracefully."""
        chart_path = self.test_dir / "empty_data_smoke.png"
        
        # Empty data
        empty_price_data = {ticker: pd.DataFrame(columns=['Date', 'Close']) for ticker in self.tickers}
        empty_forecast_data = {ticker: pd.DataFrame(columns=['Date', 'Forecast_Close']) for ticker in self.tickers}
        
        # Should not crash with empty data
        result = create_enhanced_collage(
            self.tickers, empty_price_data, empty_forecast_data,
            "Empty Data Smoke Test", str(chart_path),
            portfolio_analytics={},  # Empty analytics
            smart_alerts=[],         # No alerts
            benchmark_data={}        # No benchmarks
        )
        
        self.assertTrue(chart_path.exists())
        self.assertEqual(result, str(chart_path))
        self.assertGreater(chart_path.stat().st_size, 0)
    
    def test_chart_handles_single_ticker(self):
        """Smoke test: charts work with single ticker."""
        single_ticker = ['AAPL']
        single_price_data = {'AAPL': self.price_data['AAPL']}
        single_forecast_data = {'AAPL': self.forecast_data['AAPL']}
        
        chart_path = self.test_dir / "single_ticker_smoke.png"
        
        result = create_enhanced_collage(
            single_ticker, single_price_data, single_forecast_data,
            "Single Ticker Smoke Test", str(chart_path)
        )
        
        self.assertTrue(chart_path.exists())
        self.assertEqual(result, str(chart_path))
        self.assertGreater(chart_path.stat().st_size, 0)
    
    def test_chart_handles_many_tickers(self):
        """Smoke test: charts work with many tickers."""
        many_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
        many_price_data = {}
        many_forecast_data = {}
        
        # Create data for additional tickers
        for ticker in many_tickers:
            if ticker in self.price_data:
                many_price_data[ticker] = self.price_data[ticker]
                many_forecast_data[ticker] = self.forecast_data[ticker]
            else:
                # Create mock data for new tickers
                dates = pd.date_range('2024-12-01', periods=30, freq='D')
                prices = [100.0 + i for i in range(30)]
                many_price_data[ticker] = pd.DataFrame({'Date': dates, 'Close': prices})
                
                fc_dates = pd.date_range('2024-12-31', periods=3, freq='D')
                forecasts = [130.0 + i for i in range(3)]
                many_forecast_data[ticker] = pd.DataFrame({'Date': fc_dates, 'Forecast_Close': forecasts})
        
        chart_path = self.test_dir / "many_tickers_smoke.png"
        
        result = create_enhanced_collage(
            many_tickers, many_price_data, many_forecast_data,
            "Many Tickers Smoke Test", str(chart_path)
        )
        
        self.assertTrue(chart_path.exists())
        self.assertEqual(result, str(chart_path))
        self.assertGreater(chart_path.stat().st_size, 0)
    
    def test_chart_matplotlib_backend_compatibility(self):
        """Smoke test: charts work with different matplotlib backends."""
        import matplotlib
        original_backend = matplotlib.get_backend()
        
        # Test with Agg backend (common in headless environments)
        try:
            matplotlib.use('Agg')
            
            chart_path = self.test_dir / "backend_smoke.png"
            
            result = create_collage(
                ['AAPL'], {'AAPL': self.price_data['AAPL']}, 
                {'AAPL': self.forecast_data['AAPL']},
                "Backend Compatibility Test", str(chart_path)
            )
            
            self.assertTrue(chart_path.exists())
            self.assertGreater(chart_path.stat().st_size, 0)
            
        finally:
            # Restore original backend
            matplotlib.use(original_backend)
    
    def test_chart_performance_smoke(self):
        """Smoke test: chart generation completes in reasonable time."""
        import time
        
        chart_path = self.test_dir / "performance_smoke.png"
        
        start_time = time.time()
        
        # Enable all features for maximum complexity
        feature_flags.set_flag('enable_portfolio_analytics', True)
        feature_flags.set_flag('enable_smart_alerts', True)
        feature_flags.set_flag('enable_finbert_pipeline', True)
        
        result = create_enhanced_collage(
            self.tickers, self.price_data, self.forecast_data,
            "Performance Smoke Test", str(chart_path),
            portfolio_analytics=self._create_test_portfolio_analytics(),
            benchmark_data=self._create_test_portfolio_analytics()['benchmark_performance'],
            smart_alerts=self._create_test_smart_alerts()
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Should complete within reasonable time (5 seconds)
        self.assertLess(generation_time, 5.0, f"Chart generation took {generation_time:.2f}s")
        
        self.assertTrue(chart_path.exists())
        self.assertEqual(result, str(chart_path))
    
    def test_chart_thread_safety_smoke(self):
        """Smoke test: charts can be generated concurrently."""
        import threading
        import time
        
        results = []
        errors = []
        
        def generate_chart(thread_id):
            try:
                chart_path = self.test_dir / f"thread_{thread_id}_smoke.png"
                
                result = create_collage(
                    ['AAPL'], {'AAPL': self.price_data['AAPL']},
                    {'AAPL': self.forecast_data['AAPL']},
                    f"Thread {thread_id} Test", str(chart_path)
                )
                
                results.append((thread_id, result, chart_path.exists()))
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=generate_chart, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Verify results
        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertEqual(len(results), 3)
        
        for thread_id, result_path, file_exists in results:
            self.assertTrue(file_exists, f"Thread {thread_id} chart file not created")


if __name__ == '__main__':
    unittest.main(verbosity=2)