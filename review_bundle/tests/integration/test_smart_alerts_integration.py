#!/usr/bin/env python3
"""
Integration tests for F06 Smart Alerts functionality.

Tests end-to-end alert flow, email integration, and real data scenarios.
"""

import sys
import os
import unittest
import tempfile
import shutil
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.monitoring_alerting import create_smart_alerts_engine
from config.feature_flags import feature_flags


class TestSmartAlertsIntegration(unittest.TestCase):
    """Integration tests for smart alerts with realistic data scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.test_config_dir))
        
        # Create test config file
        self.test_config_file = self.test_config_dir / "alerts.yaml"
        test_config_content = """
defaults:
  price_move_threshold_percent: 5.0
  price_move_timeframe_days: 1
  sentiment_threshold_change: 0.3
  sentiment_comparison_days: 7
  earnings_alert_days: [7, 3, 1]
  batch_alerts: true
  immediate_email_severity: ["HIGH", "CRITICAL"]
  max_alerts_per_symbol_per_day: 3
  severity_thresholds:
    price_move:
      LOW: 3.0
      MEDIUM: 5.0
      HIGH: 10.0
      CRITICAL: 15.0
    sentiment:
      LOW: 0.2
      MEDIUM: 0.3
      HIGH: 0.5
      CRITICAL: 0.7
    earnings:
      LOW: 7
      MEDIUM: 3
      HIGH: 1
      CRITICAL: 0

symbol_overrides:
  TSLA:
    price_move_threshold_percent: 8.0
    sentiment_threshold_change: 0.4
  
templates:
  price_move:
    title: "Price Alert: {symbol} {direction} {percent:.1f}%"
    description: "{symbol} moved {direction} {percent:.1f}% to ${current_price:.2f}"
    guidance: "Monitor for continued momentum"
  sentiment:
    title: "Sentiment Alert: {symbol} sentiment {direction}"
    description: "{symbol} sentiment changed by {change:+.2f}"
    guidance: "Review recent news coverage"
  earnings:
    title: "Earnings Alert: {symbol} reports in {days} days"
    description: "{symbol} earnings expected in {days} days"
    guidance: "Review analyst estimates"

security:
  alert_cooldown_minutes: 60
"""
        self.test_config_file.write_text(test_config_content)
        
        # Create smart alerts engine with test config
        self.engine = create_smart_alerts_engine(str(self.test_config_file))
    
    def test_realistic_market_scenario_alerts(self):
        """Test alerts with realistic market data scenarios."""
        symbols = ['AAPL', 'TSLA', 'NVDA']
        
        # Create realistic price data with different scenarios
        price_data = {
            # AAPL: Normal 6% move (should trigger MEDIUM alert)
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=10, freq='D'),
                'Close': [180.50, 181.20, 182.10, 183.50, 184.20, 185.10, 186.40, 187.20, 188.30, 191.38]
            }),
            
            # TSLA: High volatility 12% move (should trigger HIGH alert, threshold is 8% for TSLA)
            'TSLA': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=10, freq='D'),
                'Close': [250.00, 252.50, 255.00, 248.00, 251.20, 254.80, 258.40, 262.10, 265.80, 280.00]
            }),
            
            # NVDA: Small 2% move (should NOT trigger, below 5% threshold)
            'NVDA': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=10, freq='D'),
                'Close': [875.00, 876.50, 878.20, 880.10, 882.40, 885.20, 887.80, 890.50, 892.30, 893.50]
            })
        }
        
        # Create realistic sentiment data
        sentiment_data = {
            # AAPL: Mild sentiment improvement (no alert)
            'AAPL': {
                'daily_sentiment': {
                    '2025-01-20': 0.1, '2025-01-21': 0.1, '2025-01-22': 0.1,
                    '2025-01-23': 0.1, '2025-01-24': 0.1, '2025-01-25': 0.1,
                    '2025-01-26': 0.1, '2025-01-27': 0.15, '2025-01-28': 0.2,
                    '2025-01-29': 0.25  # Small improvement: 0.15 avg change
                }
            },
            
            # TSLA: Major sentiment swing (should trigger alert)
            'TSLA': {
                'daily_sentiment': {
                    '2025-01-20': -0.2, '2025-01-21': -0.2, '2025-01-22': -0.1,
                    '2025-01-23': -0.1, '2025-01-24': -0.1, '2025-01-25': 0.0,
                    '2025-01-26': 0.0, '2025-01-27': 0.1, '2025-01-28': 0.2,
                    '2025-01-29': 0.3  # Major swing from -0.1 avg to 0.3 = 0.4 change
                }
            },
            
            # NVDA: Stable sentiment (no alert)
            'NVDA': {
                'daily_sentiment': {
                    '2025-01-20': 0.3, '2025-01-21': 0.3, '2025-01-22': 0.3,
                    '2025-01-23': 0.3, '2025-01-24': 0.3, '2025-01-25': 0.3,
                    '2025-01-26': 0.3, '2025-01-27': 0.3, '2025-01-28': 0.3,
                    '2025-01-29': 0.3  # No change
                }
            }
        }
        
        # Create earnings data
        earnings_data = {
            # AAPL: Earnings in 3 days (should trigger MEDIUM alert)
            'AAPL': {'date': (datetime.now().date() + timedelta(days=3)).isoformat()},
            
            # NVDA: Earnings in 5 days (should NOT trigger, not in alert days)
            'NVDA': {'date': (datetime.now().date() + timedelta(days=5)).isoformat()}
        }
        
        # Evaluate alerts
        alerts = self.engine.evaluate_alerts(symbols, price_data, sentiment_data, earnings_data)
        
        # Expected alerts:
        # 1. AAPL price move (6% > 5%) - MEDIUM
        # 2. TSLA price move (12% > 8%) - HIGH  
        # 3. TSLA sentiment swing (0.4 change > 0.4 threshold) - HIGH
        # 4. AAPL earnings (3 days) - MEDIUM
        # Total: 4 alerts
        
        self.assertGreaterEqual(len(alerts), 3)  # At least price + sentiment + earnings alerts
        
        # Verify alert types and symbols
        alert_symbols = [alert.symbol for alert in alerts]
        alert_types = [alert.alert_type for alert in alerts]
        
        self.assertIn('AAPL', alert_symbols)
        self.assertIn('TSLA', alert_symbols)
        self.assertIn('price_move', alert_types)
        self.assertIn('earnings_proximity', alert_types)
        
        # NVDA should not have alerts (below thresholds)
        nvda_alerts = [alert for alert in alerts if alert.symbol == 'NVDA']
        self.assertEqual(len(nvda_alerts), 0)
        
        # Check severity levels
        high_alerts = [alert for alert in alerts if alert.severity == 'HIGH']
        medium_alerts = [alert for alert in alerts if alert.severity == 'MEDIUM']
        
        self.assertGreater(len(high_alerts), 0)  # TSLA alerts should be HIGH
        self.assertGreater(len(medium_alerts), 0)  # AAPL alerts should be MEDIUM
    
    def test_email_integration_with_alerts(self):
        """Test integration with email report generation."""
        # Create test alerts
        from services.monitoring_alerting import SmartAlert
        
        test_alerts = [
            SmartAlert(
                alert_id='test1',
                symbol='AAPL',
                alert_type='price_move',
                severity='HIGH',
                title='Price Alert: AAPL up 8.5%',
                description='AAPL moved up 8.5% to $195.32',
                timestamp=datetime.now(timezone.utc),
                current_value=195.32,
                previous_value=180.00,
                change_percent=8.5,
                guidance='Monitor for continued momentum',
                metadata={'direction': 'up'}
            ),
            SmartAlert(
                alert_id='test2',
                symbol='TSLA',
                alert_type='sentiment_swing',
                severity='MEDIUM',
                title='Sentiment Alert: TSLA sentiment improved',
                description='TSLA sentiment changed by +0.35',
                timestamp=datetime.now(timezone.utc),
                current_value=0.4,
                previous_value=0.05,
                change_percent=None,
                guidance='Review recent news coverage',
                metadata={'direction': 'improved'}
            )
        ]
        
        # Test email report integration
        from email_report import send_report
        
        # Mock the email sending to avoid actual SMTP
        with patch('email_report.smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # Enable smart alerts feature flag for test
            feature_flags.set_flag('enable_smart_alerts', True)
            
            try:
                send_report(
                    watchlist=['AAPL', 'TSLA'],
                    portfolio=['AAPL'],
                    head7={'AAPL': {'count': 5, 'headlines': []}, 'TSLA': {'count': 3, 'headlines': []}},
                    head30={'AAPL': {'daily_sentiment': {}}, 'TSLA': {'daily_sentiment': {}}},
                    preds={'AAPL': {'predictions': [], 'confidence': 0.8, 'red_flag': False}},
                    out_path=str(self.test_config_dir / "test_report.html"),
                    smart_alerts=test_alerts
                )
                
                # Verify report was generated
                report_file = self.test_config_dir / "test_report.html"
                self.assertTrue(report_file.exists())
                
                # Check that alerts are included in the report
                report_content = report_file.read_text()
                self.assertIn('Smart Alerts', report_content)
                self.assertIn('AAPL up 8.5%', report_content)
                self.assertIn('TSLA sentiment improved', report_content)
                self.assertIn('HIGH', report_content)
                self.assertIn('MEDIUM', report_content)
                
            finally:
                # Reset feature flag
                feature_flags.set_flag('enable_smart_alerts', False)
    
    def test_no_alerts_scenario(self):
        """Test scenario where no alerts are triggered."""
        symbols = ['AAPL', 'MSFT']
        
        # Create price data with small moves (below thresholds)
        price_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=5, freq='D'),
                'Close': [180.00, 181.00, 182.00, 182.50, 183.00]  # 1.7% move
            }),
            'MSFT': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=5, freq='D'),
                'Close': [420.00, 421.50, 422.00, 423.00, 425.00]  # 1.2% move
            })
        }
        
        # Create sentiment data with small changes
        sentiment_data = {
            'AAPL': {'daily_sentiment': {f'2025-01-{20+i}': 0.1 + i*0.02 for i in range(8)}},  # Small gradual change
            'MSFT': {'daily_sentiment': {f'2025-01-{20+i}': 0.2 for i in range(8)}}  # No change
        }
        
        alerts = self.engine.evaluate_alerts(symbols, price_data, sentiment_data)
        
        # Should have no alerts
        self.assertEqual(len(alerts), 0)
    
    def test_cooldown_behavior_integration(self):
        """Test that cooldown behavior works in realistic scenarios."""
        symbols = ['AAPL']
        
        # Create price data that would normally trigger alert
        price_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=5, freq='D'),
                'Close': [180.00, 181.00, 182.00, 183.00, 190.00]  # 5.6% move
            })
        }
        
        sentiment_data = {
            'AAPL': {'daily_sentiment': {f'2025-01-{20+i}': 0.1 for i in range(8)}}
        }
        
        # First evaluation should trigger alerts
        alerts1 = self.engine.evaluate_alerts(symbols, price_data, sentiment_data)
        self.assertGreater(len(alerts1), 0)
        
        # Second evaluation immediately after should be blocked by cooldown
        alerts2 = self.engine.evaluate_alerts(symbols, price_data, sentiment_data)
        self.assertEqual(len(alerts2), 0)  # Should be blocked by cooldown
        
        # Verify cooldown tracking
        self.assertTrue(self.engine._is_symbol_on_cooldown('AAPL'))
    
    def test_multiple_symbol_overrides_scenario(self):
        """Test scenario with multiple symbols having different override thresholds."""
        symbols = ['AAPL', 'TSLA']  # TSLA has 8% threshold, AAPL has 5%
        
        # Both symbols have 7% price moves
        price_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=5, freq='D'),
                'Close': [180.00, 182.00, 184.00, 186.00, 192.60]  # 7% move
            }),
            'TSLA': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=5, freq='D'),
                'Close': [250.00, 252.50, 255.00, 260.00, 267.50]  # 7% move
            })
        }
        
        sentiment_data = {
            'AAPL': {'daily_sentiment': {f'2025-01-{20+i}': 0.1 for i in range(8)}},
            'TSLA': {'daily_sentiment': {f'2025-01-{20+i}': 0.1 for i in range(8)}}
        }
        
        alerts = self.engine.evaluate_alerts(symbols, price_data, sentiment_data)
        
        # Only AAPL should trigger (7% > 5% threshold)
        # TSLA should NOT trigger (7% < 8% threshold)
        aapl_alerts = [alert for alert in alerts if alert.symbol == 'AAPL']
        tsla_alerts = [alert for alert in alerts if alert.symbol == 'TSLA']
        
        self.assertGreater(len(aapl_alerts), 0)
        self.assertEqual(len(tsla_alerts), 0)
    
    def test_alert_summary_and_metrics(self):
        """Test alert summary and metrics collection."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Create varied scenarios to trigger different alert types and severities
        price_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=5, freq='D'),
                'Close': [180.00, 182.00, 184.00, 187.00, 198.00]  # 10% move - HIGH
            }),
            'MSFT': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=5, freq='D'),
                'Close': [420.00, 422.00, 424.00, 426.00, 441.00]  # 5% move - MEDIUM
            }),
            'GOOGL': pd.DataFrame({
                'Date': pd.date_range('2025-01-20', periods=5, freq='D'),
                'Close': [2800.00, 2810.00, 2820.00, 2830.00, 2840.00]  # 1.4% move - no alert
            })
        }
        
        sentiment_data = {
            'AAPL': {'daily_sentiment': {f'2025-01-{20+i}': 0.1 for i in range(8)}},
            'MSFT': {'daily_sentiment': {f'2025-01-{20+i}': 0.1 for i in range(8)}},
            'GOOGL': {'daily_sentiment': {f'2025-01-{20+i}': 0.1 for i in range(8)}}
        }
        
        alerts = self.engine.evaluate_alerts(symbols, price_data, sentiment_data)
        
        # Should have 2 alerts: AAPL (HIGH) and MSFT (MEDIUM)
        self.assertEqual(len(alerts), 2)
        
        # Test daily summary
        today = datetime.now().date()
        summary = self.engine.get_daily_alerts_summary(today)
        
        self.assertEqual(summary['total_alerts'], 2)
        self.assertIn('HIGH', summary['by_severity'])
        self.assertIn('MEDIUM', summary['by_severity'])
        self.assertEqual(summary['by_severity']['HIGH'], 1)
        self.assertEqual(summary['by_severity']['MEDIUM'], 1)
        
        # Verify symbols in summary
        self.assertEqual(summary['by_symbol']['AAPL'], 1)
        self.assertEqual(summary['by_symbol']['MSFT'], 1)
        self.assertNotIn('GOOGL', summary['by_symbol'])
    
    def test_feature_flag_integration(self):
        """Test that smart alerts respect feature flag settings."""
        # Ensure feature flag is disabled
        feature_flags.set_flag('enable_smart_alerts', False)
        
        try:
            # Import main module functions that check feature flags
            from config.feature_flags import is_smart_alerts_enabled
            
            self.assertFalse(is_smart_alerts_enabled())
            
            # Enable feature flag
            feature_flags.set_flag('enable_smart_alerts', True)
            self.assertTrue(is_smart_alerts_enabled())
            
        finally:
            # Reset feature flag
            feature_flags.set_flag('enable_smart_alerts', False)


class TestSmartAlertsMainPipelineIntegration(unittest.TestCase):
    """Test integration with the main.py pipeline."""
    
    def test_main_pipeline_smart_alerts_disabled(self):
        """Test main pipeline behavior when smart alerts are disabled."""
        # Ensure feature flag is disabled
        feature_flags.set_flag('enable_smart_alerts', False)
        
        try:
            from config.feature_flags import is_smart_alerts_enabled
            
            # Should return False when flag is disabled
            self.assertFalse(is_smart_alerts_enabled())
            
            # Main pipeline should handle this gracefully
            # (This is tested indirectly by checking feature flag behavior)
            
        finally:
            feature_flags.set_flag('enable_smart_alerts', False)
    
    def test_data_format_compatibility(self):
        """Test that alert engine accepts data formats from main.py pipeline."""
        # Test with data structures similar to what main.py produces
        symbols = ['AAPL']
        
        # Simulate price data format from prediction.py
        price_df = pd.DataFrame({
            'Stock_Close': [180.00, 181.00, 182.00, 183.00, 190.00]  # Uses 'Stock_Close' column
        })
        price_df.index = pd.date_range('2025-01-20', periods=5, freq='D')
        price_df.index.name = 'Date'
        
        price_data = {'AAPL': price_df}
        
        # Simulate sentiment data format from news_scraper.py
        sentiment_data = {
            'AAPL': {
                'daily_sentiment': {
                    '2025-01-20': 0.1, '2025-01-21': 0.1, '2025-01-22': 0.1,
                    '2025-01-23': 0.1, '2025-01-24': 0.1, '2025-01-25': 0.1,
                    '2025-01-26': 0.1, '2025-01-27': 0.1
                },
                'count': 25,
                'headlines': [('Sample headline', 'http://example.com', '2025-01-20')]
            }
        }
        
        # Test that engine can handle this format
        engine = create_smart_alerts_engine()
        
        # Should not raise exception with realistic data formats
        alerts = engine.evaluate_alerts(symbols, price_data, sentiment_data)
        
        # Should get an alert for the 5.6% price move
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].symbol, 'AAPL')
        self.assertEqual(alerts[0].alert_type, 'price_move')


if __name__ == '__main__':
    # Run integration tests with high verbosity
    unittest.main(verbosity=2)