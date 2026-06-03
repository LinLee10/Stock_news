#!/usr/bin/env python3
"""
Unit tests for F06 Smart Alerts functionality.

Tests alert evaluation logic, configuration loading, and edge cases.
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.monitoring_alerting import (
    SmartAlertsConfig, SmartAlertsEngine, SmartAlert,
    create_smart_alerts_engine
)


class TestSmartAlertsConfig(unittest.TestCase):
    """Test smart alerts configuration loading and merging."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.test_config_dir))
        
        self.test_config_file = self.test_config_dir / "alerts.yaml"
    
    def test_load_default_config_when_file_missing(self):
        """Test that default config is loaded when file doesn't exist."""
        config = SmartAlertsConfig(str(self.test_config_file))
        
        self.assertIsNotNone(config.config)
        self.assertIn('defaults', config.config)
        self.assertEqual(config.config['defaults']['price_move_threshold_percent'], 5.0)
    
    def test_load_config_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
defaults:
  price_move_threshold_percent: 3.0
  sentiment_threshold_change: 0.4
  
symbol_overrides:
  AAPL:
    price_move_threshold_percent: 2.0
"""
        self.test_config_file.write_text(yaml_content)
        
        config = SmartAlertsConfig(str(self.test_config_file))
        
        self.assertEqual(config.config['defaults']['price_move_threshold_percent'], 3.0)
        self.assertEqual(config.config['defaults']['sentiment_threshold_change'], 0.4)
        self.assertIn('AAPL', config.config['symbol_overrides'])
    
    def test_symbol_config_with_overrides(self):
        """Test that symbol-specific overrides merge with defaults."""
        yaml_content = """
defaults:
  price_move_threshold_percent: 5.0
  sentiment_threshold_change: 0.3
  
symbol_overrides:
  TSLA:
    price_move_threshold_percent: 8.0
"""
        self.test_config_file.write_text(yaml_content)
        
        config = SmartAlertsConfig(str(self.test_config_file))
        
        # Test default symbol
        aapl_config = config.get_symbol_config('AAPL')
        self.assertEqual(aapl_config['price_move_threshold_percent'], 5.0)
        self.assertEqual(aapl_config['sentiment_threshold_change'], 0.3)
        
        # Test symbol with overrides
        tsla_config = config.get_symbol_config('TSLA')
        self.assertEqual(tsla_config['price_move_threshold_percent'], 8.0)  # Overridden
        self.assertEqual(tsla_config['sentiment_threshold_change'], 0.3)     # From defaults


class TestSmartAlertsEngine(unittest.TestCase):
    """Test smart alerts evaluation engine."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test config
        self.test_config = SmartAlertsConfig.__new__(SmartAlertsConfig)
        self.test_config.config = {
            'defaults': {
                'price_move_threshold_percent': 5.0,
                'price_move_timeframe_days': 1,
                'sentiment_threshold_change': 0.3,
                'sentiment_comparison_days': 7,
                'earnings_alert_days': [7, 3, 1],
                'severity_thresholds': {
                    'price_move': {'LOW': 3.0, 'MEDIUM': 5.0, 'HIGH': 10.0, 'CRITICAL': 15.0},
                    'sentiment': {'LOW': 0.2, 'MEDIUM': 0.3, 'HIGH': 0.5, 'CRITICAL': 0.7},
                    'earnings': {'LOW': 7, 'MEDIUM': 3, 'HIGH': 1, 'CRITICAL': 0}
                }
            },
            'symbol_overrides': {
                'TSLA': {'price_move_threshold_percent': 8.0}
            },
            'templates': {
                'price_move': {
                    'title': 'Price Alert: {symbol} {direction} {percent:.1f}%',
                    'description': '{symbol} moved {direction} {percent:.1f}%',
                    'guidance': 'Monitor for momentum'
                }
            },
            'security': {'alert_cooldown_minutes': 60}
        }
        
        self.engine = SmartAlertsEngine(self.test_config)
    
    def test_price_alert_threshold_detection(self):
        """Test that price alerts trigger when thresholds are exceeded."""
        # Create test price data with 6% move (above 5% threshold)
        price_df = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
            'Close': [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 106.0]
        })
        
        alerts = self.engine._evaluate_price_alerts('AAPL', price_df, self.test_config.config['defaults'])
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].symbol, 'AAPL')
        self.assertEqual(alerts[0].alert_type, 'price_move')
        self.assertEqual(alerts[0].severity, 'MEDIUM')  # 6% is in MEDIUM range (5-10%)
        self.assertGreater(alerts[0].change_percent, 5.0)
    
    def test_price_alert_no_trigger_below_threshold(self):
        """Test that price alerts don't trigger below threshold."""
        # Create test price data with 3% move (below 5% threshold)
        price_df = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
            'Close': [100.0] * 9 + [103.0]  # 3% move
        })
        
        alerts = self.engine._evaluate_price_alerts('AAPL', price_df, self.test_config.config['defaults'])
        
        self.assertEqual(len(alerts), 0)
    
    def test_price_alert_severity_levels(self):
        """Test that price alert severity is correctly determined."""
        test_cases = [
            (4.0, 'LOW'),     # Above LOW threshold (3%)
            (6.0, 'MEDIUM'),  # Above MEDIUM threshold (5%)
            (12.0, 'HIGH'),   # Above HIGH threshold (10%)
            (18.0, 'CRITICAL') # Above CRITICAL threshold (15%)
        ]
        
        for percent_change, expected_severity in test_cases:
            severity = self.engine._determine_price_severity(percent_change, self.test_config.config['defaults'])
            self.assertEqual(severity, expected_severity, f"Failed for {percent_change}%")
    
    def test_sentiment_alert_threshold_detection(self):
        """Test that sentiment alerts trigger when thresholds are exceeded."""
        # Create sentiment data with significant change
        sentiment_info = {
            'daily_sentiment': {
                '2025-01-01': 0.1, '2025-01-02': 0.1, '2025-01-03': 0.1, 
                '2025-01-04': 0.1, '2025-01-05': 0.1, '2025-01-06': 0.1,
                '2025-01-07': 0.1, '2025-01-08': 0.6  # Big jump from 0.1 to 0.6 = 0.5 change
            }
        }
        
        alerts = self.engine._evaluate_sentiment_alerts('AAPL', sentiment_info, self.test_config.config['defaults'])
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].symbol, 'AAPL')
        self.assertEqual(alerts[0].alert_type, 'sentiment_swing')
        self.assertEqual(alerts[0].severity, 'HIGH')  # 0.5 change is in HIGH range
    
    def test_sentiment_alert_no_trigger_below_threshold(self):
        """Test that sentiment alerts don't trigger below threshold."""
        # Create sentiment data with small change (below 0.3 threshold)
        sentiment_info = {
            'daily_sentiment': {
                '2025-01-01': 0.1, '2025-01-02': 0.1, '2025-01-03': 0.1,
                '2025-01-04': 0.1, '2025-01-05': 0.1, '2025-01-06': 0.1,
                '2025-01-07': 0.1, '2025-01-08': 0.2  # Small change: 0.1
            }
        }
        
        alerts = self.engine._evaluate_sentiment_alerts('AAPL', sentiment_info, self.test_config.config['defaults'])
        
        self.assertEqual(len(alerts), 0)
    
    def test_earnings_alert_proximity_detection(self):
        """Test that earnings alerts trigger at correct time intervals."""
        # Test earnings in 3 days (should trigger MEDIUM alert)
        earnings_date = datetime.now().date() + timedelta(days=3)
        earnings_info = {'date': earnings_date.isoformat()}
        
        alerts = self.engine._evaluate_earnings_alerts('AAPL', earnings_info, self.test_config.config['defaults'])
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].symbol, 'AAPL')
        self.assertEqual(alerts[0].alert_type, 'earnings_proximity')
        self.assertEqual(alerts[0].severity, 'MEDIUM')  # 3 days = MEDIUM
    
    def test_earnings_alert_no_trigger_outside_windows(self):
        """Test that earnings alerts don't trigger outside configured windows."""
        # Test earnings in 5 days (not in [7, 3, 1] alert days)
        earnings_date = datetime.now().date() + timedelta(days=5)
        earnings_info = {'date': earnings_date.isoformat()}
        
        alerts = self.engine._evaluate_earnings_alerts('AAPL', earnings_info, self.test_config.config['defaults'])
        
        self.assertEqual(len(alerts), 0)
    
    def test_cooldown_prevents_duplicate_alerts(self):
        """Test that cooldown prevents spam alerts for same symbol."""
        # Set a recent alert time for AAPL
        self.engine.alert_cooldowns['AAPL'] = datetime.now(timezone.utc) - timedelta(minutes=30)  # 30 mins ago
        
        # Should be on cooldown (60 min cooldown configured)
        self.assertTrue(self.engine._is_symbol_on_cooldown('AAPL'))
        
        # Symbol not on cooldown should return False
        self.assertFalse(self.engine._is_symbol_on_cooldown('MSFT'))
        
        # Old cooldown should not prevent alerts
        self.engine.alert_cooldowns['GOOGL'] = datetime.now(timezone.utc) - timedelta(minutes=90)  # 90 mins ago
        self.assertFalse(self.engine._is_symbol_on_cooldown('GOOGL'))
    
    def test_symbol_specific_overrides(self):
        """Test that symbol-specific configuration overrides work."""
        # TSLA has price_move_threshold_percent: 8.0 in config
        tsla_config = self.test_config.get_symbol_config('TSLA')
        self.assertEqual(tsla_config['price_move_threshold_percent'], 8.0)
        
        # Default symbol should use default threshold
        aapl_config = self.test_config.get_symbol_config('AAPL')
        self.assertEqual(aapl_config['price_move_threshold_percent'], 5.0)
        
        # Test that TSLA needs higher threshold to trigger
        price_df_6_percent = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
            'Close': [100.0] * 9 + [106.0]  # 6% move
        })
        
        # AAPL should trigger (6% > 5% threshold)
        aapl_alerts = self.engine._evaluate_price_alerts('AAPL', price_df_6_percent, aapl_config)
        self.assertEqual(len(aapl_alerts), 1)
        
        # TSLA should NOT trigger (6% < 8% threshold)
        tsla_alerts = self.engine._evaluate_price_alerts('TSLA', price_df_6_percent, tsla_config)
        self.assertEqual(len(tsla_alerts), 0)
    
    def test_evaluate_alerts_integration(self):
        """Test full alert evaluation with multiple symbols and data types."""
        # Create test data
        symbols = ['AAPL', 'TSLA']
        
        # Price data with AAPL having 6% move, TSLA having 4% move
        price_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
                'Close': [100.0] * 9 + [106.0]
            }),
            'TSLA': pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
                'Close': [200.0] * 9 + [208.0]  # 4% move
            })
        }
        
        # Sentiment data with no significant changes
        sentiment_data = {
            'AAPL': {'daily_sentiment': {f'2025-01-0{i}': 0.1 for i in range(1, 9)}},
            'TSLA': {'daily_sentiment': {f'2025-01-0{i}': 0.2 for i in range(1, 9)}}
        }
        
        alerts = self.engine.evaluate_alerts(symbols, price_data, sentiment_data)
        
        # Should get 1 alert: AAPL price move (6% > 5%)
        # TSLA should not trigger (4% < 8% threshold for TSLA)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].symbol, 'AAPL')
        self.assertEqual(alerts[0].alert_type, 'price_move')
    
    def test_alert_creation_with_templates(self):
        """Test that alerts are created with proper templated messages."""
        price_df = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=5, freq='D'),
            'Close': [100.0, 101.0, 102.0, 103.0, 106.0]  # 6% move
        })
        
        alerts = self.engine._evaluate_price_alerts('AAPL', price_df, self.test_config.config['defaults'])
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        
        # Check that template was applied
        self.assertIn('AAPL', alert.title)
        self.assertIn('up', alert.title)  # Positive move
        self.assertIn('6.0%', alert.title)
        
        self.assertIn('AAPL moved up 6.0%', alert.description)
        self.assertEqual(alert.guidance, 'Monitor for momentum')
    
    def test_daily_alerts_summary(self):
        """Test daily alerts summary functionality."""
        # Create some test alerts
        test_date = datetime.now().date()
        
        alert1 = SmartAlert(
            alert_id='test1', symbol='AAPL', alert_type='price_move',
            severity='HIGH', title='Test Alert 1', description='Test',
            timestamp=datetime.combine(test_date, datetime.min.time(), timezone.utc),
            current_value=106.0, previous_value=100.0, change_percent=6.0,
            guidance='Test guidance', metadata={}
        )
        
        alert2 = SmartAlert(
            alert_id='test2', symbol='TSLA', alert_type='sentiment_swing',
            severity='MEDIUM', title='Test Alert 2', description='Test',
            timestamp=datetime.combine(test_date, datetime.min.time(), timezone.utc),
            current_value=0.5, previous_value=0.1, change_percent=None,
            guidance='Test guidance', metadata={}
        )
        
        self.engine.alert_history = [alert1, alert2]
        
        summary = self.engine.get_daily_alerts_summary(test_date)
        
        self.assertEqual(summary['total_alerts'], 2)
        self.assertEqual(summary['by_severity']['HIGH'], 1)
        self.assertEqual(summary['by_severity']['MEDIUM'], 1)
        self.assertEqual(summary['by_type']['price_move'], 1)
        self.assertEqual(summary['by_type']['sentiment_swing'], 1)
        self.assertEqual(summary['by_symbol']['AAPL'], 1)
        self.assertEqual(summary['by_symbol']['TSLA'], 1)


class TestSmartAlertsFactory(unittest.TestCase):
    """Test smart alerts factory function."""
    
    def test_create_smart_alerts_engine(self):
        """Test that factory function creates engine properly."""
        with patch('services.monitoring_alerting.SmartAlertsConfig') as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            
            engine = create_smart_alerts_engine('test_config.yaml')
            
            self.assertIsInstance(engine, SmartAlertsEngine)
            mock_config_class.assert_called_once_with('test_config.yaml')


class TestAlertEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for smart alerts."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config = SmartAlertsConfig.__new__(SmartAlertsConfig)
        self.test_config.config = {
            'defaults': {'price_move_threshold_percent': 5.0, 'price_move_timeframe_days': 1},
            'symbol_overrides': {},
            'templates': {},
            'security': {'alert_cooldown_minutes': 60}
        }
        self.engine = SmartAlertsEngine(self.test_config)
    
    def test_empty_price_data(self):
        """Test handling of empty price data."""
        empty_df = pd.DataFrame()
        alerts = self.engine._evaluate_price_alerts('AAPL', empty_df, self.test_config.config['defaults'])
        self.assertEqual(len(alerts), 0)
        
        # Test None price data
        alerts = self.engine._evaluate_price_alerts('AAPL', None, self.test_config.config['defaults'])
        self.assertEqual(len(alerts), 0)
    
    def test_insufficient_price_data(self):
        """Test handling of insufficient price data for comparison."""
        # Only 1 row, but need 2 for timeframe_days=1
        single_row_df = pd.DataFrame({'Close': [100.0], 'Date': [pd.Timestamp.now()]})
        alerts = self.engine._evaluate_price_alerts('AAPL', single_row_df, self.test_config.config['defaults'])
        self.assertEqual(len(alerts), 0)
    
    def test_empty_sentiment_data(self):
        """Test handling of empty sentiment data."""
        empty_sentiment = {}
        alerts = self.engine._evaluate_sentiment_alerts('AAPL', empty_sentiment, self.test_config.config['defaults'])
        self.assertEqual(len(alerts), 0)
        
        # Test missing daily_sentiment key
        invalid_sentiment = {'count': 10}
        alerts = self.engine._evaluate_sentiment_alerts('AAPL', invalid_sentiment, self.test_config.config['defaults'])
        self.assertEqual(len(alerts), 0)
    
    def test_invalid_earnings_date(self):
        """Test handling of invalid earnings date."""
        invalid_earnings = {'date': 'not-a-date'}
        alerts = self.engine._evaluate_earnings_alerts('AAPL', invalid_earnings, self.test_config.config['defaults'])
        self.assertEqual(len(alerts), 0)
    
    def test_evaluation_with_exceptions(self):
        """Test that exceptions in individual symbol evaluation don't break overall process."""
        symbols = ['AAPL', 'INVALID', 'MSFT']
        
        # Valid price data for AAPL and MSFT
        price_data = {
            'AAPL': pd.DataFrame({'Date': pd.date_range('2025-01-01', periods=5), 'Close': [100, 101, 102, 103, 106]}),
            'MSFT': pd.DataFrame({'Date': pd.date_range('2025-01-01', periods=5), 'Close': [200, 201, 202, 203, 210]})
        }
        
        sentiment_data = {
            'AAPL': {'daily_sentiment': {}},
            'MSFT': {'daily_sentiment': {}}
        }
        
        # Mock price evaluation to raise exception for INVALID symbol
        original_method = self.engine._evaluate_price_alerts
        def mock_evaluate_price_alerts(symbol, price_df, config):
            if symbol == 'INVALID':
                raise ValueError("Invalid symbol")
            return original_method(symbol, price_df, config)
        
        with patch.object(self.engine, '_evaluate_price_alerts', side_effect=mock_evaluate_price_alerts):
            alerts = self.engine.evaluate_alerts(symbols, price_data, sentiment_data)
            
            # Should still get alerts for valid symbols
            self.assertGreater(len(alerts), 0)
            alert_symbols = [alert.symbol for alert in alerts]
            self.assertIn('AAPL', alert_symbols)
            self.assertNotIn('INVALID', alert_symbols)


if __name__ == '__main__':
    unittest.main(verbosity=2)