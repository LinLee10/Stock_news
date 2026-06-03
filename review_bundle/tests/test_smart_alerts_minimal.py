#!/usr/bin/env python3
"""
Minimal unit tests for F06 Smart Alerts functionality.
Tests only the core smart alerts logic without external dependencies.
"""

import sys
import unittest
import tempfile
import shutil
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSmartAlertsMinimal(unittest.TestCase):
    """Minimal tests for smart alerts without external dependencies."""
    
    def test_import_smart_alerts_functions(self):
        """Test that we can import the main smart alerts components."""
        try:
            # Test imports
            from services.monitoring_alerting import SmartAlert, SmartAlertsConfig, SmartAlertsEngine
            
            # Test that classes can be instantiated
            alert = SmartAlert(
                alert_id='test',
                symbol='AAPL',
                alert_type='price_move',
                severity='HIGH',
                title='Test Alert',
                description='Test Description',
                timestamp=datetime.now(timezone.utc),
                current_value=100.0,
                previous_value=95.0,
                change_percent=5.26,
                guidance='Test guidance',
                metadata={}
            )
            
            self.assertEqual(alert.symbol, 'AAPL')
            self.assertEqual(alert.alert_type, 'price_move')
            self.assertEqual(alert.severity, 'HIGH')
            
        except ImportError as e:
            self.fail(f"Failed to import smart alerts components: {e}")
    
    def test_feature_flag_integration(self):
        """Test that smart alerts feature flag works."""
        try:
            from config.feature_flags import feature_flags, is_smart_alerts_enabled
            
            # Test default state
            original_state = is_smart_alerts_enabled()
            
            # Test enabling
            feature_flags.set_flag('enable_smart_alerts', True)
            self.assertTrue(is_smart_alerts_enabled())
            
            # Test disabling
            feature_flags.set_flag('enable_smart_alerts', False)
            self.assertFalse(is_smart_alerts_enabled())
            
            # Restore original state
            feature_flags.set_flag('enable_smart_alerts', original_state)
            
        except ImportError as e:
            self.fail(f"Failed to test feature flag: {e}")
    
    def test_yaml_config_loading(self):
        """Test YAML configuration loading."""
        test_config_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(test_config_dir))
        
        test_config_file = test_config_dir / "alerts.yaml"
        
        # Create test YAML content
        yaml_content = """
defaults:
  price_move_threshold_percent: 5.0
  sentiment_threshold_change: 0.3

symbol_overrides:
  AAPL:
    price_move_threshold_percent: 3.0
"""
        test_config_file.write_text(yaml_content)
        
        try:
            from services.monitoring_alerting import SmartAlertsConfig
            
            config = SmartAlertsConfig(str(test_config_file))
            
            # Test that config was loaded
            self.assertIsNotNone(config.config)
            self.assertEqual(config.config['defaults']['price_move_threshold_percent'], 5.0)
            
            # Test symbol-specific config
            aapl_config = config.get_symbol_config('AAPL')
            self.assertEqual(aapl_config['price_move_threshold_percent'], 3.0)
            
            # Test default symbol
            msft_config = config.get_symbol_config('MSFT')
            self.assertEqual(msft_config['price_move_threshold_percent'], 5.0)
            
        except Exception as e:
            self.fail(f"Failed to test YAML config loading: {e}")
    
    def test_price_change_calculation(self):
        """Test price change calculation logic."""
        try:
            from services.monitoring_alerting import SmartAlertsConfig, SmartAlertsEngine
            
            # Create minimal config
            test_config = SmartAlertsConfig.__new__(SmartAlertsConfig)
            test_config.config = {
                'defaults': {
                    'price_move_threshold_percent': 5.0,
                    'price_move_timeframe_days': 1,
                    'severity_thresholds': {
                        'price_move': {'LOW': 3.0, 'MEDIUM': 5.0, 'HIGH': 10.0, 'CRITICAL': 15.0}
                    }
                },
                'symbol_overrides': {},
                'templates': {
                    'price_move': {
                        'title': 'Price Alert: {symbol}',
                        'description': 'Price moved',
                        'guidance': 'Monitor'
                    }
                }
            }
            
            engine = SmartAlertsEngine(test_config)
            
            # Test price data with significant move
            price_df = pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=5, freq='D'),
                'Close': [100.0, 101.0, 102.0, 103.0, 107.0]  # 7% move from 100 to 107
            })
            
            alerts = engine._evaluate_price_alerts('AAPL', price_df, test_config.config['defaults'])
            
            # Should trigger one alert (7% > 5% threshold)
            self.assertEqual(len(alerts), 1)
            alert = alerts[0]
            self.assertEqual(alert.symbol, 'AAPL')
            self.assertEqual(alert.alert_type, 'price_move')
            self.assertGreater(alert.change_percent, 5.0)
            
        except Exception as e:
            self.fail(f"Failed to test price change calculation: {e}")
    
    def test_sentiment_change_calculation(self):
        """Test sentiment change calculation logic."""
        try:
            from services.monitoring_alerting import SmartAlertsConfig, SmartAlertsEngine
            
            # Create minimal config
            test_config = SmartAlertsConfig.__new__(SmartAlertsConfig)
            test_config.config = {
                'defaults': {
                    'sentiment_threshold_change': 0.3,
                    'sentiment_comparison_days': 7,
                    'severity_thresholds': {
                        'sentiment': {'LOW': 0.2, 'MEDIUM': 0.3, 'HIGH': 0.5, 'CRITICAL': 0.7}
                    }
                },
                'symbol_overrides': {},
                'templates': {
                    'sentiment': {
                        'title': 'Sentiment Alert: {symbol}',
                        'description': 'Sentiment changed',
                        'guidance': 'Review'
                    }
                }
            }
            
            engine = SmartAlertsEngine(test_config)
            
            # Test sentiment data with significant change
            sentiment_info = {
                'daily_sentiment': {
                    '2025-01-01': 0.0, '2025-01-02': 0.0, '2025-01-03': 0.0,
                    '2025-01-04': 0.0, '2025-01-05': 0.0, '2025-01-06': 0.0,
                    '2025-01-07': 0.0, '2025-01-08': 0.4  # Jump to 0.4 (0.4 change from 0.0 avg)
                }
            }
            
            alerts = engine._evaluate_sentiment_alerts('AAPL', sentiment_info, test_config.config['defaults'])
            
            # Should trigger one alert (0.4 change > 0.3 threshold)
            self.assertEqual(len(alerts), 1)
            alert = alerts[0]
            self.assertEqual(alert.symbol, 'AAPL')
            self.assertEqual(alert.alert_type, 'sentiment_swing')
            
        except Exception as e:
            self.fail(f"Failed to test sentiment change calculation: {e}")
    
    def test_severity_determination(self):
        """Test severity level determination."""
        try:
            from services.monitoring_alerting import SmartAlertsConfig, SmartAlertsEngine
            
            test_config = SmartAlertsConfig.__new__(SmartAlertsConfig)
            test_config.config = {
                'defaults': {
                    'severity_thresholds': {
                        'price_move': {'LOW': 3.0, 'MEDIUM': 5.0, 'HIGH': 10.0, 'CRITICAL': 15.0}
                    }
                }
            }
            
            engine = SmartAlertsEngine(test_config)
            
            # Test different severity levels
            test_cases = [
                (4.0, 'LOW'),      # 4% = LOW
                (7.0, 'MEDIUM'),   # 7% = MEDIUM
                (12.0, 'HIGH'),    # 12% = HIGH
                (20.0, 'CRITICAL') # 20% = CRITICAL
            ]
            
            for percent_change, expected_severity in test_cases:
                severity = engine._determine_price_severity(percent_change, test_config.config['defaults'])
                self.assertEqual(
                    severity, 
                    expected_severity, 
                    f"Expected {expected_severity} for {percent_change}%, got {severity}"
                )
                
        except Exception as e:
            self.fail(f"Failed to test severity determination: {e}")
    
    def test_cooldown_logic(self):
        """Test alert cooldown logic."""
        try:
            from services.monitoring_alerting import SmartAlertsConfig, SmartAlertsEngine
            
            test_config = SmartAlertsConfig.__new__(SmartAlertsConfig)
            test_config.config = {
                'security': {'alert_cooldown_minutes': 60}
            }
            
            engine = SmartAlertsEngine(test_config)
            
            # Test no cooldown initially
            self.assertFalse(engine._is_symbol_on_cooldown('AAPL'))
            
            # Set recent cooldown
            engine.alert_cooldowns['AAPL'] = datetime.now(timezone.utc) - timedelta(minutes=30)
            self.assertTrue(engine._is_symbol_on_cooldown('AAPL'))
            
            # Set old cooldown (should not be on cooldown)
            engine.alert_cooldowns['MSFT'] = datetime.now(timezone.utc) - timedelta(minutes=90)
            self.assertFalse(engine._is_symbol_on_cooldown('MSFT'))
            
        except Exception as e:
            self.fail(f"Failed to test cooldown logic: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)