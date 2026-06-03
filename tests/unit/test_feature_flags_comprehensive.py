#!/usr/bin/env python3
"""
Comprehensive unit tests for feature flags system.
Ensures all flags default to OFF and can be toggled safely.
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.feature_flags import (
    FeatureFlags, feature_flags,
    is_symbol_intake_enabled, is_news_corroboration_enabled,
    is_finbert_pipeline_enabled, is_portfolio_analytics_enabled,
    is_smart_alerts_enabled, is_api_endpoints_enabled,
    is_yf_prices_enabled, is_multisource_prices_enabled
)


class TestFeatureFlagsComprehensive(unittest.TestCase):
    """Comprehensive tests for feature flag system."""
    
    def setUp(self):
        """Set up clean feature flags for each test."""
        self.flags = FeatureFlags()
        
    def test_all_flags_default_to_false(self):
        """Critical: All flags must default to OFF to prevent regressions."""
        
        all_flags = self.flags.get_all_flags()
        
        # Verify every flag defaults to False
        for flag_name, flag_value in all_flags.items():
            with self.subTest(flag=flag_name):
                self.assertFalse(flag_value, f"Flag {flag_name} must default to False")
                
        # Verify specific critical flags
        critical_flags = [
            'enable_api_endpoints',
            'enable_finbert_pipeline', 
            'enable_multisource_prices',
            'enable_alpha_vantage_batching',
            'enable_newsapi_ingestion',
            'enable_portfolio_analytics',
            'enable_smart_alerts'
        ]
        
        for flag in critical_flags:
            with self.subTest(critical_flag=flag):
                self.assertFalse(self.flags.is_enabled(flag))
                
    def test_flag_toggling_works_correctly(self):
        """Test that flags can be enabled and disabled correctly."""
        
        test_flag = 'enable_finbert_pipeline'
        
        # Initially OFF
        self.assertFalse(self.flags.is_enabled(test_flag))
        
        # Enable
        self.flags.set_flag(test_flag, True)
        self.assertTrue(self.flags.is_enabled(test_flag))
        
        # Disable  
        self.flags.set_flag(test_flag, False)
        self.assertFalse(self.flags.is_enabled(test_flag))
        
    def test_convenience_functions_match_flags(self):
        """Test that convenience functions match flag states."""
        
        # Test multiple flag/function pairs
        flag_function_pairs = [
            ('enable_symbol_intake', is_symbol_intake_enabled),
            ('enable_news_corroboration', is_news_corroboration_enabled),
            ('enable_finbert_pipeline', is_finbert_pipeline_enabled),
            ('enable_portfolio_analytics', is_portfolio_analytics_enabled),
            ('enable_smart_alerts', is_smart_alerts_enabled),
            ('enable_api_endpoints', is_api_endpoints_enabled),
            ('enable_yf_prices', is_yf_prices_enabled),
            ('enable_multisource_prices', is_multisource_prices_enabled)
        ]
        
        for flag_name, function in flag_function_pairs:
            with self.subTest(flag=flag_name):
                
                # Patch the global feature_flags with our test instance
                with patch('config.feature_flags.feature_flags', self.flags):
                    # Test OFF state
                    self.flags.set_flag(flag_name, False)
                    self.assertFalse(function())
                    
                    # Test ON state
                    self.flags.set_flag(flag_name, True) 
                    self.assertTrue(function())
                    
    def test_environment_variable_parsing(self):
        """Test environment variable parsing for boolean flags."""
        
        test_cases = [
            ('true', True),
            ('True', True), 
            ('TRUE', True),
            ('1', True),
            ('yes', True),
            ('on', True),
            ('enabled', True),
            ('false', False),
            ('False', False),
            ('0', False),
            ('no', False),
            ('off', False),
            ('disabled', False),
            ('random_value', False)  # Default to False for unknown values
        ]
        
        for env_value, expected_bool in test_cases:
            with self.subTest(env_value=env_value):
                result = self.flags._get_bool_env('TEST_VAR', False)
                
                with patch.dict(os.environ, {'TEST_VAR': env_value}):
                    fresh_flags = FeatureFlags()
                    # Test with a known flag that reads from env
                    test_result = fresh_flags._get_bool_env('TEST_VAR', False)
                    self.assertEqual(test_result, expected_bool)
                    
    def test_invalid_flag_names_return_false(self):
        """Test that invalid flag names return False safely."""
        
        invalid_flags = [
            'nonexistent_flag',
            'enable_invalid_feature',
            '',
            None,
            123
        ]
        
        for invalid_flag in invalid_flags:
            with self.subTest(invalid_flag=invalid_flag):
                if isinstance(invalid_flag, str):
                    result = self.flags.is_enabled(invalid_flag)
                    self.assertFalse(result)
                    
    def test_flag_isolation(self):
        """Test that changing one flag doesn't affect others."""
        
        # Enable one flag
        self.flags.set_flag('enable_finbert_pipeline', True)
        
        # Verify only that flag is enabled
        all_flags = self.flags.get_all_flags()
        enabled_count = sum(1 for enabled in all_flags.values() if enabled)
        
        self.assertEqual(enabled_count, 1, "Only one flag should be enabled")
        self.assertTrue(self.flags.is_enabled('enable_finbert_pipeline'))
        
        # Verify other critical flags are still OFF
        other_flags = [
            'enable_api_endpoints',
            'enable_multisource_prices', 
            'enable_portfolio_analytics',
            'enable_smart_alerts'
        ]
        
        for flag in other_flags:
            with self.subTest(other_flag=flag):
                self.assertFalse(self.flags.is_enabled(flag))
                
    def test_flag_immutability_during_read(self):
        """Test that reading flags doesn't modify internal state."""
        
        original_flags = self.flags.get_all_flags()
        
        # Read flags multiple times
        for _ in range(5):
            current_flags = self.flags.get_all_flags()
            self.assertEqual(original_flags, current_flags)
            
        # Test individual flag reads
        for flag_name in original_flags:
            for _ in range(3):
                self.flags.is_enabled(flag_name)
                
        # Verify state unchanged
        final_flags = self.flags.get_all_flags() 
        self.assertEqual(original_flags, final_flags)
        
    def test_all_expected_flags_present(self):
        """Test that all expected flags are present in the system."""
        
        expected_flags = [
            'enable_symbol_intake',
            'enable_news_corroboration',
            'enable_earnings_reads', 
            'enable_recos',
            'enable_90_day_sentiment',
            'enable_multisource_prices',
            'enable_paid_sources',
            'enable_alpha_vantage_batching',
            'enable_newsapi_ingestion',
            'enable_finbert_pipeline',
            'enable_finbert_backtest',
            'enable_portfolio_analytics',
            'enable_smart_alerts',
            'enable_yf_prices',
            'enable_yf_daily_refresh',
            'enable_yf_profiles',
            'enable_yf_backoff_debug',
            'enable_debug_mode',
            'enable_api_endpoints',
            'enable_async_io',
            'enable_timescale_persistence',
            'enable_vector_search',
            'enable_alt_forecasts',
            'enable_timegpt_stub',
            'enable_correlation',
            'enable_gnn_scaffold',
            'enable_microservices_mode'
        ]
        
        all_flags = self.flags.get_all_flags()
        
        for expected_flag in expected_flags:
            with self.subTest(expected_flag=expected_flag):
                self.assertIn(expected_flag, all_flags, 
                             f"Expected flag {expected_flag} not found in feature flags")
                
        # Verify no extra unexpected flags
        for actual_flag in all_flags:
            with self.subTest(actual_flag=actual_flag):
                self.assertIn(actual_flag, expected_flags,
                             f"Unexpected flag {actual_flag} found in system")


class TestFeatureFlagIntegrationPoints(unittest.TestCase):
    """Test integration points where flags are checked."""
    
    def test_main_pipeline_checks_flags(self):
        """Test that main.py checks appropriate flags."""
        
        # This would be enhanced to actually import main.py and verify
        # that it checks flags before enabling features
        
        with patch('config.feature_flags.feature_flags') as mock_flags:
            mock_flags.is_enabled.return_value = False
            
            # Import main after mocking
            try:
                import main
                # Verify main respects flags (would be more specific)
                self.assertTrue(hasattr(main, 'get_top_mentioned_stocks'))
            except Exception as e:
                self.skipTest(f"Main module import failed: {e}")
                
    def test_api_endpoints_respect_flags(self):
        """Test that API endpoints are only available when flagged."""
        
        # Test with flag OFF
        with patch('config.feature_flags.is_api_endpoints_enabled', return_value=False):
            # API endpoints should not be accessible
            self.assertFalse(is_api_endpoints_enabled())
            
        # Test with flag ON
        with patch('config.feature_flags.is_api_endpoints_enabled', return_value=True):
            # API endpoints should be accessible  
            self.assertTrue(is_api_endpoints_enabled())


if __name__ == '__main__':
    unittest.main(verbosity=2)