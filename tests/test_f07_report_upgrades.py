#!/usr/bin/env python3
"""
F07 Report & Chart Upgrades Tests

Tests for ensuring backward compatibility and enhanced features work correctly.
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

from email_report import send_report
from charts import create_collage, create_enhanced_collage
from config.feature_flags import feature_flags


class TestF07ReportUpgrades(unittest.TestCase):
    """Test F07 report upgrades with backward compatibility checks."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.test_dir))
        
        # Save original feature flag states
        self.original_flags = {}
        for flag in ['enable_portfolio_analytics', 'enable_smart_alerts', 'enable_finbert_pipeline']:
            self.original_flags[flag] = feature_flags.is_enabled(flag)
        
        # Disable all features for baseline testing
        self._disable_all_features()
    
    def tearDown(self):
        """Reset feature flags to original state."""
        for flag, state in self.original_flags.items():
            feature_flags.set_flag(flag, state)
    
    def _disable_all_features(self):
        """Disable all F04/F05/F06 features to test baseline."""
        feature_flags.set_flag('enable_portfolio_analytics', False)
        feature_flags.set_flag('enable_smart_alerts', False)
        feature_flags.set_flag('enable_finbert_pipeline', False)
    
    def _enable_all_features(self):
        """Enable all features to test enhanced mode."""
        feature_flags.set_flag('enable_portfolio_analytics', True)
        feature_flags.set_flag('enable_smart_alerts', True)
        feature_flags.set_flag('enable_finbert_pipeline', True)
    
    def test_ac1_legacy_html_structure_unchanged(self):
        """AC1: With all flags off, HTML matches current legacy structure."""
        # Create test data
        watchlist = ['AAPL', 'MSFT']
        portfolio = ['AAPL']
        
        head7 = {
            'AAPL': {'count': 5, 'headlines': [('Test headline', 'http://test.com', '2025-01-01')]},
            'MSFT': {'count': 3, 'headlines': []}
        }
        head30 = {
            'AAPL': {'daily_sentiment': {'2025-01-01': 0.1}, 'count': 10},
            'MSFT': {'daily_sentiment': {'2025-01-01': 0.2}, 'count': 8}
        }
        preds = {
            'AAPL': {'predictions': [150.0, 151.0, 152.0], 'confidence': 0.8, 'red_flag': False}
        }
        
        # Generate baseline report (all flags OFF)
        baseline_report = self.test_dir / "baseline_report.html"
        
        with patch('email_report.smtplib.SMTP'):
            send_report(
                watchlist=watchlist,
                portfolio=portfolio,
                head7=head7,
                head30=head30,
                preds=preds,
                out_path=str(baseline_report),
                finbert_results=None,
                portfolio_analytics=None,
                smart_alerts=None
            )
        
        # Verify baseline report exists and contains expected legacy sections
        self.assertTrue(baseline_report.exists())
        content = baseline_report.read_text()
        
        # Verify core legacy sections are present
        self.assertIn('30-Day Sentiment — Portfolio', content)
        self.assertIn('30-Day Sentiment — Watchlist', content) 
        self.assertIn('7-Day Mention Leaders', content)
        
        # Verify F04/F05/F06/F07 sections are NOT present
        self.assertNotIn('FinBERT AI Sentiment Analysis', content)
        self.assertNotIn('Portfolio Analytics & Benchmarks', content)
        self.assertNotIn('Smart Alerts', content)
        self.assertNotIn('Enhanced Benchmark Analysis', content)
        self.assertNotIn('Enhanced Alert Dashboard', content)
        self.assertNotIn('Enhanced FinBERT AI Analysis', content)
        self.assertNotIn('Report Generation Metrics', content)
        
        # Verify HTML structure integrity
        self.assertIn('<html><body>', content)
        self.assertIn('</body></html>', content)
        self.assertIn('Stock News Forecast Report', content)
    
    def test_ac2_enhanced_sections_append_below_legacy(self):
        """AC2: With flags on, enhanced sections append below legacy blocks."""
        # Enable all features
        self._enable_all_features()
        
        # Create enhanced test data
        watchlist = ['AAPL', 'MSFT']
        portfolio = ['AAPL']
        
        head7 = {
            'AAPL': {'count': 5, 'headlines': [('Test headline', 'http://test.com', '2025-01-01')]},
            'MSFT': {'count': 3, 'headlines': []}
        }
        head30 = {
            'AAPL': {'daily_sentiment': {'2025-01-01': 0.1}, 'count': 10},
            'MSFT': {'daily_sentiment': {'2025-01-01': 0.2}, 'count': 8}
        }
        preds = {
            'AAPL': {'predictions': [150.0, 151.0, 152.0], 'confidence': 0.8, 'red_flag': False}
        }
        
        # Create mock enhanced data
        from services.monitoring_alerting import SmartAlert
        smart_alerts = [
            SmartAlert(
                alert_id='test1', symbol='AAPL', alert_type='price_move',
                severity='HIGH', title='Test Alert', description='Test description',
                timestamp=datetime.now(timezone.utc), current_value=150.0,
                previous_value=140.0, change_percent=7.1, guidance='Test guidance',
                metadata={}
            )
        ]
        
        portfolio_analytics = {
            'sector_allocation': {'Technology': 0.6, 'Healthcare': 0.4},
            'beta_stats': {'AAPL': {'beta': 1.2, 'r_squared': 0.8, 'ticker_volatility': 25.0}},
            'benchmark_performance': {
                '^GSPC': {'1M': 2.5, '3M': 5.1, '1Y': 12.3},
                '^IXIC': {'1M': 3.2, '3M': 6.8, '1Y': 15.7}
            },
            'portfolio_performance': {'1M': 4.1, '3M': 7.2, '1Y': 18.5}
        }
        
        finbert_results = {
            'AAPL': {
                'recommendation': MagicMock(action=MagicMock(value='buy'), 
                                          confidence=85.0, conviction_score=7.5, sentiment_score=0.3)
            }
        }
        
        # Generate enhanced report (all flags ON)
        enhanced_report = self.test_dir / "enhanced_report.html"
        
        with patch('email_report.smtplib.SMTP'):
            send_report(
                watchlist=watchlist,
                portfolio=portfolio,
                head7=head7,
                head30=head30,
                preds=preds,
                out_path=str(enhanced_report),
                finbert_results=finbert_results,
                portfolio_analytics=portfolio_analytics,
                smart_alerts=smart_alerts
            )
        
        # Verify enhanced report exists
        self.assertTrue(enhanced_report.exists())
        content = enhanced_report.read_text()
        
        # Verify legacy sections are still present (backward compatibility)
        self.assertIn('30-Day Sentiment — Portfolio', content)
        self.assertIn('30-Day Sentiment — Watchlist', content)
        self.assertIn('7-Day Mention Leaders', content)
        
        # Verify original F04/F05/F06 sections are present
        self.assertIn('FinBERT AI Sentiment Analysis', content)
        self.assertIn('Portfolio Analytics & Benchmarks', content)
        self.assertIn('Smart Alerts', content)
        
        # Verify F07 enhanced sections are present
        self.assertIn('Enhanced Benchmark Analysis', content)
        self.assertIn('Enhanced Alert Dashboard', content)
        self.assertIn('Enhanced FinBERT AI Analysis', content)
        self.assertIn('Report Generation Metrics', content)
        
        # Verify section ordering (legacy sections come first)
        legacy_pos = content.find('30-Day Sentiment — Portfolio')
        enhanced_pos = content.find('Enhanced Benchmark Analysis')
        footer_pos = content.find('Report Generation Metrics')
        
        self.assertLess(legacy_pos, enhanced_pos, "Legacy sections should come before enhanced sections")
        self.assertLess(enhanced_pos, footer_pos, "Enhanced sections should come before footer")
    
    def test_legacy_chart_creation_unchanged(self):
        """Test that legacy create_collage function works unchanged."""
        # Create test data
        tickers = ['AAPL', 'MSFT']
        price_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
                'Close': [150.0 + i for i in range(10)]
            }),
            'MSFT': pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
                'Close': [300.0 + i for i in range(10)]
            })
        }
        forecast_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-11', periods=3, freq='D'),
                'Forecast_Close': [160.0, 161.0, 162.0]
            }),
            'MSFT': pd.DataFrame({
                'Date': pd.date_range('2025-01-11', periods=3, freq='D'),
                'Forecast_Close': [310.0, 311.0, 312.0]
            })
        }
        
        # Test legacy function
        chart_path = self.test_dir / "legacy_chart.png"
        result = create_collage(
            tickers, price_data, forecast_data,
            "Legacy Test Chart", str(chart_path)
        )
        
        # Verify chart was created
        self.assertTrue(Path(chart_path).exists())
        self.assertEqual(result, str(chart_path))
    
    def test_enhanced_chart_creation_with_features(self):
        """Test enhanced chart creation with optional panes."""
        # Enable features
        self._enable_all_features()
        
        # Create test data
        tickers = ['AAPL', 'MSFT']
        price_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
                'Close': [150.0 + i for i in range(10)]
            }),
            'MSFT': pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=10, freq='D'),
                'Close': [300.0 + i for i in range(10)]
            })
        }
        forecast_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-11', periods=3, freq='D'),
                'Forecast_Close': [160.0, 161.0, 162.0]
            }),
            'MSFT': pd.DataFrame({
                'Date': pd.date_range('2025-01-11', periods=3, freq='D'),
                'Forecast_Close': [310.0, 311.0, 312.0]
            })
        }
        
        # Enhanced data
        from services.monitoring_alerting import SmartAlert
        smart_alerts = [
            SmartAlert(
                alert_id='test1', symbol='AAPL', alert_type='price_move',
                severity='HIGH', title='Test Alert', description='Test',
                timestamp=datetime.now(timezone.utc), current_value=150.0,
                previous_value=140.0, change_percent=7.1, guidance='Test',
                metadata={}
            )
        ]
        
        portfolio_analytics = {
            'sector_allocation': {'Technology': 0.7, 'Healthcare': 0.3},
            'beta_stats': {'AAPL': {'beta': 1.1, 'r_squared': 0.85, 'ticker_volatility': 22.0}}
        }
        
        benchmark_data = {
            '^GSPC': {'1M': 2.1, '3M': 4.8, '1Y': 11.2},
            '^IXIC': {'1M': 3.5, '3M': 7.1, '1Y': 16.8}
        }
        
        # Test enhanced function
        chart_path = self.test_dir / "enhanced_chart.png"
        result = create_enhanced_collage(
            tickers, price_data, forecast_data,
            "Enhanced Test Chart", str(chart_path),
            portfolio_analytics=portfolio_analytics,
            benchmark_data=benchmark_data,
            smart_alerts=smart_alerts
        )
        
        # Verify enhanced chart was created
        self.assertTrue(Path(chart_path).exists())
        self.assertEqual(result, str(chart_path))
    
    def test_enhanced_chart_graceful_fallback(self):
        """Test enhanced chart gracefully falls back when features disabled."""
        # Disable all features
        self._disable_all_features()
        
        # Create test data
        tickers = ['AAPL']
        price_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-01', periods=5, freq='D'),
                'Close': [150.0, 151.0, 152.0, 153.0, 154.0]
            })
        }
        forecast_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2025-01-06', periods=3, freq='D'),
                'Forecast_Close': [155.0, 156.0, 157.0]
            })
        }
        
        # Test enhanced function with features disabled (should work like legacy)
        chart_path = self.test_dir / "fallback_chart.png"
        result = create_enhanced_collage(
            tickers, price_data, forecast_data,
            "Fallback Test Chart", str(chart_path),
            portfolio_analytics={'test': 'data'},  # Provided but should be ignored
            smart_alerts=[],  # Provided but should be ignored
        )
        
        # Verify chart was created (should look like legacy)
        self.assertTrue(Path(chart_path).exists())
        self.assertEqual(result, str(chart_path))
    
    def test_renderer_helper_functions(self):
        """Test F07 renderer helper functions work correctly."""
        from email_report import (render_enhanced_benchmark_section,
                                 render_enhanced_alerts_section,
                                 render_enhanced_finbert_section,
                                 render_performance_footer)
        
        # Test benchmark renderer
        benchmark_data = {'^GSPC': {'1M': 2.5, '3M': 5.0, '1Y': 12.0}}
        portfolio_analytics = {'portfolio_performance': {'1M': 3.0, '3M': 6.0, '1Y': 15.0}}
        
        benchmark_html = render_enhanced_benchmark_section(benchmark_data, portfolio_analytics)
        self.assertIn('Enhanced Benchmark Analysis', benchmark_html)
        self.assertIn('S&P 500', benchmark_html)
        self.assertIn('Your Portfolio', benchmark_html)
        
        # Test alerts renderer
        from services.monitoring_alerting import SmartAlert
        alerts = [
            SmartAlert(
                alert_id='test', symbol='AAPL', alert_type='price_move',
                severity='HIGH', title='Test Alert', description='Test description',
                timestamp=datetime.now(timezone.utc), current_value=150.0,
                previous_value=140.0, change_percent=7.1, guidance='Test guidance',
                metadata={}
            )
        ]
        
        alerts_html = render_enhanced_alerts_section(alerts)
        self.assertIn('Enhanced Alert Dashboard', alerts_html)
        self.assertIn('AAPL', alerts_html)
        self.assertIn('HIGH', alerts_html)
        
        # Test FinBERT renderer  
        finbert_data = {
            'AAPL': {
                'recommendation': MagicMock(
                    action=MagicMock(value='buy'),
                    confidence=85.0,
                    conviction_score=7.5
                )
            }
        }
        
        finbert_html = render_enhanced_finbert_section(finbert_data)
        self.assertIn('Enhanced FinBERT AI Analysis', finbert_html)
        self.assertIn('Portfolio Sentiment Overview', finbert_html)
        
        # Test performance footer
        footer_html = render_performance_footer()
        self.assertIn('Report Generation Metrics', footer_html)
        self.assertIn('F04 FinBERT', footer_html)
        self.assertIn('F07 Enhanced', footer_html)
    
    def test_css_unchanged_for_legacy_parts(self):
        """Test that CSS styling for legacy parts remains unchanged."""
        # Generate baseline report 
        watchlist = ['AAPL']
        portfolio = ['AAPL']
        head7 = {'AAPL': {'count': 1, 'headlines': []}}
        head30 = {'AAPL': {'daily_sentiment': {}, 'count': 1}}
        preds = {'AAPL': {'predictions': [], 'confidence': 0.8, 'red_flag': False}}
        
        report_path = self.test_dir / "css_test.html"
        
        with patch('email_report.smtplib.SMTP'):
            send_report(
                watchlist=watchlist, portfolio=portfolio,
                head7=head7, head30=head30, preds=preds,
                out_path=str(report_path)
            )
        
        content = report_path.read_text()
        
        # Verify legacy CSS patterns are preserved
        self.assertIn("font-family: Arial, sans-serif", content)
        self.assertIn("max-width:600px", content)
        self.assertIn("border-collapse: collapse", content)
        self.assertIn("background-color:#f0f0f0", content)
        
        # Verify no conflicting F07 styles in legacy sections
        legacy_section = content.split('30-Day Sentiment — Portfolio')[1].split('</table>')[0]
        self.assertNotIn("linear-gradient", legacy_section)
        self.assertNotIn("border-radius:", legacy_section)
    
    def test_stable_html_ids_and_ordering(self):
        """Test idempotency: stable ordering and IDs in HTML output."""
        # Create identical test data
        test_data = {
            'watchlist': ['AAPL', 'MSFT'],
            'portfolio': ['AAPL'],
            'head7': {'AAPL': {'count': 5, 'headlines': []}, 'MSFT': {'count': 3, 'headlines': []}},
            'head30': {'AAPL': {'daily_sentiment': {}}, 'MSFT': {'daily_sentiment': {}}},
            'preds': {'AAPL': {'predictions': [], 'confidence': 0.8, 'red_flag': False}}
        }
        
        reports = []
        for i in range(3):
            report_path = self.test_dir / f"stable_test_{i}.html"
            
            with patch('email_report.smtplib.SMTP'):
                send_report(out_path=str(report_path), **test_data)
            
            reports.append(report_path.read_text())
        
        # Remove timestamps which naturally vary
        clean_reports = []
        for content in reports:
            # Remove timestamp from title
            import re
            cleaned = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC', 'TIMESTAMP', content)
            clean_reports.append(cleaned)
        
        # Verify all reports are identical (stable output)
        self.assertEqual(clean_reports[0], clean_reports[1])
        self.assertEqual(clean_reports[1], clean_reports[2])


if __name__ == '__main__':
    unittest.main(verbosity=2)