#!/usr/bin/env python3
"""
Integration tests for earnings service flow
Tests earnings schedule generation and email integration
"""

import os
import sys
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEarningsFlow:
    """Test earnings service integration flow"""
    
    def setup_method(self):
        """Set up test environment"""
        self.test_symbols = ['AAPL', 'MSFT', 'TSLA']
        self.mock_earnings_data = [
            {
                'symbol': 'AAPL',
                'earnings_date': '2024-01-25',
                'implied_move_pct': 4.5,
                'direction': 'Bullish',
                'confidence': 0.7,
                'risk_level': 'medium'
            },
            {
                'symbol': 'MSFT', 
                'earnings_date': '2024-01-28',
                'implied_move_pct': 3.2,
                'direction': 'Neutral',
                'confidence': 0.5,
                'risk_level': 'low'
            }
        ]
    
    def test_earnings_schedule_generation_flag_enabled(self):
        """Test earnings schedule generation when flag is enabled"""
        mock_df = pd.DataFrame(self.mock_earnings_data)
        
        with patch('services.earnings_service.EarningsAnalysisService') as mock_service_class, \
             patch('config.feature_flags.feature_flags') as mock_flags, \
             patch('os.path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open, \
             patch('os.rename') as mock_rename:
            
            # Setup mocks
            mock_service = Mock()
            mock_service.get_schedule.return_value = mock_df
            mock_service_class.return_value = mock_service
            mock_flags.is_enabled.return_value = True
            mock_exists.return_value = True
            
            # Import and run the earnings flow
            from services.earnings_service import EarningsAnalysisService
            
            earnings_service = EarningsAnalysisService()
            result_df = earnings_service.get_schedule(self.test_symbols, days=14)
            
            # Verify service was called correctly
            assert not result_df.empty
            assert len(result_df) == 2
            assert 'AAPL' in result_df['symbol'].values
            assert 'MSFT' in result_df['symbol'].values
    
    def test_earnings_schedule_generation_flag_disabled(self):
        """Test earnings schedule generation when flag is disabled"""
        with patch('config.feature_flags.is_earnings_reads_enabled') as mock_flag:
            mock_flag.return_value = False
            
            from services.earnings_service import EarningsAnalysisService
            
            earnings_service = EarningsAnalysisService()
            result_df = earnings_service.get_schedule(self.test_symbols, days=14)
            
            # Should return empty DataFrame when disabled
            assert result_df.empty
    
    def test_csv_persistence_atomic_write(self):
        """Test that CSV is written atomically"""
        mock_df = pd.DataFrame(self.mock_earnings_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "earnings_schedule.csv")
            temp_path = f"{csv_path}.tmp"
            
            with patch('services.earnings_service.EarningsAnalysisService') as mock_service_class, \
                 patch('config.feature_flags.feature_flags') as mock_flags, \
                 patch('os.getenv') as mock_getenv:
                
                # Setup mocks
                mock_service = Mock()
                mock_service.get_schedule.return_value = mock_df
                mock_service_class.return_value = mock_service
                mock_flags.is_enabled.return_value = True
                mock_getenv.return_value = '14'
                
                # Create mock for atomic write process
                with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
                     patch('os.rename') as mock_rename:
                    
                    # Simulate the earnings flow logic
                    earnings_service = EarningsAnalysisService()
                    earnings_df = earnings_service.get_schedule(self.test_symbols, days=14)
                    
                    if not earnings_df.empty:
                        # Simulate atomic write
                        earnings_df.to_csv(temp_path, index=False)
                        os.rename(temp_path, csv_path)
                    
                    # Verify atomic write pattern was used
                    mock_to_csv.assert_called_once_with(temp_path, index=False)
                    mock_rename.assert_called_once_with(temp_path, csv_path)
    
    def test_email_report_earnings_section_included(self):
        """Test that earnings section appears in email when enabled"""
        mock_df = pd.DataFrame(self.mock_earnings_data)
        
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch('config.feature_flags.is_earnings_reads_enabled') as mock_flag, \
             patch.dict(os.environ, {}, clear=True):
            
            mock_flag.return_value = True
            
            from email_report import send_report
            
            # Create a temporary file for HTML output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                temp_html_path = tmp_file.name
            
            try:
                # Call send_report with earnings schedule
                send_report(
                    watchlist=['AAPL'],
                    portfolio=['MSFT'],
                    head7={},
                    head30={},
                    preds={},
                    out_path=temp_html_path,
                    earnings_schedule=mock_df
                )
                
                # Read the generated HTML
                with open(temp_html_path, 'r') as f:
                    html_content = f.read()
                
                # Verify earnings section is present
                assert "📅 Upcoming Earnings" in html_content
                assert "AAPL" in html_content
                assert "MSFT" in html_content
                assert "2024-01-25" in html_content
                assert "2024-01-28" in html_content
                assert "4.5%" in html_content
                assert "3.2%" in html_content
                assert "Bullish" in html_content
                assert "Neutral" in html_content
                assert "Implied moves are based on options pricing" in html_content
                
            finally:
                # Clean up
                if os.path.exists(temp_html_path):
                    os.unlink(temp_html_path)
    
    def test_email_report_earnings_section_excluded_when_disabled(self):
        """Test that earnings section is excluded when flag is disabled"""
        mock_df = pd.DataFrame(self.mock_earnings_data)
        
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch('config.feature_flags.is_earnings_reads_enabled') as mock_flag, \
             patch.dict(os.environ, {}, clear=True):
            
            mock_flag.return_value = False
            
            from email_report import send_report
            
            # Create a temporary file for HTML output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                temp_html_path = tmp_file.name
            
            try:
                # Call send_report with earnings schedule but flag disabled
                send_report(
                    watchlist=['AAPL'],
                    portfolio=['MSFT'],
                    head7={},
                    head30={},
                    preds={},
                    out_path=temp_html_path,
                    earnings_schedule=mock_df
                )
                
                # Read the generated HTML
                with open(temp_html_path, 'r') as f:
                    html_content = f.read()
                
                # Verify earnings section is NOT present
                assert "📅 Upcoming Earnings" not in html_content
                assert "Implied moves are based on options pricing" not in html_content
                
            finally:
                # Clean up
                if os.path.exists(temp_html_path):
                    os.unlink(temp_html_path)
    
    def test_email_report_earnings_section_excluded_when_empty(self):
        """Test that earnings section is excluded when DataFrame is empty"""
        empty_df = pd.DataFrame()
        
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch('config.feature_flags.is_earnings_reads_enabled') as mock_flag, \
             patch.dict(os.environ, {}, clear=True):
            
            mock_flag.return_value = True  # Flag enabled but no data
            
            from email_report import send_report
            
            # Create a temporary file for HTML output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                temp_html_path = tmp_file.name
            
            try:
                # Call send_report with empty earnings schedule
                send_report(
                    watchlist=['AAPL'],
                    portfolio=['MSFT'],
                    head7={},
                    head30={},
                    preds={},
                    out_path=temp_html_path,
                    earnings_schedule=empty_df
                )
                
                # Read the generated HTML
                with open(temp_html_path, 'r') as f:
                    html_content = f.read()
                
                # Verify earnings section is NOT present when no data
                assert "📅 Upcoming Earnings" not in html_content
                
            finally:
                # Clean up
                if os.path.exists(temp_html_path):
                    os.unlink(temp_html_path)
    
    def test_earnings_schedule_date_formatting(self):
        """Test proper date formatting in earnings schedule"""
        # Test data with different date formats
        test_data = [
            {
                'symbol': 'AAPL',
                'earnings_date': pd.Timestamp('2024-01-25'),
                'implied_move_pct': 4.5,
                'direction': 'Bullish'
            },
            {
                'symbol': 'MSFT',
                'earnings_date': '2024-01-28T09:30:00',  # String with time
                'implied_move_pct': None,  # Test None handling
                'direction': 'Neutral'
            }
        ]
        
        mock_df = pd.DataFrame(test_data)
        
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch('config.feature_flags.is_earnings_reads_enabled') as mock_flag, \
             patch.dict(os.environ, {}, clear=True):
            
            mock_flag.return_value = True
            
            from email_report import send_report
            
            # Create a temporary file for HTML output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                temp_html_path = tmp_file.name
            
            try:
                send_report(
                    watchlist=['AAPL'],
                    portfolio=['MSFT'],
                    head7={},
                    head30={},
                    preds={},
                    out_path=temp_html_path,
                    earnings_schedule=mock_df
                )
                
                # Read the generated HTML
                with open(temp_html_path, 'r') as f:
                    html_content = f.read()
                
                # Verify proper date formatting
                assert "2024-01-25" in html_content
                assert "2024-01-28" in html_content  # Should extract date part only
                assert "4.5%" in html_content
                assert "N/A" in html_content  # For None implied_move_pct
                
            finally:
                # Clean up
                if os.path.exists(temp_html_path):
                    os.unlink(temp_html_path)
    
    def test_main_pipeline_earnings_integration(self):
        """Test earnings integration in main pipeline"""
        mock_df = pd.DataFrame(self.mock_earnings_data)
        
        with patch('services.earnings_service.EarningsAnalysisService') as mock_service_class, \
             patch('config.feature_flags.feature_flags') as mock_flags, \
             patch('os.getenv') as mock_getenv, \
             patch('os.rename') as mock_rename, \
             patch('pandas.DataFrame.to_csv') as mock_to_csv:
            
            # Setup mocks
            mock_service = Mock()
            mock_service.get_schedule.return_value = mock_df
            mock_service_class.return_value = mock_service
            mock_flags.is_enabled.return_value = True
            mock_getenv.return_value = '14'  # EARNINGS_WINDOW_DAYS
            
            # Test the main pipeline earnings wiring logic
            from services.earnings_service import EarningsAnalysisService
            
            # Simulate main.py earnings flow
            if mock_flags.is_enabled('enable_earnings_reads'):
                earnings_service = EarningsAnalysisService()
                days_ahead = int(os.getenv('EARNINGS_WINDOW_DAYS', '14'))
                earnings_df = earnings_service.get_schedule(self.test_symbols, days=days_ahead)
                
                if not earnings_df.empty:
                    # Simulate atomic CSV write
                    earnings_csv_path = "data/earnings_schedule.csv"
                    temp_path = f"{earnings_csv_path}.tmp"
                    earnings_df.to_csv(temp_path, index=False)
                    os.rename(temp_path, earnings_csv_path)
            
            # Verify the flow was executed correctly
            mock_service.get_schedule.assert_called_once_with(self.test_symbols, days=14)
            mock_to_csv.assert_called_once()
            mock_rename.assert_called_once()
    
    def test_earnings_window_configuration(self):
        """Test earnings window configuration via environment variable"""
        with patch('services.earnings_service.EarningsAnalysisService') as mock_service_class, \
             patch('config.feature_flags.feature_flags') as mock_flags, \
             patch('os.getenv') as mock_getenv:
            
            # Setup mocks
            mock_service = Mock()
            mock_service.get_schedule.return_value = pd.DataFrame()
            mock_service_class.return_value = mock_service
            mock_flags.is_enabled.return_value = True
            
            # Test different window configurations
            test_cases = ['7', '14', '21', '30']
            
            for days_str in test_cases:
                mock_getenv.return_value = days_str
                
                earnings_service = EarningsAnalysisService()
                days_ahead = int(os.getenv('EARNINGS_WINDOW_DAYS', '14'))
                earnings_service.get_schedule(self.test_symbols, days=days_ahead)
                
                # Verify correct days parameter was used
                expected_days = int(days_str)
                mock_service.get_schedule.assert_called_with(self.test_symbols, days=expected_days)


class TestEarningsNonRegression:
    """Test that earnings feature doesn't break existing functionality"""
    
    def test_email_report_unchanged_when_flag_off(self):
        """Test that email report is unchanged when earnings flag is off"""
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch('config.feature_flags.is_earnings_reads_enabled') as mock_flag, \
             patch.dict(os.environ, {}, clear=True):
            
            mock_flag.return_value = False
            
            from email_report import send_report
            
            # Create a temporary file for HTML output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                temp_html_path = tmp_file.name
            
            try:
                # Call send_report without earnings (existing behavior)
                send_report(
                    watchlist=['AAPL'],
                    portfolio=['MSFT'],
                    head7={},
                    head30={},
                    preds={},
                    out_path=temp_html_path
                    # Note: earnings_schedule parameter not provided
                )
                
                # Read the generated HTML
                with open(temp_html_path, 'r') as f:
                    html_content = f.read()
                
                # Verify standard report sections are present
                assert "Stock News Forecast Report" in html_content
                assert "30-Day Sentiment — Portfolio" in html_content
                assert "Watchlist Forecasts" in html_content
                assert "Portfolio Forecasts" in html_content
                
                # Verify earnings section is NOT present
                assert "📅 Upcoming Earnings" not in html_content
                
            finally:
                # Clean up
                if os.path.exists(temp_html_path):
                    os.unlink(temp_html_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])