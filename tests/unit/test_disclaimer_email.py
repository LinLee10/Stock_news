#!/usr/bin/env python3
"""
Unit tests for email disclaimer functionality
Tests that disclaimer text appears in email reports
"""

import os
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEmailDisclaimer:
    """Test disclaimer inclusion in email reports"""
    
    def test_default_disclaimer_in_email(self):
        """Test that default disclaimer appears in email HTML"""
        # Mock the email_report function to avoid SMTP dependencies
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch.dict(os.environ, {}, clear=True):  # Clear env vars
            
            # Import after patching to avoid loading real env vars
            from email_report import generate_html_report
            
            # Generate minimal report
            html_output = generate_html_report(
                "test_output.html",
                sentiment_by_ticker={},
                preds={},
                mention_leaders=[],
                watchlist=[],
                portfolio=[]
            )
            
            # Check that both header and footer disclaimers exist
            assert "⚠️ DISCLAIMER" in html_output
            assert "This report is for research and educational purposes only" in html_output
            assert "Not financial advice" in html_output
            assert "Stonk News" in html_output  # Default company name
    
    def test_custom_disclaimer_in_email(self):
        """Test that custom disclaimer from environment appears in email"""
        custom_disclaimer = "Custom research disclaimer for testing purposes only."
        custom_company = "Test Company Inc"
        
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch.dict(os.environ, {
                 'DISCLAIMER_TEXT': custom_disclaimer,
                 'COMPANY_NAME': custom_company
             }):
            
            from email_report import generate_html_report
            
            html_output = generate_html_report(
                "test_output.html",
                sentiment_by_ticker={},
                preds={},
                mention_leaders=[],
                watchlist=[],
                portfolio=[]
            )
            
            # Check custom disclaimer appears
            assert custom_disclaimer in html_output
            assert custom_company in html_output
            assert "⚠️ DISCLAIMER" in html_output
    
    def test_disclaimer_appears_once_in_header(self):
        """Test that disclaimer appears exactly once in header section"""
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch.dict(os.environ, {}, clear=True):
            
            from email_report import generate_html_report
            
            html_output = generate_html_report(
                "test_output.html",
                sentiment_by_ticker={},
                preds={},
                mention_leaders=[],
                watchlist=[],
                portfolio=[]
            )
            
            # Split by main heading to check header section
            parts = html_output.split("<h2>Stock News Forecast Report")
            assert len(parts) == 2, "Should have header and body sections"
            
            header_section = parts[0]
            assert "⚠️ DISCLAIMER" in header_section
            assert header_section.count("⚠️ DISCLAIMER") == 1
    
    def test_disclaimer_appears_once_in_footer(self):
        """Test that disclaimer appears exactly once in footer section"""
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch.dict(os.environ, {}, clear=True):
            
            from email_report import generate_html_report
            
            html_output = generate_html_report(
                "test_output.html",
                sentiment_by_ticker={},
                preds={},
                mention_leaders=[],
                watchlist=[],
                portfolio=[]
            )
            
            # Check footer disclaimer
            assert "This analysis is automated and should not be used as the sole basis" in html_output
            assert html_output.count("This analysis is automated") == 1
    
    def test_disclaimer_html_structure(self):
        """Test that disclaimer has proper HTML structure and styling"""
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch.dict(os.environ, {}, clear=True):
            
            from email_report import generate_html_report
            
            html_output = generate_html_report(
                "test_output.html",
                sentiment_by_ticker={},
                preds={},
                mention_leaders=[],
                watchlist=[],
                portfolio=[]
            )
            
            # Check for proper HTML structure
            assert "background-color: #f8f9fa" in html_output  # Header disclaimer styling
            assert "border-left: 4px solid #ffc107" in html_output  # Warning color
            assert "font-size: 12px" in html_output  # Proper font sizing
            assert "⚠️ DISCLAIMER" in html_output  # Warning emoji
    
    def test_disclaimer_with_feature_flags_disabled(self):
        """Test disclaimer appears even when all feature flags are off"""
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch.dict(os.environ, {
                 'ENABLE_PORTFOLIO_ANALYTICS': 'false',
                 'ENABLE_FINBERT_PIPELINE': 'false',
                 'ENABLE_SMART_ALERTS': 'false',
                 'ENABLE_CORRELATION': 'false'
             }):
            
            from email_report import generate_html_report
            
            html_output = generate_html_report(
                "test_output.html",
                sentiment_by_ticker={},
                preds={},
                mention_leaders=[],
                watchlist=[],
                portfolio=[]
            )
            
            # Disclaimer should always appear regardless of feature flags
            assert "⚠️ DISCLAIMER" in html_output
            assert "Not financial advice" in html_output
    
    def test_disclaimer_persists_in_file_output(self):
        """Test that disclaimer is written to HTML file"""
        import tempfile
        
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch.dict(os.environ, {}, clear=True):
            
            from email_report import generate_html_report
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                generate_html_report(
                    temp_path,
                    sentiment_by_ticker={},
                    preds={},
                    mention_leaders=[],
                    watchlist=[],
                    portfolio=[]
                )
                
                # Read the file and verify disclaimer
                with open(temp_path, 'r') as f:
                    file_content = f.read()
                
                assert "⚠️ DISCLAIMER" in file_content
                assert "Not financial advice" in file_content
                
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_disclaimer_encoding_safety(self):
        """Test that disclaimer handles special characters safely"""
        special_disclaimer = "Tëst dïsclãimer with spéciâl çharåcters & symbols <>&"
        
        with patch('email_report.smtplib.SMTP'), \
             patch('email_report.load_dotenv'), \
             patch.dict(os.environ, {
                 'DISCLAIMER_TEXT': special_disclaimer
             }):
            
            from email_report import generate_html_report
            
            html_output = generate_html_report(
                "test_output.html",
                sentiment_by_ticker={},
                preds={},
                mention_leaders=[],
                watchlist=[],
                portfolio=[]
            )
            
            # Should handle special characters without breaking HTML
            assert special_disclaimer in html_output
            assert "<html>" in html_output and "</html>" in html_output  # Valid HTML structure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])