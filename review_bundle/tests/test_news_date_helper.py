#!/usr/bin/env python3
"""
Unit tests for news scraper date helper function
"""
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
from news_scraper import get_date_window


class TestNewsDateHelper(unittest.TestCase):
    """Test cases for news date helper functionality"""
    
    def test_get_date_window_default_lookback(self):
        """Test get_date_window with default 30-day lookback"""
        # Mock current time to 2025-08-18 12:00 PM Los Angeles time
        frozen_datetime = datetime(2025, 8, 18, 12, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        
        with patch('news_scraper.datetime') as mock_datetime:
            mock_datetime.now.return_value = frozen_datetime
            
            start_iso, end_iso = get_date_window()
            
            # Should return 30 days back from 2025-08-18
            self.assertEqual(start_iso, "2025-07-19")  # 30 days before 2025-08-18
            self.assertEqual(end_iso, "2025-08-18")
            
            # Verify it was called with correct timezone
            mock_datetime.now.assert_called_with(ZoneInfo("America/Los_Angeles"))
    
    def test_get_date_window_custom_lookback(self):
        """Test get_date_window with custom lookback days"""
        frozen_datetime = datetime(2025, 8, 18, 9, 30, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        
        with patch('news_scraper.datetime') as mock_datetime:
            mock_datetime.now.return_value = frozen_datetime
            
            # Test 7-day lookback
            start_iso, end_iso = get_date_window(lookback_days=7)
            
            self.assertEqual(start_iso, "2025-08-11")  # 7 days before 2025-08-18
            self.assertEqual(end_iso, "2025-08-18")
    
    def test_get_date_window_custom_timezone(self):
        """Test get_date_window with custom timezone"""
        # Test with UTC timezone
        frozen_datetime = datetime(2025, 8, 19, 5, 0, 0, tzinfo=ZoneInfo("UTC"))
        
        with patch('news_scraper.datetime') as mock_datetime:
            mock_datetime.now.return_value = frozen_datetime
            
            start_iso, end_iso = get_date_window(lookback_days=14, tz="UTC")
            
            self.assertEqual(start_iso, "2025-08-05")  # 14 days before 2025-08-19
            self.assertEqual(end_iso, "2025-08-19")
            
            mock_datetime.now.assert_called_with(ZoneInfo("UTC"))
    
    def test_get_date_window_edge_cases(self):
        """Test edge cases like month/year boundaries"""
        # Test month boundary crossing
        frozen_datetime = datetime(2025, 9, 2, 15, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        
        with patch('news_scraper.datetime') as mock_datetime:
            mock_datetime.now.return_value = frozen_datetime
            
            # 5-day lookback should cross month boundary
            start_iso, end_iso = get_date_window(lookback_days=5)
            
            self.assertEqual(start_iso, "2025-08-28")  # 5 days before 2025-09-02
            self.assertEqual(end_iso, "2025-09-02")
    
    def test_get_date_window_one_day_lookback(self):
        """Test minimal 1-day lookback"""
        frozen_datetime = datetime(2025, 8, 18, 18, 45, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        
        with patch('news_scraper.datetime') as mock_datetime:
            mock_datetime.now.return_value = frozen_datetime
            
            start_iso, end_iso = get_date_window(lookback_days=1)
            
            self.assertEqual(start_iso, "2025-08-17")  # 1 day before 2025-08-18
            self.assertEqual(end_iso, "2025-08-18")
    
    def test_date_window_returns_strings(self):
        """Test that function returns ISO format strings"""
        frozen_datetime = datetime(2025, 8, 18, 10, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        
        with patch('news_scraper.datetime') as mock_datetime:
            mock_datetime.now.return_value = frozen_datetime
            
            start_iso, end_iso = get_date_window()
            
            # Verify return types are strings
            self.assertIsInstance(start_iso, str)
            self.assertIsInstance(end_iso, str)
            
            # Verify ISO format (YYYY-MM-DD)
            self.assertRegex(start_iso, r'^\d{4}-\d{2}-\d{2}$')
            self.assertRegex(end_iso, r'^\d{4}-\d{2}-\d{2}$')


if __name__ == '__main__':
    unittest.main()