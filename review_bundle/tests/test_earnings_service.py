#!/usr/bin/env python3
"""
Unit tests for earnings analysis service
Tests earnings calendar, implied moves, and directional classification
"""
import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Set up test environment
import os
os.environ['ENABLE_EARNINGS_READS'] = 'true'

from services.earnings_service import EarningsAnalysisService


class TestEarningsAnalysisService(unittest.TestCase):
    """Test cases for earnings analysis functionality"""
    
    def setUp(self):
        """Set up test environment with mock data"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = Path(self.temp_dir)
        
        # Create mock earnings calendar
        self.earnings_calendar_path = self.test_data_path / "earnings_calendar.csv"
        earnings_data = {
            'ticker': ['AAPL', 'MSFT', 'TSLA', 'NVDA'],
            'earnings_date': [
                (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
                (datetime.now() + timedelta(days=21)).strftime('%Y-%m-%d')
            ],
            'confirmed': [True, True, False, True],
            'time_of_day': ['after_close', 'before_open', 'after_close', 'before_open'],
            'fiscal_quarter': ['Q4', 'Q1', 'Q3', 'Q4'],
            'fiscal_year': [2023, 2024, 2023, 2023]
        }
        
        earnings_df = pd.DataFrame(earnings_data)
        earnings_df.to_csv(self.earnings_calendar_path, index=False)
        
        # Create mock earnings stats file
        self.earnings_stats_path = self.test_data_path / "earnings_stats.csv"
        stats_data = {
            'ticker': ['AAPL', 'MSFT'],
            'implied_move_pct': [4.5, 3.2],
            'avg_abs_move_8q': [5.1, 4.8],
            'last_updated': [datetime.now().isoformat(), datetime.now().isoformat()],
            'next_earnings_date': [None, None],
            'quarters_tracked': [8, 8]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(self.earnings_stats_path, index=False)
        
        # Create mock earnings history
        self.earnings_history_dir = self.test_data_path / "earnings_history"
        self.earnings_history_dir.mkdir(exist_ok=True)
        
        # Mock AAPL earnings history
        aapl_history = {
            'earnings_date': [
                '2023-07-15', '2023-04-15', '2023-01-15', '2022-10-15'
            ],
            'day_after_move_pct': [2.3, -1.8, 4.5, -3.2]
        }
        aapl_history_df = pd.DataFrame(aapl_history)
        aapl_history_path = self.earnings_history_dir / "AAPL_earnings_history.csv"
        aapl_history_df.to_csv(aapl_history_path, index=False)
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('services.earnings_service.EarningsAnalysisService.data_path', new_callable=lambda: None)
    def test_upcoming_earnings_filtering(self, mock_data_path):
        """Test filtering of upcoming earnings within date range"""
        # Setup service with test data path
        service = EarningsAnalysisService()
        service.data_path = self.test_data_path
        service.earnings_calendar_path = self.earnings_calendar_path
        service.earnings_stats_path = self.earnings_stats_path
        service.earnings_history_dir = self.earnings_history_dir
        
        with patch.object(service, 'analyze_earnings_setup') as mock_analyze:
            # Mock analysis to avoid complex dependencies
            mock_analyze.return_value = {
                'implied_move_pct': 4.2,
                'avg_abs_move_8q': 5.0,
                'direction_prediction': 'Up',
                'confidence': 0.75,
                'why': 'Test analysis',
                'risk_level': 'medium'
            }
            
            # Test 14-day window
            upcoming = service.get_upcoming_earnings(days=14)
            
            # Should return AAPL, MSFT, TSLA (within 14 days)
            self.assertEqual(len(upcoming), 3)
            symbols = [e['symbol'] for e in upcoming]
            self.assertIn('AAPL', symbols)
            self.assertIn('MSFT', symbols)
            self.assertIn('TSLA', symbols)
            self.assertNotIn('NVDA', symbols)  # 21 days out
            
            # Check date sorting
            dates = [pd.to_datetime(e['earnings_date']) for e in upcoming]
            self.assertEqual(dates, sorted(dates))
            
            # Check structure
            first_earning = upcoming[0]
            self.assertIn('symbol', first_earning)
            self.assertIn('earnings_date', first_earning)
            self.assertIn('confirmed', first_earning)
            self.assertIn('days_until', first_earning)
            self.assertIn('analysis', first_earning)
    
    @patch('prediction.fetch_price_history')
    def test_implied_move_calculation(self, mock_price_history):
        """Test implied move calculation from price volatility"""
        # Mock price data for AAPL
        dates = pd.date_range(start='2023-10-01', periods=30, freq='D')
        mock_prices = np.random.normal(180, 5, 30)  # Mock price around 180 with some volatility
        mock_df = pd.DataFrame({
            'Close': mock_prices,
            'Date': dates
        })
        mock_price_history.return_value = mock_df
        
        service = EarningsAnalysisService()
        service.data_path = self.test_data_path
        
        implied_move = service._calculate_implied_move('AAPL')
        
        # Should return a reasonable percentage
        self.assertIsNotNone(implied_move)
        self.assertIsInstance(implied_move, float)
        self.assertGreater(implied_move, 0)
        self.assertLess(implied_move, 25)  # Should be capped at 25%
    
    def test_8q_average_move_from_history(self):
        """Test 8-quarter average move calculation from historical data"""
        service = EarningsAnalysisService()
        service.data_path = self.test_data_path
        service.earnings_history_dir = self.earnings_history_dir
        
        avg_move = service._calculate_8q_average_move('AAPL')
        
        # Should calculate from mock history: |2.3| + |-1.8| + |4.5| + |-3.2| / 4
        expected = (2.3 + 1.8 + 4.5 + 3.2) / 4
        self.assertAlmostEqual(avg_move, expected, places=1)
    
    @patch('prediction.fetch_price_history')
    def test_8q_average_move_from_price_fallback(self, mock_price_history):
        """Test fallback to price data when no earnings history available"""
        # Mock price data with some volatility
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        base_price = 100
        returns = np.random.normal(0, 0.02, 365)  # 2% daily volatility
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        mock_df = pd.DataFrame({
            'Close': prices,
            'Date': dates
        })
        mock_price_history.return_value = mock_df
        
        service = EarningsAnalysisService()
        service.data_path = self.test_data_path
        service.earnings_history_dir = self.earnings_history_dir
        
        # Test symbol without history file
        avg_move = service._calculate_8q_average_move('XYZ')
        
        # Should return 95th percentile of daily moves
        self.assertIsNotNone(avg_move)
        self.assertIsInstance(avg_move, float)
        self.assertGreater(avg_move, 0)
    
    @patch('services.earnings_service.EarningsAnalysisService._gather_earnings_signals')
    def test_earnings_direction_classification(self, mock_signals):
        """Test earnings direction classification logic"""
        service = EarningsAnalysisService()
        
        # Test bullish scenario
        mock_signals.return_value = {
            'momentum_1m': 0.08,  # Strong positive momentum
            'news_consensus': 0.20,  # Positive news
            'sector_momentum': 0.03,  # Positive sector
            'options_flow': 'bullish',
            'vol_expansion': False,
            'data_quality': 0.8
        }
        
        result = service._classify_earnings_direction('AAPL')
        
        self.assertEqual(result['direction_prediction'], 'Up')
        self.assertGreater(result['confidence'], 0.5)
        self.assertIn('bullish', result['why'].lower())
        
        # Test bearish scenario
        mock_signals.return_value = {
            'momentum_1m': -0.07,  # Strong negative momentum
            'news_consensus': -0.18,  # Negative news
            'sector_momentum': -0.03,  # Negative sector
            'options_flow': 'bearish',
            'vol_expansion': False,
            'data_quality': 0.8
        }
        
        result = service._classify_earnings_direction('AAPL')
        
        self.assertEqual(result['direction_prediction'], 'Down')
        self.assertGreater(result['confidence'], 0.5)
        self.assertIn('bearish', result['why'].lower())
        
        # Test uncertain scenario with volatility expansion
        mock_signals.return_value = {
            'momentum_1m': 0.02,  # Weak momentum
            'news_consensus': -0.05,  # Mixed news
            'sector_momentum': 0.01,
            'options_flow': 'neutral',
            'vol_expansion': True,  # High uncertainty
            'data_quality': 0.6
        }
        
        result = service._classify_earnings_direction('AAPL')
        
        self.assertEqual(result['direction_prediction'], 'Big Swing Unsure')
        self.assertEqual(result['risk_level'], 'high')
        self.assertIn('mixed', result['why'].lower())
    
    def test_earnings_stats_persistence(self):
        """Test that earnings analysis stats are persisted correctly"""
        service = EarningsAnalysisService()
        service.data_path = self.test_data_path
        service.earnings_stats_path = self.earnings_stats_path
        
        # Test updating existing record
        test_analysis = {
            'implied_move_pct': 5.5,
            'avg_abs_move_8q': 6.2,
            'direction_prediction': 'Up',
            'confidence': 0.8
        }
        
        service._update_earnings_stats('AAPL', test_analysis)
        
        # Verify stats were updated
        updated_stats = pd.read_csv(self.earnings_stats_path)
        aapl_row = updated_stats[updated_stats['ticker'] == 'AAPL'].iloc[0]
        
        self.assertEqual(aapl_row['implied_move_pct'], 5.5)
        self.assertEqual(aapl_row['avg_abs_move_8q'], 6.2)
        
        # Test adding new record
        service._update_earnings_stats('GOOGL', test_analysis)
        
        updated_stats = pd.read_csv(self.earnings_stats_path)
        self.assertIn('GOOGL', updated_stats['ticker'].values)
    
    @patch('services.earnings_service.EarningsAnalysisService.analyze_earnings_setup')
    def test_earnings_explanation_generation(self, mock_analyze):
        """Test detailed earnings explanation generation"""
        service = EarningsAnalysisService()
        service.data_path = self.test_data_path
        
        # Mock analysis result
        mock_analyze.return_value = {
            'implied_move_pct': 4.8,
            'avg_abs_move_8q': 5.2,
            'direction_prediction': 'Up',
            'confidence': 0.75,
            'why': 'Strong technical and fundamental signals',
            'risk_level': 'medium'
        }
        
        with patch.object(service, '_gather_earnings_signals') as mock_signals:
            mock_signals.return_value = {
                'momentum_1m': 0.06,
                'news_consensus': 0.15,
                'sector_momentum': 0.02,
                'options_flow': 'bullish',
                'vol_expansion': False,
                'data_quality': 0.8
            }
            
            with patch.object(service, '_calculate_expected_range') as mock_range:
                mock_range.return_value = {
                    'current_price': 180.0,
                    'upside_target': 188.64,
                    'downside_target': 171.36,
                    'range_width_pct': 9.6
                }
                
                explanation = service.explain_earnings_analysis('AAPL')
                
                # Check explanation structure
                self.assertEqual(explanation['symbol'], 'AAPL')
                self.assertIn('analysis_timestamp', explanation)
                self.assertIn('prediction', explanation)
                self.assertIn('quantitative_metrics', explanation)
                self.assertIn('contributing_factors', explanation)
                self.assertIn('disclaimer', explanation)
                
                # Check prediction details
                prediction = explanation['prediction']
                self.assertEqual(prediction['direction'], 'Up')
                self.assertEqual(prediction['confidence'], 0.75)
                self.assertEqual(prediction['risk_level'], 'medium')
                
                # Check contributing factors
                factors = explanation['contributing_factors']
                self.assertIn('technical_momentum', factors)
                self.assertIn('news_sentiment', factors)
                self.assertIn('volatility_environment', factors)
                self.assertIn('options_activity', factors)
                
                # Check technical momentum signal
                tech_momentum = factors['technical_momentum']
                self.assertEqual(tech_momentum['signal'], 'bullish')
                self.assertIn('+6.0%', tech_momentum['value'])


class TestEarningsAPIIntegration(unittest.TestCase):
    """Integration tests for earnings API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal test app
        from api.app import create_app
        self.app = create_app()
        self.client = self.app.test_client()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('services.earnings_service.EarningsAnalysisService.get_upcoming_earnings')
    def test_upcoming_earnings_api_success(self, mock_get_earnings):
        """Test successful upcoming earnings API call"""
        # Mock service response
        mock_get_earnings.return_value = [
            {
                'symbol': 'AAPL',
                'earnings_date': '2023-11-01T00:00:00',
                'confirmed': True,
                'days_until': 3,
                'analysis': {
                    'implied_move_pct': 4.5,
                    'direction_prediction': 'Up',
                    'confidence': 0.75,
                    'risk_level': 'medium'
                }
            }
        ]
        
        response = self.client.get('/api/v1/earnings/upcoming?days=14')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        # Check response structure
        self.assertIn('upcoming_earnings', data)
        self.assertIn('summary', data)
        self.assertIn('meta', data)
        
        # Check earnings data
        earnings = data['upcoming_earnings']
        self.assertEqual(len(earnings), 1)
        self.assertEqual(earnings[0]['symbol'], 'AAPL')
        
        # Check summary
        summary = data['summary']
        self.assertEqual(summary['total_earnings'], 1)
        self.assertEqual(summary['confirmed_count'], 1)
        
        # Check disclaimer
        self.assertIn('disclaimer', data['meta'])
    
    @patch('services.earnings_service.EarningsAnalysisService.explain_earnings_analysis')
    def test_earnings_explanation_api_success(self, mock_explain):
        """Test successful earnings explanation API call"""
        # Mock service response
        mock_explain.return_value = {
            'symbol': 'AAPL',
            'analysis_timestamp': '2023-10-18T12:00:00',
            'prediction': {
                'direction': 'Up',
                'confidence': 0.75,
                'reasoning': 'Strong momentum with positive sentiment',
                'risk_level': 'medium'
            },
            'quantitative_metrics': {
                'implied_move_pct': 4.8,
                'historical_avg_move_8q': 5.2
            }
        }
        
        response = self.client.get('/api/v1/earnings/AAPL/explain')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        # Check explanation structure
        self.assertEqual(data['symbol'], 'AAPL')
        self.assertIn('prediction', data)
        self.assertIn('quantitative_metrics', data)
    
    def test_earnings_api_invalid_parameters(self):
        """Test API parameter validation"""
        # Test invalid days parameter
        response = self.client.get('/api/v1/earnings/upcoming?days=100')
        self.assertEqual(response.status_code, 400)
        
        # Test invalid symbol
        response = self.client.get('/api/v1/earnings/TOOLONGSYMBOL/explain')
        self.assertEqual(response.status_code, 400)
    
    @patch.dict(os.environ, {'ENABLE_EARNINGS_READS': 'false'})
    def test_earnings_api_feature_disabled(self):
        """Test API when earnings feature is disabled"""
        app = create_app()
        client = app.test_client()
        
        response = client.get('/api/v1/earnings/upcoming')
        self.assertEqual(response.status_code, 503)
        
        data = response.get_json()
        self.assertEqual(data['code'], 'FEATURE_DISABLED')


if __name__ == '__main__':
    unittest.main()