#!/usr/bin/env python3
"""
Unit tests for recommendations API endpoint
Tests watchlist and portfolio recommendation generation
"""
import unittest
import json
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Set up test environment
import os
os.environ['ENABLE_RECOS'] = 'true'
os.environ['ENABLE_API_ENDPOINTS'] = 'true'

from api.app import create_app

class TestRecommendationsAPI(unittest.TestCase):
    """Test cases for recommendations API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = create_app()
        self.client = self.app.test_client()
        
        # Create temporary data directory
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('services.recommendations_service.RecommendationsService')
    def test_watchlist_recommendations_success(self, mock_service):
        """Test successful watchlist recommendations request"""
        # Mock service response
        mock_service_instance = MagicMock()
        mock_service_instance.generate_watchlist_recommendations.return_value = [
            {
                'symbol': 'AAPL',
                'action': 'buy',
                'confidence': 0.85,
                'why': 'Strong momentum with positive news consensus',
                'horizon_days': 30,
                'next_check_date': '2023-11-18T12:00:00',
                'context': 'watchlist',
                'current_price': 185.25,
                'position_status': 'not_held'
            },
            {
                'symbol': 'MSFT',
                'action': 'hold',
                'confidence': 0.65,
                'why': 'Mixed signals with upcoming earnings',
                'horizon_days': 14,
                'next_check_date': '2023-11-04T12:00:00',
                'context': 'watchlist',
                'current_price': 338.50,
                'position_status': 'not_held'
            }
        ]
        mock_service.return_value = mock_service_instance
        
        response = self.client.get('/api/v1/recs?scope=watchlist')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response structure
        self.assertEqual(data['scope'], 'watchlist')
        self.assertIn('recommendations', data)
        self.assertIn('summary', data)
        self.assertIn('meta', data)
        
        # Check recommendations
        recommendations = data['recommendations']
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]['symbol'], 'AAPL')
        self.assertEqual(recommendations[0]['action'], 'buy')
        self.assertEqual(recommendations[0]['confidence'], 0.85)
        
        # Check summary
        summary = data['summary']
        self.assertEqual(summary['total'], 2)
        self.assertIn('action_breakdown', summary)
        self.assertIn('average_confidence', summary)
        
        # Check disclaimer
        self.assertIn('disclaimer', data['meta'])
        self.assertIn('financial advice', data['meta']['disclaimer'])
    
    @patch('services.recommendations_service.RecommendationsService')
    def test_portfolio_recommendations_with_pnl(self, mock_service):
        """Test portfolio recommendations with P/L context"""
        mock_service_instance = MagicMock()
        mock_service_instance.generate_portfolio_recommendations.return_value = [
            {
                'symbol': 'TSLA',
                'action': 'reduce',
                'confidence': 0.75,
                'why': 'Overweight position with declining momentum',
                'horizon_days': 14,
                'next_check_date': '2023-11-04T12:00:00',
                'context': 'portfolio',
                'current_price': 242.50,
                'position_status': 'held',
                'position_details': {
                    'quantity': 50.0,
                    'cost_basis': 220.00,
                    'current_value': 12125.0,
                    'unrealized_pnl': 1125.0,
                    'unrealized_pnl_pct': 0.1023,
                    'position_pct_of_portfolio': 0.18,
                    'days_held': 45
                }
            }
        ]
        mock_service.return_value = mock_service_instance
        
        response = self.client.get('/api/v1/recs?scope=portfolio&include_details=true')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check portfolio-specific data
        recommendation = data['recommendations'][0]
        self.assertEqual(recommendation['context'], 'portfolio')
        self.assertEqual(recommendation['position_status'], 'held')
        
        # Check position details
        position = recommendation['position_details']
        self.assertEqual(position['quantity'], 50.0)
        self.assertEqual(position['cost_basis'], 220.00)
        self.assertAlmostEqual(position['unrealized_pnl_pct'], 0.1023, places=3)
        self.assertEqual(position['days_held'], 45)
        
        # Should call with include_details=True
        mock_service_instance.generate_portfolio_recommendations.assert_called_with(
            include_details=True, max_age_hours=24
        )
    
    def test_invalid_scope_parameter(self):
        """Test invalid scope parameter handling"""
        response = self.client.get('/api/v1/recs?scope=invalid')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Invalid scope')
        self.assertIn('watchlist', data['details'])
        self.assertIn('portfolio', data['details'])
    
    @patch.dict(os.environ, {'ENABLE_RECOS': 'false'})
    def test_feature_disabled(self):
        """Test endpoint when recommendations feature is disabled"""
        app = create_app()
        client = app.test_client()
        
        response = client.get('/api/v1/recs?scope=watchlist')
        
        self.assertEqual(response.status_code, 503)
        data = json.loads(response.data)
        self.assertEqual(data['code'], 'FEATURE_DISABLED')
    
    @patch('services.recommendations_service.RecommendationsService')
    def test_service_error_handling(self, mock_service):
        """Test handling of service errors"""
        mock_service_instance = MagicMock()
        mock_service_instance.generate_watchlist_recommendations.side_effect = Exception("Service error")
        mock_service.return_value = mock_service_instance
        
        response = self.client.get('/api/v1/recs?scope=watchlist')
        
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Internal server error')
    
    @patch('services.recommendations_service.RecommendationsService')
    def test_empty_recommendations_handling(self, mock_service):
        """Test handling of empty recommendations"""
        mock_service_instance = MagicMock()
        mock_service_instance.generate_watchlist_recommendations.return_value = []
        mock_service.return_value = mock_service_instance
        
        response = self.client.get('/api/v1/recs?scope=watchlist')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(len(data['recommendations']), 0)
        self.assertEqual(data['summary']['total'], 0)
    
    @patch('services.recommendations_service.RecommendationsService')
    def test_query_parameters(self, mock_service):
        """Test various query parameter combinations"""
        mock_service_instance = MagicMock()
        mock_service_instance.generate_watchlist_recommendations.return_value = []
        mock_service.return_value = mock_service_instance
        
        # Test with all parameters
        response = self.client.get('/api/v1/recs?scope=watchlist&include_details=true&max_age_hours=12')
        
        self.assertEqual(response.status_code, 200)
        
        # Verify service was called with correct parameters
        mock_service_instance.generate_watchlist_recommendations.assert_called_with(
            include_details=True, max_age_hours=12
        )

class TestRecommendationsService(unittest.TestCase):
    """Test cases for RecommendationsService logic"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock portfolio file
        portfolio_data = {
            'ticker': ['AAPL', 'MSFT', 'TSLA'],
            'quantity': [100, 50, 25],
            'cost_basis': [150.0, 300.0, 200.0],
            'purchase_date': ['2023-01-15', '2023-03-01', '2023-06-15']
        }
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_path = Path(self.temp_dir) / "portfolio.csv"
        portfolio_df.to_csv(portfolio_path, index=False)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('services.recommendations_service.RecommendationsService._load_portfolio_positions')
    @patch('services.recommendations_service.generate_symbol_recommendation')
    @patch('services.recommendations_service.PORTFOLIO', ['AAPL'])
    def test_portfolio_context_calculation(self, mock_generate_rec, mock_load_positions):
        """Test portfolio context calculation with real positions"""
        from services.recommendations_service import RecommendationsService
        
        # Mock portfolio position
        mock_load_positions.return_value = {
            'AAPL': {
                'quantity': 100.0,
                'cost_basis': 150.0,
                'purchase_date': '2023-01-15',
                'days_held': 300
            }
        }
        
        # Mock recommendation generation
        mock_generate_rec.return_value = {
            'symbol': 'AAPL',
            'action': 'hold',
            'confidence': 0.7,
            'why': 'Test recommendation',
            'horizon_days': 14,
            'next_check_date': '2023-11-04T12:00:00'
        }
        
        service = RecommendationsService()
        
        # Mock technical data
        with patch.object(service, '_get_technical_data') as mock_tech:
            mock_tech.return_value = {
                'price_current': 180.0,  # 20% gain from cost basis
                'dma_20': 175.0, 'dma_50': 170.0, 'dma_200': 165.0,
                'rsi': 55.0, 'macd_histogram': 0.1, 'volatility': 0.25
            }
            
            with patch.object(service, '_get_momentum_data') as mock_momentum:
                mock_momentum.return_value = {'momentum_1m': 0.05, 'momentum_3m': 0.08, 'momentum_6m': 0.12}
                
                with patch.object(service, '_get_news_data') as mock_news:
                    mock_news.return_value = {'consensus_14d': 0.15, 'trend_direction': 'rising', 'confidence': 0.8}
                    
                    with patch.object(service, '_get_earnings_data') as mock_earnings:
                        mock_earnings.return_value = {'hours_until_earnings': 120.0}
                        
                        recommendations = service.generate_portfolio_recommendations()
                        
                        self.assertEqual(len(recommendations), 1)
                        rec = recommendations[0]
                        
                        # Check position details calculation
                        position = rec['position_details']
                        self.assertEqual(position['quantity'], 100.0)
                        self.assertEqual(position['cost_basis'], 150.0)
                        self.assertEqual(position['current_value'], 18000.0)  # 100 * 180
                        self.assertEqual(position['unrealized_pnl'], 3000.0)  # 18000 - 15000
                        self.assertAlmostEqual(position['unrealized_pnl_pct'], 0.2, places=2)  # 20% gain

if __name__ == '__main__':
    unittest.main()