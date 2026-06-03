#!/usr/bin/env python3
"""
Unit tests for symbol intake endpoint and services
"""
import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set up test environment before imports
import os
os.environ['ENABLE_SYMBOL_INTAKE'] = 'true'
os.environ['ENABLE_API_ENDPOINTS'] = 'true'

from api.app import create_app
from services.symbol_intake import SymbolIntakeService
from services.job_queue import JobQueue

class TestSymbolIntakeEndpoint(unittest.TestCase):
    """Test cases for /symbols/intake endpoint"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = create_app()
        self.client = self.app.test_client()
        
        # Create temporary data directory
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_path = Path("data")
        
        # Mock data directory
        self.patcher = patch('services.symbol_intake.Path')
        mock_path = self.patcher.start()
        mock_path.return_value = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        self.patcher.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_symbol_intake_valid_request(self):
        """Test valid symbol intake request"""
        request_data = {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.'
        }
        
        with patch('services.job_queue.JobQueue') as mock_queue:
            mock_queue_instance = MagicMock()
            mock_queue_instance.get_queue_size.return_value = 5
            mock_queue_instance.enqueue_job.return_value = True
            mock_queue.return_value = mock_queue_instance
            
            response = self.client.post(
                '/api/v1/symbols/intake',
                data=json.dumps(request_data),
                content_type='application/json'
            )
        
        self.assertEqual(response.status_code, 202)
        data = json.loads(response.data)
        
        self.assertIn('job_id', data)
        self.assertIn('symbol_id', data)
        self.assertEqual(data['status'], 'queued')
        self.assertIn('estimated_completion_seconds', data)
    
    def test_symbol_intake_invalid_ticker(self):
        """Test invalid ticker format"""
        request_data = {
            'ticker': 'INVALID_TICKER_TOO_LONG',
            'company_name': 'Invalid Company'
        }
        
        response = self.client.post(
            '/api/v1/symbols/intake',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Invalid ticker format')
    
    def test_symbol_intake_missing_fields(self):
        """Test request with missing required fields"""
        request_data = {
            'ticker': 'AAPL'
            # Missing company_name
        }
        
        response = self.client.post(
            '/api/v1/symbols/intake',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Validation failed')
    
    def test_symbol_intake_duplicate_ticker(self):
        """Test duplicate ticker handling"""
        request_data = {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.'
        }
        
        with patch('services.symbol_intake.SymbolIntakeService') as mock_service:
            mock_instance = MagicMock()
            mock_instance.get_symbol_by_ticker.return_value = {
                'ticker': 'AAPL',
                'intake_status': 'completed',
                'last_job_id': 'existing-job-123',
                'symbol_id': 'sym_AAPL_123'
            }
            mock_service.return_value = mock_instance
            
            response = self.client.post(
                '/api/v1/symbols/intake',
                data=json.dumps(request_data),
                content_type='application/json'
            )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'already_exists')
    
    @patch.dict(os.environ, {'ENABLE_SYMBOL_INTAKE': 'false'})
    def test_symbol_intake_feature_disabled(self):
        """Test endpoint when feature is disabled"""
        # Reload app with disabled feature
        app = create_app()
        client = app.test_client()
        
        request_data = {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.'
        }
        
        response = client.post(
            '/api/v1/symbols/intake',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 503)
        data = json.loads(response.data)
        self.assertEqual(data['code'], 'FEATURE_DISABLED')
    
    def test_symbol_intake_status_endpoint(self):
        """Test GET /symbols/{symbol}/intake_status endpoint"""
        with patch('services.symbol_intake.SymbolIntakeService') as mock_service:
            with patch('services.job_queue.JobQueue') as mock_queue:
                # Mock services
                mock_service_instance = MagicMock()
                mock_service_instance.get_symbol_by_ticker.return_value = {
                    'ticker': 'AAPL',
                    'last_job_id': 'test-job-123'
                }
                mock_service.return_value = mock_service_instance
                
                mock_queue_instance = MagicMock()
                mock_queue_instance.get_job_status.return_value = {
                    'job_id': 'test-job-123',
                    'status': 'processing',
                    'created_at': 1634567890.0,
                    'updated_at': 1634567950.0
                }
                mock_queue.return_value = mock_queue_instance
                
                response = self.client.get('/api/v1/symbols/AAPL/intake_status')
                
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                
                self.assertEqual(data['symbol'], 'AAPL')
                self.assertEqual(data['state'], 'processing')
                self.assertEqual(data['percent'], 50)
                self.assertEqual(data['last_step'], 'Fetching price data and news')
    
    def test_symbol_snapshot_endpoint(self):
        """Test GET /symbols/{symbol}/snapshot endpoint"""
        with patch('services.symbol_snapshot.SymbolSnapshotService') as mock_service:
            # Mock snapshot service
            mock_service_instance = MagicMock()
            mock_service_instance.generate_snapshot.return_value = {
                'symbol': 'AAPL',
                'generated_at': '2023-10-18T12:00:00',
                'price_card': {
                    'current_price': 150.25,
                    'currency': 'USD',
                    'price_changes': {
                        '1d': {'change': 2.50, 'change_pct': 1.69}
                    }
                },
                'chart_data': [
                    {'date': '2023-10-17', 'close': 147.75, 'timestamp': 1697587200},
                    {'date': '2023-10-18', 'close': 150.25, 'timestamp': 1697673600}
                ],
                'headlines': [
                    {
                        'title': 'Apple Reports Strong Q4 Earnings',
                        'url': 'https://example.com/news/1',
                        'published_date': '2023-10-18T10:00:00',
                        'sentiment': {
                            'score': 0.3,
                            'confidence': 0.85,
                            'label': 'positive'
                        },
                        'source': 'example'
                    }
                ],
                'earnings': {
                    'next_earnings_date': '2023-11-02',
                    'confirmed': True,
                    'implied_move_pct': 4.2,
                    'days_until': 15
                }
            }
            mock_service.return_value = mock_service_instance
            
            response = self.client.get('/api/v1/symbols/AAPL/snapshot')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            self.assertEqual(data['symbol'], 'AAPL')
            self.assertIn('price_card', data)
            self.assertIn('chart_data', data)
            self.assertIn('headlines', data)
            self.assertIn('earnings', data)
            self.assertEqual(data['price_card']['current_price'], 150.25)
            self.assertEqual(len(data['headlines']), 1)
            self.assertEqual(data['headlines'][0]['sentiment']['label'], 'positive')

class TestSymbolIntakeService(unittest.TestCase):
    """Test cases for SymbolIntakeService"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.service = SymbolIntakeService()
        
        # Mock the data path
        self.service.data_path = Path(self.temp_dir)
        self.service.registry_path = self.service.data_path / "tickers_registry.csv"
        self.service._ensure_registry_exists()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_upsert_new_symbol(self):
        """Test inserting a new symbol"""
        result = self.service.upsert_symbol(
            ticker='TSLA',
            company_name='Tesla Inc.',
            symbol_id='sym_TSLA_123',
            intake_status='queued',
            job_id='job-456'
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result['ticker'], 'TSLA')
        self.assertEqual(result['company_name'], 'Tesla Inc.')
        self.assertEqual(result['intake_status'], 'queued')
    
    def test_upsert_existing_symbol(self):
        """Test updating an existing symbol"""
        # Insert initial record
        self.service.upsert_symbol(
            ticker='MSFT',
            company_name='Microsoft Corp.',
            symbol_id='sym_MSFT_123',
            intake_status='queued'
        )
        
        # Update the same ticker
        result = self.service.upsert_symbol(
            ticker='MSFT',
            company_name='Microsoft Corporation',
            symbol_id='sym_MSFT_456',
            intake_status='processing'
        )
        
        self.assertEqual(result['company_name'], 'Microsoft Corporation')
        self.assertEqual(result['intake_status'], 'processing')
    
    def test_get_symbol_by_ticker(self):
        """Test retrieving symbol by ticker"""
        # Insert a symbol
        self.service.upsert_symbol(
            ticker='GOOGL',
            company_name='Alphabet Inc.',
            symbol_id='sym_GOOGL_789'
        )
        
        # Retrieve it
        result = self.service.get_symbol_by_ticker('GOOGL')
        self.assertIsNotNone(result)
        self.assertEqual(result['ticker'], 'GOOGL')
        
        # Test case insensitivity
        result_lower = self.service.get_symbol_by_ticker('googl')
        self.assertIsNotNone(result_lower)
        self.assertEqual(result_lower['ticker'], 'GOOGL')
        
        # Test non-existent ticker
        result_none = self.service.get_symbol_by_ticker('NONEXISTENT')
        self.assertIsNone(result_none)

if __name__ == '__main__':
    unittest.main()