#!/usr/bin/env python3
"""
Tests for F12 Forecasting extensions
Validates baseline forecasters (ARIMA/Prophet), TimeGPT stub, and comparison table generation
"""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

# Test imports
from services.forecasting.baselines import (
    ARIMAForecaster, ProphetForecaster, BaselineForecasterRegistry,
    ForecastResult, BaselineMetrics, run_baseline_forecasts, get_baseline_summary
)
from services.forecasting.timegpt_stub import (
    TimeGPTStub, TimeGPTForecaster, TimeGPTResult, 
    run_timegpt_forecast, get_timegpt_status
)
from config.feature_flags import is_alt_forecasts_enabled, is_timegpt_stub_enabled


class TestBaselineForecasters:
    """Test ARIMA and Prophet baseline forecasters"""
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Generate realistic price series with trend and noise
        np.random.seed(42)
        base_price = 100
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 2, 100)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 30)  # Monthly seasonality
        
        prices = base_price + trend + seasonal + noise
        
        return [
            {'date': date, 'close': price} 
            for date, price in zip(dates, prices)
        ]
    
    @pytest.fixture
    def insufficient_price_data(self):
        """Generate insufficient price data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        prices = [100 + i * 0.1 for i in range(20)]
        
        return [
            {'date': date, 'close': price} 
            for date, price in zip(dates, prices)
        ]
    
    @pytest.mark.asyncio
    async def test_arima_forecaster_insufficient_data(self, insufficient_price_data):
        """Test ARIMA forecaster with insufficient data"""
        forecaster = ARIMAForecaster(forecast_horizon=7)
        result = forecaster.forecast(insufficient_price_data, "AAPL")
        
        assert isinstance(result, ForecastResult)
        assert result.model_name == "ARIMA"
        assert result.symbol == "AAPL"
        assert result.success is False
        assert "Insufficient" in result.error_message
        assert result.predictions == []
    
    @patch('services.forecasting.baselines.STATSMODELS_AVAILABLE', True)
    @pytest.mark.asyncio
    async def test_arima_forecaster_success(self, sample_price_data):
        """Test successful ARIMA forecasting"""
        forecaster = ARIMAForecaster(forecast_horizon=3)
        
        # Mock ARIMA components
        with patch('services.forecasting.baselines.ARIMA') as mock_arima:
            mock_model = MagicMock()
            mock_fitted = MagicMock()
            
            # Mock forecast results
            mock_fitted.forecast.return_value = np.array([105.0, 106.0, 107.0])
            mock_fitted.get_forecast.return_value.conf_int.return_value = pd.DataFrame({
                0: [104.0, 105.0, 106.0],
                1: [106.0, 107.0, 108.0]
            })
            mock_fitted.aic = 250.5
            
            mock_model.fit.return_value = mock_fitted
            mock_arima.return_value = mock_model
            
            result = forecaster.forecast(sample_price_data, "AAPL")
            
            assert result.success is True
            assert result.model_name == "ARIMA"
            assert len(result.predictions) == 3
            assert result.predictions == [105.0, 106.0, 107.0]
            assert len(result.forecast_dates) == 3
            assert result.confidence_intervals is not None
            assert len(result.confidence_intervals) == 3
    
    @patch('services.forecasting.baselines.STATSMODELS_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_arima_forecaster_no_statsmodels(self, sample_price_data):
        """Test ARIMA forecaster when statsmodels not available"""
        forecaster = ARIMAForecaster()
        result = forecaster.forecast(sample_price_data, "AAPL")
        
        assert result.success is False
        assert "statsmodels not available" in result.error_message
    
    @patch('services.forecasting.baselines.PROPHET_AVAILABLE', True)
    @pytest.mark.asyncio 
    async def test_prophet_forecaster_success(self, sample_price_data):
        """Test successful Prophet forecasting"""
        forecaster = ProphetForecaster(forecast_horizon=3)
        
        # Mock Prophet components
        with patch('services.forecasting.baselines.Prophet') as mock_prophet_class:
            mock_prophet = MagicMock()
            mock_prophet_class.return_value = mock_prophet
            
            # Mock forecast DataFrame
            forecast_df = pd.DataFrame({
                'ds': pd.date_range(start='2023-04-11', periods=3, freq='D'),
                'yhat': [105.0, 106.0, 107.0],
                'yhat_lower': [104.0, 105.0, 106.0],
                'yhat_upper': [106.0, 107.0, 108.0]
            })
            
            # Mock future dataframe creation
            future_df = pd.DataFrame({
                'ds': pd.date_range(start='2023-01-01', periods=103, freq='D')
            })
            
            mock_prophet.make_future_dataframe.return_value = future_df
            mock_prophet.predict.return_value = pd.concat([
                pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=100),
                             'yhat': [100] * 100, 'yhat_lower': [99] * 100, 'yhat_upper': [101] * 100}),
                forecast_df
            ])
            
            result = forecaster.forecast(sample_price_data, "AAPL")
            
            assert result.success is True
            assert result.model_name == "Prophet"
            assert len(result.predictions) == 3
            assert result.predictions == [105.0, 106.0, 107.0]
            assert len(result.forecast_dates) == 3
            assert result.confidence_intervals is not None
    
    @patch('services.forecasting.baselines.PROPHET_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_prophet_forecaster_no_prophet(self, sample_price_data):
        """Test Prophet forecaster when prophet not available"""
        forecaster = ProphetForecaster()
        result = forecaster.forecast(sample_price_data, "AAPL")
        
        assert result.success is False
        assert "prophet not available" in result.error_message


class TestBaselineForecasterRegistry:
    """Test baseline forecaster registry"""
    
    @pytest.fixture
    def sample_data_by_symbol(self):
        """Sample data for multiple symbols"""
        return {
            'AAPL': [{'date': datetime(2023, 1, 1) + timedelta(days=i), 'close': 100 + i * 0.5} for i in range(50)],
            'MSFT': [{'date': datetime(2023, 1, 1) + timedelta(days=i), 'close': 200 + i * 0.3} for i in range(50)]
        }
    
    @patch('config.feature_flags.is_alt_forecasts_enabled', return_value=False)
    @pytest.mark.asyncio
    async def test_registry_disabled(self, mock_flag, sample_data_by_symbol):
        """Test registry returns empty when feature disabled"""
        registry = BaselineForecasterRegistry()
        results = registry.run_baselines(sample_data_by_symbol)
        
        assert results == {}
    
    @patch('config.feature_flags.is_alt_forecasts_enabled', return_value=True)
    @pytest.mark.asyncio
    async def test_registry_enabled(self, mock_flag, sample_data_by_symbol):
        """Test registry runs forecasters when enabled"""
        with patch('services.forecasting.baselines.STATSMODELS_AVAILABLE', False):
            with patch('services.forecasting.baselines.PROPHET_AVAILABLE', False):
                registry = BaselineForecasterRegistry()
                results = registry.run_baselines(sample_data_by_symbol)
                
                # Should have results for both symbols
                assert 'AAPL' in results
                assert 'MSFT' in results
                
                # Each symbol should have 2 forecasters (ARIMA, Prophet)
                assert len(results['AAPL']) == 2
                assert len(results['MSFT']) == 2
                
                # Results should be failed due to missing dependencies
                for symbol_results in results.values():
                    for result in symbol_results:
                        assert isinstance(result, ForecastResult)
                        assert result.success is False
    
    @pytest.mark.asyncio
    async def test_get_baseline_metrics(self):
        """Test baseline metrics extraction"""
        # Create mock results
        results = {
            'AAPL': [
                ForecastResult('ARIMA', 'AAPL', [105, 106], [], mae=2.5, rmse=3.2, runtime_ms=1500, success=True),
                ForecastResult('Prophet', 'AAPL', [104, 107], [], mae=2.8, rmse=3.5, runtime_ms=2200, success=True)
            ]
        }
        
        registry = BaselineForecasterRegistry()
        metrics = registry.get_baseline_metrics(results)
        
        assert 'AAPL' in metrics
        
        aapl_metrics = metrics['AAPL']
        assert isinstance(aapl_metrics, BaselineMetrics)
        assert aapl_metrics.arima_success is True
        assert aapl_metrics.prophet_success is True
        assert aapl_metrics.arima_mae == 2.5
        assert aapl_metrics.prophet_rmse == 3.5


class TestTimeGPTStub:
    """Test TimeGPT stub implementation"""
    
    @pytest.fixture
    def timegpt_stub(self):
        """Create TimeGPT stub for testing"""
        return TimeGPTStub()
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data with enough points for TimeGPT"""
        return [
            {'date': datetime(2023, 1, 1) + timedelta(days=i), 'close': 100 + i * 0.1} 
            for i in range(60)
        ]
    
    @patch('config.feature_flags.is_timegpt_stub_enabled', return_value=False)
    @pytest.mark.asyncio
    async def test_timegpt_disabled(self, mock_flag, timegpt_stub, sample_prices):
        """Test TimeGPT stub when feature disabled"""
        result = timegpt_stub.forecast(sample_prices, "AAPL")
        
        assert isinstance(result, TimeGPTResult)
        assert result.success is False
        assert "disabled by feature flag" in result.error_message
    
    @patch('config.feature_flags.is_timegpt_stub_enabled', return_value=True)
    @pytest.mark.asyncio
    async def test_timegpt_insufficient_data(self, mock_flag, timegpt_stub):
        """Test TimeGPT with insufficient data"""
        insufficient_data = [
            {'date': datetime(2023, 1, 1) + timedelta(days=i), 'close': 100 + i} 
            for i in range(30)
        ]
        
        result = timegpt_stub.forecast(insufficient_data, "AAPL")
        
        assert result.success is False
        assert "Insufficient data" in result.error_message
    
    @patch('config.feature_flags.is_timegpt_stub_enabled', return_value=True)
    @pytest.mark.asyncio
    async def test_timegpt_success(self, mock_flag, timegpt_stub, sample_prices):
        """Test successful TimeGPT forecast"""
        # Set very short simulated latency for testing
        timegpt_stub.simulated_latency_ms = 10
        
        result = timegpt_stub.forecast(sample_prices, "AAPL")
        
        assert result.success is True
        assert result.symbol == "AAPL"
        assert len(result.predictions) == 7  # Default forecast horizon
        assert len(result.forecast_dates) == 7
        assert all(isinstance(pred, float) for pred in result.predictions)
        assert result.api_credits_used == 1
        assert result.mae is not None
        assert result.rmse is not None
    
    @pytest.mark.asyncio
    async def test_timegpt_rate_limiting(self, timegpt_stub, sample_prices):
        """Test TimeGPT rate limiting"""
        # Fill up the rate limit
        timegpt_stub.call_history = [time.time()] * timegpt_stub.rate_limit_calls
        
        with patch('config.feature_flags.is_timegpt_stub_enabled', return_value=True):
            result = timegpt_stub.forecast(sample_prices, "AAPL")
        
        assert result.success is False
        assert "Rate limit exceeded" in result.error_message
    
    @pytest.mark.asyncio
    async def test_timegpt_api_info(self, timegpt_stub):
        """Test TimeGPT API info"""
        info = timegpt_stub.get_api_info()
        
        assert info['service'] == 'TimeGPT (Stub)'
        assert 'rate_limit' in info
        assert 'settings' in info
        assert info['rate_limit']['calls_per_hour'] == 100
    
    @pytest.mark.asyncio
    async def test_timegpt_reset_rate_limit(self, timegpt_stub):
        """Test rate limit reset functionality"""
        # Add some call history
        timegpt_stub.call_history = [time.time()] * 50
        assert len(timegpt_stub.call_history) == 50
        
        timegpt_stub.reset_rate_limit()
        assert len(timegpt_stub.call_history) == 0


class TestTimeGPTForecaster:
    """Test TimeGPT forecaster high-level interface"""
    
    @pytest.fixture
    def timegpt_forecaster(self):
        """Create TimeGPT forecaster"""
        return TimeGPTForecaster()
    
    @patch('config.feature_flags.is_timegpt_stub_enabled', return_value=False)
    @pytest.mark.asyncio
    async def test_timegpt_forecaster_disabled(self, mock_flag, timegpt_forecaster):
        """Test TimeGPT forecaster when disabled"""
        prices_by_symbol = {
            'AAPL': [{'date': datetime(2023, 1, 1), 'close': 100}]
        }
        
        results = timegpt_forecaster.run_timegpt_forecasts(prices_by_symbol)
        assert results == {}
    
    @patch('config.feature_flags.is_timegpt_stub_enabled', return_value=True)
    @pytest.mark.asyncio
    async def test_timegpt_forecaster_enabled(self, mock_flag, timegpt_forecaster):
        """Test TimeGPT forecaster when enabled"""
        # Reduce latency for testing
        timegpt_forecaster.timegpt.simulated_latency_ms = 10
        
        prices_by_symbol = {
            'AAPL': [
                {'date': datetime(2023, 1, 1) + timedelta(days=i), 'close': 100 + i * 0.1}
                for i in range(60)
            ]
        }
        
        results = timegpt_forecaster.run_timegpt_forecasts(prices_by_symbol)
        
        assert 'AAPL' in results
        assert isinstance(results['AAPL'], TimeGPTResult)


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for convenience function testing"""
        return {
            'AAPL': [
                {'date': datetime(2023, 1, 1) + timedelta(days=i), 'close': 100 + i}
                for i in range(50)
            ]
        }
    
    @patch('config.feature_flags.is_alt_forecasts_enabled', return_value=False)
    @pytest.mark.asyncio
    async def test_run_baseline_forecasts_disabled(self, mock_flag, sample_data):
        """Test run_baseline_forecasts when disabled"""
        results = run_baseline_forecasts(sample_data)
        assert results == {}
    
    @patch('config.feature_flags.is_timegpt_stub_enabled', return_value=False)
    @pytest.mark.asyncio
    async def test_run_timegpt_forecast_disabled(self, mock_flag, sample_data):
        """Test run_timegpt_forecast when disabled"""
        results = run_timegpt_forecast(sample_data)
        assert results == {}
    
    @pytest.mark.asyncio
    async def test_get_baseline_summary(self):
        """Test baseline forecast summary generation"""
        # Create mock results
        results = {
            'AAPL': [
                ForecastResult('ARIMA', 'AAPL', [105, 106], [], mae=2.5, rmse=3.2, runtime_ms=1500, success=True),
                ForecastResult('Prophet', 'AAPL', [], [], success=False, error_message='Test error', runtime_ms=100)
            ],
            'MSFT': [
                ForecastResult('ARIMA', 'MSFT', [205, 206], [], mae=3.5, rmse=4.2, runtime_ms=1800, success=True)
            ]
        }
        
        summary = get_baseline_summary(results)
        
        assert summary['total_symbols'] == 2
        assert summary['successful_forecasts'] == 2
        assert summary['failed_forecasts'] == 1
        assert 'ARIMA' in summary['models_run']
        assert 'Prophet' in summary['models_run']
        
        # Check averages
        assert summary['avg_mae']['ARIMA'] == 3.0  # (2.5 + 3.5) / 2
        assert summary['avg_runtime_ms']['ARIMA'] == 1650  # (1500 + 1800) / 2
    
    @pytest.mark.asyncio
    async def test_get_timegpt_status(self):
        """Test TimeGPT status retrieval"""
        status = get_timegpt_status()
        
        assert 'service' in status
        assert status['service'] == 'TimeGPT (Stub)'
        assert 'enabled' in status
        assert 'rate_limit' in status


class TestPredictionIntegration:
    """Test integration with prediction.py"""
    
    @pytest.mark.asyncio
    async def test_forecast_comparison_functions_exist(self):
        """Test that F12 functions are available in prediction module"""
        from prediction import get_forecast_comparison_summary, _save_forecast_comparison_table, _generate_forecast_comparison
        
        # Functions should be importable
        assert callable(get_forecast_comparison_summary)
        assert callable(_save_forecast_comparison_table)  
        assert callable(_generate_forecast_comparison)
    
    @pytest.mark.asyncio
    async def test_generate_forecast_comparison(self):
        """Test forecast comparison data generation"""
        from prediction import _generate_forecast_comparison
        
        # Mock baseline results
        baseline_results = [
            ForecastResult('ARIMA', 'AAPL', [105.0, 106.0], [], mae=2.5, rmse=3.2, runtime_ms=1500, success=True)
        ]
        
        # Mock TimeGPT result
        timegpt_result = TimeGPTResult('AAPL', [104.0, 107.0], [], success=True, mae=2.8, rmse=3.5, runtime_ms=2200, api_credits_used=1)
        
        comparison_data = _generate_forecast_comparison(
            'AAPL', [103.0, 105.0], 0.85, baseline_results, timegpt_result
        )
        
        assert len(comparison_data) == 6  # 2 RF/XGB + 2 ARIMA + 2 TimeGPT
        
        # Check RF/XGB entries
        rf_entries = [d for d in comparison_data if d['model'] == 'RF_XGB_Ensemble']
        assert len(rf_entries) == 2
        assert rf_entries[0]['prediction'] == 103.0
        assert rf_entries[1]['prediction'] == 105.0
        
        # Check ARIMA entries
        arima_entries = [d for d in comparison_data if d['model'] == 'ARIMA']
        assert len(arima_entries) == 2
        assert arima_entries[0]['mae'] == 2.5
        
        # Check TimeGPT entries
        timegpt_entries = [d for d in comparison_data if d['model'] == 'TimeGPT_Stub']
        assert len(timegpt_entries) == 2
        assert timegpt_entries[0]['api_credits_used'] == 1
    
    @pytest.mark.asyncio
    async def test_save_forecast_comparison_table(self):
        """Test saving comparison table to CSV"""
        from prediction import _save_forecast_comparison_table
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the data directory
            with patch('prediction.os.makedirs') as mock_makedirs:
                comparison_file = os.path.join(temp_dir, 'forecast_comparison.csv')
                
                with patch('prediction.os.path.exists', return_value=False):
                    with patch('prediction.pd.DataFrame.to_csv') as mock_to_csv:
                        comparison_data = [
                            {'symbol': 'AAPL', 'model': 'ARIMA', 'prediction': 105.0, 'mae': 2.5}
                        ]
                        
                        _save_forecast_comparison_table(comparison_data)
                        
                        mock_makedirs.assert_called_once_with('data', exist_ok=True)
                        mock_to_csv.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_forecast_comparison_summary_no_file(self):
        """Test getting summary when no comparison file exists"""
        from prediction import get_forecast_comparison_summary
        
        with patch('prediction.os.path.exists', return_value=False):
            summary = get_forecast_comparison_summary()
            assert 'error' in summary
            assert 'No comparison data available' in summary['error']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])