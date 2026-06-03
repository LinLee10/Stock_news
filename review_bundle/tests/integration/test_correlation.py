#!/usr/bin/env python3
"""
Integration tests for correlation analysis module
Tests correlation matrix computation and heatmap generation
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics.correlation import (
    CorrelationAnalyzer, 
    compute_portfolio_correlations,
    generate_correlation_heatmap,
    get_correlation_summary,
    load_correlation_from_csv
)
from config.feature_flags import feature_flags


class TestCorrelationAnalysis:
    """Test suite for correlation analysis functionality"""
    
    def setup_method(self):
        """Set up test data before each test"""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create correlated price series
        np.random.seed(42)  # For reproducible tests
        base_returns = np.random.normal(0.001, 0.02, len(dates))
        
        # AAPL - base series
        aapl_returns = base_returns + np.random.normal(0, 0.01, len(dates))
        aapl_prices = 150 * (1 + aapl_returns).cumprod()
        
        # MSFT - correlated with AAPL
        msft_returns = 0.7 * base_returns + np.random.normal(0, 0.015, len(dates))
        msft_prices = 300 * (1 + msft_returns).cumprod()
        
        # TSLA - less correlated
        tsla_returns = 0.3 * base_returns + np.random.normal(0, 0.03, len(dates))
        tsla_prices = 200 * (1 + tsla_returns).cumprod()
        
        self.price_data = {
            'AAPL': pd.DataFrame({
                'Date': dates,
                'Close': aapl_prices
            }),
            'MSFT': pd.DataFrame({
                'Date': dates, 
                'Close': msft_prices
            }),
            'TSLA': pd.DataFrame({
                'Date': dates,
                'Close': tsla_prices
            })
        }
        
        # Enable correlation for testing
        feature_flags.set_flag('enable_correlation', True)
    
    def teardown_method(self):
        """Clean up after each test"""
        # Reset feature flag
        feature_flags.set_flag('enable_correlation', False)
        
        # Clean up test files
        test_files = [
            'data/correlation.csv',
            'charts/corr_heatmap.png',
            'charts/test_heatmap.png'
        ]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_correlation_matrix_computation(self):
        """Test correlation matrix computation with valid data"""
        analyzer = CorrelationAnalyzer(lookback_days=252, min_data_points=30)
        
        correlation_matrix = analyzer.compute_correlation_matrix(self.price_data)
        
        assert correlation_matrix is not None
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.shape == (3, 3)
        assert list(correlation_matrix.index) == ['AAPL', 'MSFT', 'TSLA']
        assert list(correlation_matrix.columns) == ['AAPL', 'MSFT', 'TSLA']
        
        # Check diagonal is 1.0 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix), [1.0, 1.0, 1.0])
        
        # Check matrix is symmetric
        np.testing.assert_array_almost_equal(
            correlation_matrix.values, 
            correlation_matrix.values.T
        )
        
        # Check correlation values are in valid range [-1, 1]
        assert correlation_matrix.min().min() >= -1.0
        assert correlation_matrix.max().max() <= 1.0
    
    def test_correlation_matrix_insufficient_data(self):
        """Test correlation matrix with insufficient data"""
        analyzer = CorrelationAnalyzer(min_data_points=500)  # Require more data than available
        
        correlation_matrix = analyzer.compute_correlation_matrix(self.price_data)
        
        assert correlation_matrix is None
    
    def test_correlation_matrix_empty_data(self):
        """Test correlation matrix with empty data"""
        analyzer = CorrelationAnalyzer()
        
        correlation_matrix = analyzer.compute_correlation_matrix({})
        
        assert correlation_matrix is None
    
    def test_correlation_matrix_single_symbol(self):
        """Test correlation matrix with single symbol"""
        analyzer = CorrelationAnalyzer()
        single_symbol_data = {'AAPL': self.price_data['AAPL']}
        
        correlation_matrix = analyzer.compute_correlation_matrix(single_symbol_data)
        
        assert correlation_matrix is None  # Need at least 2 symbols
    
    def test_heatmap_generation(self):
        """Test correlation heatmap generation"""
        analyzer = CorrelationAnalyzer()
        correlation_matrix = analyzer.compute_correlation_matrix(self.price_data)
        
        assert correlation_matrix is not None
        
        # Create temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'test_heatmap.png')
            
            success = analyzer.render_heatmap(
                correlation_matrix,
                title="Test Correlation Heatmap", 
                save_path=save_path
            )
            
            assert success is True
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0  # File has content
    
    def test_heatmap_generation_no_matplotlib(self):
        """Test heatmap generation when matplotlib is not available"""
        analyzer = CorrelationAnalyzer()
        correlation_matrix = analyzer.compute_correlation_matrix(self.price_data)
        
        # Temporarily disable matplotlib
        import analytics.correlation as corr_module
        original_matplotlib = corr_module.MATPLOTLIB_AVAILABLE
        corr_module.MATPLOTLIB_AVAILABLE = False
        
        try:
            success = analyzer.render_heatmap(correlation_matrix)
            assert success is False
        finally:
            # Restore original state
            corr_module.MATPLOTLIB_AVAILABLE = original_matplotlib
    
    def test_csv_export(self):
        """Test correlation matrix CSV export"""
        analyzer = CorrelationAnalyzer()
        correlation_matrix = analyzer.compute_correlation_matrix(self.price_data)
        
        assert correlation_matrix is not None
        
        # CSV should be created automatically
        assert os.path.exists('data/correlation.csv')
        
        # Load and verify CSV content
        loaded_matrix = load_correlation_from_csv('data/correlation.csv')
        assert loaded_matrix is not None
        assert loaded_matrix.shape == correlation_matrix.shape
        
        # Check values are approximately equal
        np.testing.assert_array_almost_equal(
            loaded_matrix.values,
            correlation_matrix.values,
            decimal=6
        )
    
    def test_correlation_stats(self):
        """Test correlation statistics generation"""
        analyzer = CorrelationAnalyzer()
        correlation_matrix = analyzer.compute_correlation_matrix(self.price_data)
        
        assert correlation_matrix is not None
        
        stats = analyzer.get_correlation_stats()
        
        assert 'symbol_count' in stats
        assert 'symbols' in stats 
        assert 'data_points' in stats
        assert 'correlation_stats' in stats
        assert 'highest_correlation' in stats
        assert 'lowest_correlation' in stats
        
        assert stats['symbol_count'] == 3
        assert set(stats['symbols']) == {'AAPL', 'MSFT', 'TSLA'}
        assert stats['data_points'] > 0
        
        corr_stats = stats['correlation_stats']
        assert -1.0 <= corr_stats['min'] <= corr_stats['max'] <= 1.0
        assert corr_stats['count'] == 6  # 3x3 matrix minus 3 diagonal elements
    
    def test_symbol_correlations(self):
        """Test getting correlations for specific symbol"""
        analyzer = CorrelationAnalyzer()
        correlation_matrix = analyzer.compute_correlation_matrix(self.price_data)
        
        assert correlation_matrix is not None
        
        aapl_correlations = analyzer.get_symbol_correlations('AAPL')
        
        assert len(aapl_correlations) == 2  # MSFT and TSLA
        assert 'MSFT' in aapl_correlations
        assert 'TSLA' in aapl_correlations
        assert 'AAPL' not in aapl_correlations  # Should not include self
        
        # Values should be sorted in descending order
        values = list(aapl_correlations.values())
        assert values == sorted(values, reverse=True)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test compute_portfolio_correlations
        correlation_matrix = compute_portfolio_correlations(self.price_data, lookback_days=252)
        assert correlation_matrix is not None
        assert correlation_matrix.shape == (3, 3)
        
        # Test generate_correlation_heatmap
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'test_convenience_heatmap.png')
            success = generate_correlation_heatmap(
                self.price_data,
                title="Test Convenience Heatmap",
                save_path=save_path
            )
            assert success is True
            assert os.path.exists(save_path)
        
        # Test get_correlation_summary
        summary = get_correlation_summary(self.price_data)
        assert 'symbol_count' in summary
        assert summary['symbol_count'] == 3
    
    def test_feature_flag_disabled(self):
        """Test behavior when correlation feature flag is disabled"""
        feature_flags.set_flag('enable_correlation', False)
        
        analyzer = CorrelationAnalyzer()
        correlation_matrix = analyzer.compute_correlation_matrix(self.price_data)
        
        assert correlation_matrix is None
        
        # Heatmap should also fail
        success = analyzer.render_heatmap(None)
        assert success is False
    
    def test_lookback_period(self):
        """Test different lookback periods"""
        # Test with short lookback
        analyzer_short = CorrelationAnalyzer(lookback_days=30)
        correlation_short = analyzer_short.compute_correlation_matrix(self.price_data)
        
        # Test with long lookback
        analyzer_long = CorrelationAnalyzer(lookback_days=365)
        correlation_long = analyzer_long.compute_correlation_matrix(self.price_data)
        
        assert correlation_short is not None
        assert correlation_long is not None
        
        # Both should have same shape but potentially different values
        assert correlation_short.shape == correlation_long.shape
        
        # Data points should reflect lookback period
        assert analyzer_short.returns_data is not None
        assert analyzer_long.returns_data is not None
        assert len(analyzer_short.returns_data) <= len(analyzer_long.returns_data)
    
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        bad_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=100),
                'Price': np.random.randn(100)  # Missing 'Close' column
            })
        }
        
        analyzer = CorrelationAnalyzer()
        correlation_matrix = analyzer.compute_correlation_matrix(bad_data)
        
        assert correlation_matrix is None
    
    def test_duplicate_dates(self):
        """Test handling of duplicate dates in data"""
        # Create data with duplicate dates
        dates = pd.date_range('2023-01-01', periods=50)
        duplicate_dates = list(dates) + list(dates[:10])  # Add some duplicates
        
        duplicate_data = {
            'AAPL': pd.DataFrame({
                'Date': duplicate_dates,
                'Close': np.random.randn(len(duplicate_dates)) + 150
            }),
            'MSFT': pd.DataFrame({
                'Date': duplicate_dates,
                'Close': np.random.randn(len(duplicate_dates)) + 300
            })
        }
        
        analyzer = CorrelationAnalyzer()
        correlation_matrix = analyzer.compute_correlation_matrix(duplicate_data)
        
        # Should handle duplicates and still compute correlation
        assert correlation_matrix is not None
        assert correlation_matrix.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])