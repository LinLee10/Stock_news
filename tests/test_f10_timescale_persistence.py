#!/usr/bin/env python3
"""
Tests for F10 TimescaleDB persistence feature
Validates schema generation, adapter functionality, and feature flag behavior
"""

import pytest
import os
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

# Test imports
from services.timeseries_database_schema import (
    create_hypertables, get_all_ddl, get_table_names,
    _create_prices_table, _create_sentiment_table, _create_predictions_table
)
from services.db_adapter import (
    DatabaseAdapter, PriceRecord, SentimentRecord, PredictionRecord,
    persist_prices, persist_sentiment, persist_predictions, initialize_db_adapter,
    db_adapter
)
from config.feature_flags import is_timescale_persistence_enabled


class TestTimescaleDatabaseSchema:
    """Test TimescaleDB schema generation"""
    
    def test_create_hypertables_returns_dict(self):
        """Test that create_hypertables returns expected structure"""
        hypertables = create_hypertables()
        
        assert isinstance(hypertables, dict)
        assert len(hypertables) == 3
        assert 'prices' in hypertables
        assert 'sentiment' in hypertables
        assert 'predictions' in hypertables
        
        for table_name, ddl in hypertables.items():
            assert isinstance(ddl, str)
            assert len(ddl) > 0
            assert 'CREATE TABLE' in ddl
            assert 'create_hypertable' in ddl
    
    def test_prices_table_ddl(self):
        """Test prices table DDL structure"""
        ddl = _create_prices_table()
        
        # Check required elements
        assert 'CREATE TABLE IF NOT EXISTS prices' in ddl
        assert 'symbol VARCHAR(10) NOT NULL' in ddl
        assert 'date_recorded TIMESTAMPTZ NOT NULL' in ddl
        assert 'PRIMARY KEY (symbol, date_recorded)' in ddl
        assert "create_hypertable('prices', 'date_recorded'" in ddl
        assert 'chunk_time_interval => INTERVAL \'1 month\'' in ddl
        
        # Check price columns
        required_columns = [
            'open_price DECIMAL(12,4)',
            'high_price DECIMAL(12,4)', 
            'low_price DECIMAL(12,4)',
            'close_price DECIMAL(12,4)',
            'volume BIGINT',
            'adjusted_close DECIMAL(12,4)'
        ]
        
        for column in required_columns:
            assert column in ddl
    
    def test_sentiment_table_ddl(self):
        """Test sentiment table DDL structure"""
        ddl = _create_sentiment_table()
        
        # Check required elements
        assert 'CREATE TABLE IF NOT EXISTS sentiment' in ddl
        assert 'symbol VARCHAR(10) NOT NULL' in ddl
        assert 'date_recorded TIMESTAMPTZ NOT NULL' in ddl
        assert 'PRIMARY KEY (symbol, date_recorded)' in ddl
        assert "create_hypertable('sentiment', 'date_recorded'" in ddl
        assert 'chunk_time_interval => INTERVAL \'1 week\'' in ddl
        
        # Check sentiment columns
        required_columns = [
            'sentiment_score DECIMAL(5,4)',
            'sentiment_label VARCHAR(20)',
            'confidence_score DECIMAL(5,4)',
            'article_count INTEGER DEFAULT 1',
            'positive_mentions INTEGER DEFAULT 0',
            'negative_mentions INTEGER DEFAULT 0',
            'neutral_mentions INTEGER DEFAULT 0'
        ]
        
        for column in required_columns:
            assert column in ddl
    
    def test_predictions_table_ddl(self):
        """Test predictions table DDL structure"""
        ddl = _create_predictions_table()
        
        # Check required elements
        assert 'CREATE TABLE IF NOT EXISTS predictions' in ddl
        assert 'symbol VARCHAR(10) NOT NULL' in ddl
        assert 'date_recorded TIMESTAMPTZ NOT NULL' in ddl
        assert 'prediction_type VARCHAR(50) NOT NULL' in ddl
        assert 'PRIMARY KEY (symbol, date_recorded, prediction_type)' in ddl
        assert "create_hypertable('predictions', 'date_recorded'" in ddl
        assert 'chunk_time_interval => INTERVAL \'1 month\'' in ddl
        
        # Check prediction columns
        required_columns = [
            'predicted_value DECIMAL(12,4)',
            'confidence_score DECIMAL(5,4)',
            'model_version VARCHAR(20)',
            'features_used TEXT[]',
            'prediction_horizon_days INTEGER',
            'actual_value DECIMAL(12,4)',
            'error_magnitude DECIMAL(12,4)'
        ]
        
        for column in required_columns:
            assert column in ddl
    
    def test_get_all_ddl(self):
        """Test combined DDL generation"""
        all_ddl = get_all_ddl()
        
        assert isinstance(all_ddl, str)
        assert len(all_ddl) > 0
        
        # Should contain all three tables
        assert 'CREATE TABLE IF NOT EXISTS prices' in all_ddl
        assert 'CREATE TABLE IF NOT EXISTS sentiment' in all_ddl
        assert 'CREATE TABLE IF NOT EXISTS predictions' in all_ddl
        
        # Should have proper separators
        assert all_ddl.count('\n\n') >= 2  # At least 2 separators for 3 sections
    
    def test_get_table_names(self):
        """Test table names extraction"""
        names = get_table_names()
        
        assert isinstance(names, list)
        assert len(names) == 3
        assert 'prices' in names
        assert 'sentiment' in names
        assert 'predictions' in names


class TestDatabaseAdapter:
    """Test DatabaseAdapter functionality"""
    
    @pytest.fixture
    def adapter(self):
        """Create a fresh DatabaseAdapter instance for testing"""
        return DatabaseAdapter()
    
    @pytest.fixture
    def sample_price_records(self):
        """Sample price records for testing"""
        base_date = datetime(2023, 1, 1)
        return [
            PriceRecord(
                symbol="AAPL",
                date_recorded=base_date + timedelta(days=i),
                open_price=150.0 + i,
                high_price=155.0 + i,
                low_price=149.0 + i,
                close_price=152.0 + i,
                volume=1000000,
                adjusted_close=152.0 + i
            )
            for i in range(3)
        ]
    
    @pytest.fixture
    def sample_sentiment_records(self):
        """Sample sentiment records for testing"""
        base_date = datetime(2023, 1, 1)
        return [
            SentimentRecord(
                symbol="AAPL",
                date_recorded=base_date + timedelta(days=i),
                sentiment_score=0.7 + i * 0.1,
                sentiment_label="positive",
                confidence_score=0.9,
                article_count=5 + i,
                positive_mentions=3,
                negative_mentions=1,
                neutral_mentions=1,
                data_source="test"
            )
            for i in range(3)
        ]
    
    @pytest.fixture
    def sample_prediction_records(self):
        """Sample prediction records for testing"""
        base_date = datetime(2023, 1, 1)
        return [
            PredictionRecord(
                symbol="AAPL",
                date_recorded=base_date + timedelta(days=i),
                prediction_type="price_movement",
                predicted_value=160.0 + i,
                confidence_score=0.8,
                model_version="v1.0",
                features_used=["price", "volume"],
                prediction_horizon_days=7,
                actual_value=None,
                error_magnitude=None
            )
            for i in range(3)
        ]
    
    @patch('services.db_adapter.is_timescale_persistence_enabled')
    def test_no_op_when_flag_disabled(self, mock_flag, adapter):
        """Test that adapter is no-op when feature flag is disabled"""
        mock_flag.return_value = False
        
        # Should return False and not initialize
        result = asyncio.run(adapter.initialize())
        assert result is False
        assert adapter._initialized is False
        assert adapter.pool is None
    
    @patch('services.db_adapter.is_timescale_persistence_enabled')
    @patch.dict(os.environ, {}, clear=True)
    def test_no_op_when_no_dsn(self, mock_flag, adapter):
        """Test that adapter is no-op when no PG_DSN is configured"""
        mock_flag.return_value = True
        
        # Should return False when no DSN
        result = asyncio.run(adapter.initialize())
        assert result is False
        assert adapter._initialized is False
        assert adapter.pool is None
    
    @patch('services.db_adapter.is_timescale_persistence_enabled')
    @patch('services.db_adapter.ASYNCPG_AVAILABLE', False)
    @patch.dict(os.environ, {'PG_DSN': 'postgres://test'})
    def test_no_op_when_asyncpg_unavailable(self, mock_flag, adapter):
        """Test that adapter is no-op when asyncpg is not available"""
        mock_flag.return_value = True
        
        # Should return False when asyncpg not available
        result = asyncio.run(adapter.initialize())
        assert result is False
        assert adapter._initialized is False
        assert adapter.pool is None
    
    @patch('services.db_adapter.is_timescale_persistence_enabled')
    @patch('services.db_adapter.ASYNCPG_AVAILABLE', True)
    @patch('services.db_adapter.asyncpg')
    @patch.dict(os.environ, {'PG_DSN': 'postgres://localhost/test'})
    def test_successful_initialization(self, mock_asyncpg, mock_flag, adapter):
        """Test successful initialization with all requirements met"""
        mock_flag.return_value = True
        
        # Mock asyncpg pool and connection
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_asyncpg.create_pool.return_value = mock_pool
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None
        
        # Should initialize successfully
        result = asyncio.run(adapter.initialize())
        assert result is True
        assert adapter._initialized is True
        assert adapter.pool is not None
        
        # Should have called pool creation and schema setup
        mock_asyncpg.create_pool.assert_called_once()
        mock_conn.execute.assert_called()
    
    def test_no_op_batch_operations_when_not_initialized(self, adapter, sample_price_records, 
                                                         sample_sentiment_records, sample_prediction_records):
        """Test that batch operations return 0 when not initialized"""
        # Adapter not initialized
        assert adapter._initialized is False
        
        # All operations should return 0
        result = asyncio.run(adapter.insert_prices_batch(sample_price_records))
        assert result == 0
        
        result = asyncio.run(adapter.insert_sentiment_batch(sample_sentiment_records))
        assert result == 0
        
        result = asyncio.run(adapter.insert_predictions_batch(sample_prediction_records))
        assert result == 0
    
    def test_no_op_queries_when_not_initialized(self, adapter):
        """Test that queries return empty results when not initialized"""
        # Adapter not initialized
        assert adapter._initialized is False
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        # All queries should return empty lists
        result = asyncio.run(adapter.query_prices("AAPL", start_date, end_date))
        assert result == []
        
        result = asyncio.run(adapter.query_sentiment("AAPL", start_date, end_date))
        assert result == []
        
        result = asyncio.run(adapter.query_predictions("AAPL"))
        assert result == []
    
    @patch('services.db_adapter.is_timescale_persistence_enabled')
    def test_health_check_disabled(self, mock_flag, adapter):
        """Test health check when disabled"""
        mock_flag.return_value = False
        
        health = asyncio.run(adapter.health_check())
        
        assert health['enabled'] is False
        assert health['initialized'] is False
        assert health['connection_healthy'] is False
    
    @patch('services.db_adapter.is_timescale_persistence_enabled')
    @patch.dict(os.environ, {'PG_DSN': 'postgres://localhost/test'})
    def test_health_check_configured(self, mock_flag, adapter):
        """Test health check when properly configured"""
        mock_flag.return_value = True
        
        health = asyncio.run(adapter.health_check())
        
        assert health['enabled'] is True
        assert health['configured'] is True
        assert health['initialized'] is False  # Not initialized yet
        assert health['connection_healthy'] is False


class TestConvenienceFunctions:
    """Test convenience functions and global adapter"""
    
    @pytest.fixture(autouse=True)
    def reset_global_adapter(self):
        """Reset global adapter state before each test"""
        db_adapter._initialized = False
        db_adapter.pool = None
        yield
        # Cleanup after test
        asyncio.run(db_adapter.cleanup())
    
    @patch('services.db_adapter.is_timescale_persistence_enabled')
    def test_convenience_functions_no_op(self, mock_flag):
        """Test that convenience functions are no-op when disabled"""
        mock_flag.return_value = False
        
        # Create sample data
        prices = [PriceRecord("AAPL", datetime.now(), open_price=150.0)]
        sentiments = [SentimentRecord("AAPL", datetime.now(), sentiment_score=0.7)]
        predictions = [PredictionRecord("AAPL", datetime.now(), "test", predicted_value=160.0)]
        
        # All should return 0 (no-op)
        result = asyncio.run(persist_prices(prices))
        assert result == 0
        
        result = asyncio.run(persist_sentiment(sentiments))
        assert result == 0
        
        result = asyncio.run(persist_predictions(predictions))
        assert result == 0
    
    @patch('services.db_adapter.is_timescale_persistence_enabled')
    def test_initialize_db_adapter_disabled(self, mock_flag):
        """Test initialize_db_adapter when disabled"""
        mock_flag.return_value = False
        
        result = asyncio.run(initialize_db_adapter())
        assert result is False
    
    @patch('services.db_adapter.is_timescale_persistence_enabled')
    @patch('services.db_adapter.ASYNCPG_AVAILABLE', True)
    @patch('services.db_adapter.asyncpg')
    @patch.dict(os.environ, {'PG_DSN': 'postgres://localhost/test'})
    def test_initialize_db_adapter_success(self, mock_asyncpg, mock_flag):
        """Test successful initialize_db_adapter"""
        mock_flag.return_value = True
        
        # Mock asyncpg components
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_asyncpg.create_pool.return_value = mock_pool
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None
        
        result = asyncio.run(initialize_db_adapter())
        assert result is True


class TestFeatureFlagIntegration:
    """Test feature flag integration"""
    
    @patch('config.feature_flags.feature_flags.is_enabled')
    def test_feature_flag_disabled(self, mock_is_enabled):
        """Test behavior when feature flag is disabled"""
        mock_is_enabled.return_value = False
        
        # Feature should be disabled
        assert is_timescale_persistence_enabled() is False
    
    @patch('config.feature_flags.feature_flags.is_enabled')  
    def test_feature_flag_enabled(self, mock_is_enabled):
        """Test behavior when feature flag is enabled"""
        mock_is_enabled.return_value = True
        
        # Feature should be enabled
        assert is_timescale_persistence_enabled() is True


class TestDataStructures:
    """Test data structure classes"""
    
    def test_price_record_creation(self):
        """Test PriceRecord creation and attributes"""
        record = PriceRecord(
            symbol="AAPL",
            date_recorded=datetime(2023, 1, 1),
            open_price=150.0,
            close_price=155.0
        )
        
        assert record.symbol == "AAPL"
        assert record.date_recorded == datetime(2023, 1, 1)
        assert record.open_price == 150.0
        assert record.close_price == 155.0
        assert record.volume is None  # Optional field
    
    def test_sentiment_record_creation(self):
        """Test SentimentRecord creation and defaults"""
        record = SentimentRecord(
            symbol="AAPL",
            date_recorded=datetime(2023, 1, 1),
            sentiment_score=0.8
        )
        
        assert record.symbol == "AAPL"
        assert record.sentiment_score == 0.8
        assert record.article_count == 1  # Default value
        assert record.positive_mentions == 0  # Default value
    
    def test_prediction_record_creation(self):
        """Test PredictionRecord creation"""
        record = PredictionRecord(
            symbol="AAPL",
            date_recorded=datetime(2023, 1, 1),
            prediction_type="price_movement",
            predicted_value=160.0,
            features_used=["price", "volume"]
        )
        
        assert record.symbol == "AAPL"
        assert record.prediction_type == "price_movement"
        assert record.predicted_value == 160.0
        assert record.features_used == ["price", "volume"]
        assert record.actual_value is None  # Optional field


if __name__ == "__main__":
    pytest.main([__file__, "-v"])