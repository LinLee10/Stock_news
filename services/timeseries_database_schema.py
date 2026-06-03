#!/usr/bin/env python3
"""
TimescaleDB schema definitions for financial time-series data
Provides DDL for creating hypertables optimized for time-partitioned queries
"""

from typing import Dict, List


def create_hypertables() -> Dict[str, str]:
    """
    Generate DDL strings for creating TimescaleDB hypertables
    
    Returns:
        Dict mapping table names to their CREATE TABLE + hypertable DDL
    """
    
    ddl_statements = {
        'prices': _create_prices_table(),
        'sentiment': _create_sentiment_table(), 
        'predictions': _create_predictions_table()
    }
    
    return ddl_statements


def _create_prices_table() -> str:
    """DDL for stock prices hypertable"""
    return """
-- Stock prices time-series table
CREATE TABLE IF NOT EXISTS prices (
    symbol VARCHAR(10) NOT NULL,
    date_recorded TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4),
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4),
    volume BIGINT,
    adjusted_close DECIMAL(12,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Primary key for upsert operations
    PRIMARY KEY (symbol, date_recorded)
);

-- Convert to hypertable partitioned by time
SELECT create_hypertable('prices', 'date_recorded', 
                        chunk_time_interval => INTERVAL '1 month',
                        if_not_exists => TRUE);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_prices_symbol_time 
    ON prices (symbol, date_recorded DESC);
CREATE INDEX IF NOT EXISTS idx_prices_created_at 
    ON prices (created_at);
"""


def _create_sentiment_table() -> str:
    """DDL for news sentiment hypertable"""
    return """
-- News sentiment time-series table
CREATE TABLE IF NOT EXISTS sentiment (
    symbol VARCHAR(10) NOT NULL,
    date_recorded TIMESTAMPTZ NOT NULL,
    sentiment_score DECIMAL(5,4),
    sentiment_label VARCHAR(20),
    confidence_score DECIMAL(5,4),
    article_count INTEGER DEFAULT 1,
    positive_mentions INTEGER DEFAULT 0,
    negative_mentions INTEGER DEFAULT 0,
    neutral_mentions INTEGER DEFAULT 0,
    data_source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Primary key for upsert operations
    PRIMARY KEY (symbol, date_recorded)
);

-- Convert to hypertable partitioned by time
SELECT create_hypertable('sentiment', 'date_recorded',
                        chunk_time_interval => INTERVAL '1 week',
                        if_not_exists => TRUE);

-- Create indexes for sentiment analysis queries
CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time 
    ON sentiment (symbol, date_recorded DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_score 
    ON sentiment (sentiment_score);
CREATE INDEX IF NOT EXISTS idx_sentiment_label 
    ON sentiment (sentiment_label);
"""


def _create_predictions_table() -> str:
    """DDL for ML predictions hypertable"""
    return """
-- ML predictions time-series table
CREATE TABLE IF NOT EXISTS predictions (
    symbol VARCHAR(10) NOT NULL,
    date_recorded TIMESTAMPTZ NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    predicted_value DECIMAL(12,4),
    confidence_score DECIMAL(5,4),
    model_version VARCHAR(20),
    features_used TEXT[],
    prediction_horizon_days INTEGER,
    actual_value DECIMAL(12,4),
    error_magnitude DECIMAL(12,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Composite primary key for upserts
    PRIMARY KEY (symbol, date_recorded, prediction_type)
);

-- Convert to hypertable partitioned by time
SELECT create_hypertable('predictions', 'date_recorded',
                        chunk_time_interval => INTERVAL '1 month', 
                        if_not_exists => TRUE);

-- Create indexes for prediction queries
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time 
    ON predictions (symbol, date_recorded DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_type 
    ON predictions (prediction_type);
CREATE INDEX IF NOT EXISTS idx_predictions_confidence 
    ON predictions (confidence_score);
CREATE INDEX IF NOT EXISTS idx_predictions_model 
    ON predictions (model_version);
"""


def get_all_ddl() -> str:
    """
    Get all DDL statements as a single executable string
    
    Returns:
        Combined DDL for all hypertables
    """
    hypertables = create_hypertables()
    return '\n\n'.join(hypertables.values())


def get_table_names() -> List[str]:
    """
    Get list of all hypertable names
    
    Returns:
        List of table names that will be created
    """
    return list(create_hypertables().keys())