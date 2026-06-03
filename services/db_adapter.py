#!/usr/bin/env python3
"""
Database persistence adapters for TimescaleDB integration
Provides optional database persistence controlled by feature flags and DSN configuration
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from contextlib import asynccontextmanager

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from config.feature_flags import is_timescale_persistence_enabled
from services.timeseries_database_schema import create_hypertables, get_all_ddl

logger = logging.getLogger(__name__)


@dataclass
class PriceRecord:
    symbol: str
    date_recorded: datetime
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: Optional[int] = None
    adjusted_close: Optional[float] = None


@dataclass
class SentimentRecord:
    symbol: str
    date_recorded: datetime
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    confidence_score: Optional[float] = None
    article_count: int = 1
    positive_mentions: int = 0
    negative_mentions: int = 0
    neutral_mentions: int = 0
    data_source: Optional[str] = None


@dataclass
class PredictionRecord:
    symbol: str
    date_recorded: datetime
    prediction_type: str
    predicted_value: Optional[float] = None
    confidence_score: Optional[float] = None
    model_version: Optional[str] = None
    features_used: Optional[List[str]] = None
    prediction_horizon_days: Optional[int] = None
    actual_value: Optional[float] = None
    error_magnitude: Optional[float] = None


class DatabaseAdapter:
    """
    TimescaleDB database adapter with optional persistence
    No-op unless both feature flag is enabled and DSN is provided
    """
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.dsn = os.getenv('PG_DSN')
        self._initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize database connection if enabled and configured
        
        Returns:
            bool: True if initialized successfully, False if no-op
        """
        if not is_timescale_persistence_enabled():
            logger.debug("TimescaleDB persistence disabled by feature flag")
            return False
            
        if not self.dsn:
            logger.debug("No PG_DSN configured, persistence disabled")
            return False
            
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available, persistence disabled")
            return False
            
        try:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            
            # Create schema if needed
            await self._ensure_schema_exists()
            
            self._initialized = True
            logger.info("TimescaleDB adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB adapter: {e}")
            return False
    
    async def _ensure_schema_exists(self):
        """Create hypertables if they don't exist"""
        if not self.pool:
            return
            
        async with self.pool.acquire() as conn:
            # Enable TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
            
            # Execute DDL for all hypertables
            ddl = get_all_ddl()
            await conn.execute(ddl)
            
        logger.info("TimescaleDB schema ensured")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection if available"""
        if not self._initialized or not self.pool:
            yield None
            return
            
        async with self.pool.acquire() as conn:
            yield conn
    
    async def insert_prices_batch(self, prices: List[PriceRecord]) -> int:
        """
        Batch insert price records with upsert on conflict
        
        Args:
            prices: List of price records to insert
            
        Returns:
            Number of records processed (0 if no-op)
        """
        if not prices or not self._initialized:
            return 0
            
        async with self.get_connection() as conn:
            if not conn:
                return 0
                
            # Prepare batch data
            records = [
                (
                    p.symbol, p.date_recorded, p.open_price, p.high_price,
                    p.low_price, p.close_price, p.volume, p.adjusted_close
                )
                for p in prices
            ]
            
            # Upsert with ON CONFLICT
            await conn.executemany("""
                INSERT INTO prices (
                    symbol, date_recorded, open_price, high_price, 
                    low_price, close_price, volume, adjusted_close
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol, date_recorded) 
                DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    adjusted_close = EXCLUDED.adjusted_close,
                    updated_at = NOW()
            """, records)
            
        logger.debug(f"Inserted/updated {len(prices)} price records")
        return len(prices)
    
    async def insert_sentiment_batch(self, sentiments: List[SentimentRecord]) -> int:
        """
        Batch insert sentiment records with upsert on conflict
        
        Args:
            sentiments: List of sentiment records to insert
            
        Returns:
            Number of records processed (0 if no-op)
        """
        if not sentiments or not self._initialized:
            return 0
            
        async with self.get_connection() as conn:
            if not conn:
                return 0
                
            # Prepare batch data
            records = [
                (
                    s.symbol, s.date_recorded, s.sentiment_score, s.sentiment_label,
                    s.confidence_score, s.article_count, s.positive_mentions,
                    s.negative_mentions, s.neutral_mentions, s.data_source
                )
                for s in sentiments
            ]
            
            # Upsert with ON CONFLICT
            await conn.executemany("""
                INSERT INTO sentiment (
                    symbol, date_recorded, sentiment_score, sentiment_label,
                    confidence_score, article_count, positive_mentions,
                    negative_mentions, neutral_mentions, data_source
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (symbol, date_recorded)
                DO UPDATE SET
                    sentiment_score = EXCLUDED.sentiment_score,
                    sentiment_label = EXCLUDED.sentiment_label,
                    confidence_score = EXCLUDED.confidence_score,
                    article_count = EXCLUDED.article_count,
                    positive_mentions = EXCLUDED.positive_mentions,
                    negative_mentions = EXCLUDED.negative_mentions,
                    neutral_mentions = EXCLUDED.neutral_mentions,
                    data_source = EXCLUDED.data_source,
                    updated_at = NOW()
            """, records)
            
        logger.debug(f"Inserted/updated {len(sentiments)} sentiment records")
        return len(sentiments)
    
    async def insert_predictions_batch(self, predictions: List[PredictionRecord]) -> int:
        """
        Batch insert prediction records with upsert on conflict
        
        Args:
            predictions: List of prediction records to insert
            
        Returns:
            Number of records processed (0 if no-op)
        """
        if not predictions or not self._initialized:
            return 0
            
        async with self.get_connection() as conn:
            if not conn:
                return 0
                
            # Prepare batch data
            records = [
                (
                    p.symbol, p.date_recorded, p.prediction_type, p.predicted_value,
                    p.confidence_score, p.model_version, p.features_used,
                    p.prediction_horizon_days, p.actual_value, p.error_magnitude
                )
                for p in predictions
            ]
            
            # Upsert with ON CONFLICT
            await conn.executemany("""
                INSERT INTO predictions (
                    symbol, date_recorded, prediction_type, predicted_value,
                    confidence_score, model_version, features_used,
                    prediction_horizon_days, actual_value, error_magnitude
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (symbol, date_recorded, prediction_type)
                DO UPDATE SET
                    predicted_value = EXCLUDED.predicted_value,
                    confidence_score = EXCLUDED.confidence_score,
                    model_version = EXCLUDED.model_version,
                    features_used = EXCLUDED.features_used,
                    prediction_horizon_days = EXCLUDED.prediction_horizon_days,
                    actual_value = EXCLUDED.actual_value,
                    error_magnitude = EXCLUDED.error_magnitude,
                    updated_at = NOW()
            """, records)
            
        logger.debug(f"Inserted/updated {len(predictions)} prediction records")
        return len(predictions)
    
    async def query_prices(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Query price data for a symbol within date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of price records (empty if no-op)
        """
        if not self._initialized:
            return []
            
        async with self.get_connection() as conn:
            if not conn:
                return []
                
            records = await conn.fetch("""
                SELECT symbol, date_recorded, open_price, high_price, 
                       low_price, close_price, volume, adjusted_close,
                       created_at, updated_at
                FROM prices 
                WHERE symbol = $1 
                  AND date_recorded >= $2 
                  AND date_recorded <= $3
                ORDER BY date_recorded
            """, symbol, start_date, end_date)
            
            return [dict(record) for record in records]
    
    async def query_sentiment(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Query sentiment data for a symbol within date range
        
        Args:
            symbol: Stock symbol  
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of sentiment records (empty if no-op)
        """
        if not self._initialized:
            return []
            
        async with self.get_connection() as conn:
            if not conn:
                return []
                
            records = await conn.fetch("""
                SELECT symbol, date_recorded, sentiment_score, sentiment_label,
                       confidence_score, article_count, positive_mentions,
                       negative_mentions, neutral_mentions, data_source,
                       created_at, updated_at
                FROM sentiment 
                WHERE symbol = $1 
                  AND date_recorded >= $2 
                  AND date_recorded <= $3
                ORDER BY date_recorded
            """, symbol, start_date, end_date)
            
            return [dict(record) for record in records]
    
    async def query_predictions(self, symbol: str, prediction_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query prediction data for a symbol
        
        Args:
            symbol: Stock symbol
            prediction_type: Optional filter by prediction type
            
        Returns:
            List of prediction records (empty if no-op)
        """
        if not self._initialized:
            return []
            
        async with self.get_connection() as conn:
            if not conn:
                return []
                
            if prediction_type:
                records = await conn.fetch("""
                    SELECT symbol, date_recorded, prediction_type, predicted_value,
                           confidence_score, model_version, features_used,
                           prediction_horizon_days, actual_value, error_magnitude,
                           created_at, updated_at
                    FROM predictions 
                    WHERE symbol = $1 AND prediction_type = $2
                    ORDER BY date_recorded DESC
                """, symbol, prediction_type)
            else:
                records = await conn.fetch("""
                    SELECT symbol, date_recorded, prediction_type, predicted_value,
                           confidence_score, model_version, features_used,
                           prediction_horizon_days, actual_value, error_magnitude,
                           created_at, updated_at
                    FROM predictions 
                    WHERE symbol = $1
                    ORDER BY date_recorded DESC
                """, symbol)
            
            return [dict(record) for record in records]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check database health and connectivity
        
        Returns:
            Health status dict
        """
        status = {
            'enabled': is_timescale_persistence_enabled(),
            'configured': bool(self.dsn),
            'initialized': self._initialized,
            'asyncpg_available': ASYNCPG_AVAILABLE,
            'connection_healthy': False
        }
        
        if self._initialized:
            try:
                async with self.get_connection() as conn:
                    if conn:
                        await conn.fetchval("SELECT 1")
                        status['connection_healthy'] = True
            except Exception as e:
                logger.error(f"Health check failed: {e}")
        
        return status
    
    async def cleanup(self):
        """Clean up database connections"""
        if self.pool:
            await self.pool.close()
            self.pool = None
        self._initialized = False


# Global instance
db_adapter = DatabaseAdapter()


# Convenience functions for common operations
async def persist_prices(prices: List[PriceRecord]) -> int:
    """
    Persist price records to database
    
    Args:
        prices: List of price records
        
    Returns:
        Number of records processed
    """
    return await db_adapter.insert_prices_batch(prices)


async def persist_sentiment(sentiments: List[SentimentRecord]) -> int:
    """
    Persist sentiment records to database
    
    Args:
        sentiments: List of sentiment records
        
    Returns:
        Number of records processed
    """
    return await db_adapter.insert_sentiment_batch(sentiments)


async def persist_predictions(predictions: List[PredictionRecord]) -> int:
    """
    Persist prediction records to database
    
    Args:
        predictions: List of prediction records
        
    Returns:
        Number of records processed
    """
    return await db_adapter.insert_predictions_batch(predictions)


async def initialize_db_adapter() -> bool:
    """
    Initialize the global database adapter
    
    Returns:
        True if initialized successfully
    """
    return await db_adapter.initialize()