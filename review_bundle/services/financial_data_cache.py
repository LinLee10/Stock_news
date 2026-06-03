"""
Advanced Financial Data Cache with Multi-Layer Architecture

L1: In-memory cache (current session, LRU eviction)
L2: Redis cluster (hot data, 1-24 hour TTL)  
L3: SQLite/PostgreSQL (permanent storage)

Optimized for financial time-series data with intelligent invalidation.
"""

import asyncio
import logging
import json
import gzip
import pickle
import hashlib
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import threading
import weakref
import numpy as np
import pandas as pd
import aioredis
import sqlite3
import asyncpg
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"

class DataType(Enum):
    PRICE_DAILY = "price_daily"
    PRICE_INTRADAY = "price_intraday"
    VOLUME = "volume"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    TECHNICAL_INDICATOR = "technical_indicator"

class MarketEvent(Enum):
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    EARNINGS_ANNOUNCEMENT = "earnings"
    DIVIDEND_EX_DATE = "dividend_ex"
    STOCK_SPLIT = "stock_split"
    NEWS_RELEASE = "news_release"

@dataclass
class CacheEntry:
    """Optimized cache entry for financial data"""
    key: str
    data: Any
    timestamp: datetime
    data_type: DataType
    symbol: str
    ttl: timedelta
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    compressed: bool = False
    size_bytes: int = 0
    quality_score: float = 100.0
    source: str = "unknown"

@dataclass
class CacheStatistics:
    """Comprehensive cache performance metrics"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    evictions: int = 0
    total_size_mb: float = 0.0
    avg_access_time_ms: float = 0.0
    cache_efficiency: float = 0.0

class LRUCache:
    """Thread-safe LRU cache for L1 in-memory storage"""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 500):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory = 0
        self.lock = threading.RLock()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache, moving it to end (most recently used)"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(key)
                self.cache[key] = entry
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                self.stats['hits'] += 1
                return entry
            else:
                self.stats['misses'] += 1
                return None

    def put(self, key: str, entry: CacheEntry) -> bool:
        """Add item to cache, evicting LRU items if necessary"""
        with self.lock:
            # If key exists, remove it first
            if key in self.cache:
                old_entry = self.cache.pop(key)
                self.current_memory -= old_entry.size_bytes

            # Check memory constraints
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + entry.size_bytes > self.max_memory_bytes):
                if not self.cache:
                    break
                self._evict_lru()

            # Add new entry
            self.cache[key] = entry
            self.current_memory += entry.size_bytes
            return True

    def _evict_lru(self):
        """Evict least recently used item"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.current_memory -= entry.size_bytes
            self.stats['evictions'] += 1
            logger.debug(f"Evicted LRU entry: {key}")

    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_mb': self.current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'hit_rate': hit_rate
            }

class FinancialDataCache:
    """Advanced multi-layer financial data cache"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 postgres_dsn: str = None,
                 sqlite_path: str = "data/financial_cache.db"):
        
        # Configuration
        self.redis_url = redis_url
        self.postgres_dsn = postgres_dsn
        self.sqlite_path = sqlite_path
        
        # Cache layers
        self.l1_cache = LRUCache(max_size=10000, max_memory_mb=500)
        self.redis_client: Optional[aioredis.Redis] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        
        # TTL configuration by data type and market hours
        self.ttl_config = {
            DataType.PRICE_DAILY: {
                'market_hours': timedelta(minutes=15),
                'after_hours': timedelta(hours=4),
                'weekend': timedelta(hours=24)
            },
            DataType.PRICE_INTRADAY: {
                'market_hours': timedelta(minutes=1),
                'after_hours': timedelta(minutes=30),
                'weekend': timedelta(hours=6)
            },
            DataType.FUNDAMENTAL: {
                'market_hours': timedelta(hours=6),
                'after_hours': timedelta(hours=12),
                'weekend': timedelta(days=1)
            },
            DataType.NEWS: {
                'market_hours': timedelta(minutes=5),
                'after_hours': timedelta(minutes=30),
                'weekend': timedelta(hours=2)
            }
        }
        
        # Market hours (Eastern Time)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Statistics
        self.stats = CacheStatistics()
        
        # Event subscriptions for cache invalidation
        self.event_subscriptions: Dict[MarketEvent, List[str]] = {}

    async def initialize(self) -> bool:
        """Initialize all cache layers"""
        try:
            logger.info("Initializing FinancialDataCache...")
            
            # Initialize Redis connection
            try:
                self.redis_client = aioredis.from_url(
                    self.redis_url,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                await self.redis_client.ping()
                logger.info("Redis L2 cache initialized")
            except Exception as e:
                logger.warning(f"Redis unavailable, using SQLite only: {e}")
                self.redis_client = None
            
            # Initialize PostgreSQL connection pool (if configured)
            if self.postgres_dsn:
                try:
                    self.postgres_pool = await asyncpg.create_pool(
                        self.postgres_dsn,
                        min_size=5,
                        max_size=20,
                        command_timeout=30
                    )
                    logger.info("PostgreSQL L3 cache initialized")
                except Exception as e:
                    logger.warning(f"PostgreSQL unavailable, using SQLite: {e}")
            
            # Initialize SQLite (fallback for L3)
            if not self.postgres_pool:
                await self._initialize_sqlite()
            
            # Setup cache warming scheduler
            asyncio.create_task(self._cache_warming_scheduler())
            
            logger.info("FinancialDataCache initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FinancialDataCache: {e}")
            return False

    async def _initialize_sqlite(self):
        """Initialize SQLite database with optimized schema"""
        db_dir = Path(self.sqlite_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        self.sqlite_conn = sqlite3.connect(
            self.sqlite_path,
            check_same_thread=False,
            isolation_level=None  # Autocommit mode
        )
        
        # Enable WAL mode for better concurrency
        self.sqlite_conn.execute("PRAGMA journal_mode=WAL")
        self.sqlite_conn.execute("PRAGMA synchronous=NORMAL")
        self.sqlite_conn.execute("PRAGMA cache_size=10000")
        self.sqlite_conn.execute("PRAGMA temp_store=memory")
        
        cursor = self.sqlite_conn.cursor()
        
        # Time-series optimized table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_data (
                id INTEGER PRIMARY KEY,
                cache_key TEXT NOT NULL,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                data_date DATE NOT NULL,
                data BLOB NOT NULL,
                compressed BOOLEAN DEFAULT 0,
                size_bytes INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 100.0,
                source TEXT DEFAULT 'unknown',
                access_count INTEGER DEFAULT 0,
                last_accessed INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                ttl_expires INTEGER NOT NULL
            )
        """)
        
        # Partitioning-like indexes for query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON financial_data(symbol, data_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON financial_data(cache_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_type_date ON financial_data(data_type, data_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON financial_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ttl_expires ON financial_data(ttl_expires)")
        
        # Metadata table for cache statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """)
        
        logger.info("SQLite L3 cache initialized with optimized schema")

    async def get(self, key: str, symbol: str = None) -> Optional[Tuple[Any, CacheEntry]]:
        """Get data from cache with intelligent tier selection"""
        start_time = datetime.now()
        
        try:
            # L1: In-memory cache (fastest)
            l1_entry = self.l1_cache.get(key)
            if l1_entry and not self._is_expired(l1_entry):
                self.stats.l1_hits += 1
                self._update_access_time(start_time)
                return l1_entry.data, l1_entry
            elif l1_entry:
                # Entry expired, remove from L1
                self.l1_cache.cache.pop(key, None)
            
            self.stats.l1_misses += 1
            
            # L2: Redis cache (fast)
            if self.redis_client:
                l2_result = await self._get_from_redis(key)
                if l2_result:
                    data, entry = l2_result
                    self.stats.l2_hits += 1
                    
                    # Promote to L1 if frequently accessed
                    if entry.access_count > 3:
                        self.l1_cache.put(key, entry)
                    
                    self._update_access_time(start_time)
                    return data, entry
            
            self.stats.l2_misses += 1
            
            # L3: Database cache (persistent)
            l3_result = await self._get_from_database(key, symbol)
            if l3_result:
                data, entry = l3_result
                self.stats.l3_hits += 1
                
                # Promote to L2 and L1 based on access pattern
                if self.redis_client and entry.access_count > 1:
                    await self._put_to_redis(key, entry)
                
                if entry.access_count > 5:
                    self.l1_cache.put(key, entry)
                
                self._update_access_time(start_time)
                return data, entry
            
            self.stats.l3_misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting cache entry {key}: {e}")
            return None

    async def put(self, key: str, data: Any, symbol: str, data_type: DataType,
                  source: str = "unknown", quality_score: float = 100.0) -> bool:
        """Store data in cache with intelligent tier placement"""
        try:
            # Determine TTL based on data type and market conditions
            ttl = self._calculate_dynamic_ttl(data_type)
            
            # Serialize and compress data
            serialized_data = self._serialize_data(data)
            compressed_data, is_compressed = self._compress_if_beneficial(serialized_data)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=datetime.now(),
                data_type=data_type,
                symbol=symbol,
                ttl=ttl,
                compressed=is_compressed,
                size_bytes=len(compressed_data),
                quality_score=quality_score,
                source=source
            )
            
            # Store in appropriate tiers based on data characteristics
            tasks = []
            
            # Always store in L3 for persistence
            if self.postgres_pool:
                tasks.append(self._put_to_postgres(key, entry, compressed_data))
            elif self.sqlite_conn:
                tasks.append(self._put_to_sqlite(key, entry, compressed_data))
            
            # Store in L2 (Redis) for hot data
            if self.redis_client and self._should_cache_in_redis(data_type, entry.size_bytes):
                tasks.append(self._put_to_redis(key, entry))
            
            # Store in L1 for very frequently accessed data
            if self._should_cache_in_memory(data_type, entry.size_bytes):
                self.l1_cache.put(key, entry)
            
            # Execute storage operations
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing cache entry {key}: {e}")
            return False

    def _calculate_dynamic_ttl(self, data_type: DataType) -> timedelta:
        """Calculate TTL based on data type and current market conditions"""
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        # Get base TTL configuration
        ttl_config = self.ttl_config.get(data_type, {
            'market_hours': timedelta(hours=1),
            'after_hours': timedelta(hours=6),
            'weekend': timedelta(hours=12)
        })
        
        # Weekend (Saturday=5, Sunday=6)
        if weekday >= 5:
            return ttl_config['weekend']
        
        # Market hours
        if self.market_open <= current_time <= self.market_close:
            return ttl_config['market_hours']
        
        # After hours
        return ttl_config['after_hours']

    def _should_cache_in_redis(self, data_type: DataType, size_bytes: int) -> bool:
        """Determine if data should be cached in Redis"""
        # Cache size limit for Redis (1MB)
        if size_bytes > 1024 * 1024:
            return False
        
        # Cache hot data types
        hot_types = [DataType.PRICE_INTRADAY, DataType.NEWS]
        return data_type in hot_types

    def _should_cache_in_memory(self, data_type: DataType, size_bytes: int) -> bool:
        """Determine if data should be cached in memory"""
        # Cache size limit for memory (100KB)
        if size_bytes > 100 * 1024:
            return False
        
        # Cache very hot data types
        very_hot_types = [DataType.PRICE_INTRADAY]
        return data_type in very_hot_types

    async def _get_from_redis(self, key: str) -> Optional[Tuple[Any, CacheEntry]]:
        """Get data from Redis L2 cache"""
        try:
            # Get data and metadata
            pipe = self.redis_client.pipeline()
            pipe.hget(f"fd:{key}", "data")
            pipe.hget(f"fd:{key}", "metadata")
            results = await pipe.execute()
            
            if not all(results):
                return None
            
            data_bytes, metadata_bytes = results
            
            # Deserialize metadata
            metadata = json.loads(metadata_bytes.decode())
            
            # Check TTL
            expires_at = datetime.fromtimestamp(metadata['expires_at'])
            if datetime.now() > expires_at:
                # Expired, remove from cache
                await self.redis_client.delete(f"fd:{key}")
                return None
            
            # Deserialize data
            if metadata['compressed']:
                data_bytes = gzip.decompress(data_bytes)
            
            data = self._deserialize_data(data_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=datetime.fromtimestamp(metadata['timestamp']),
                data_type=DataType(metadata['data_type']),
                symbol=metadata['symbol'],
                ttl=timedelta(seconds=metadata['ttl_seconds']),
                access_count=metadata.get('access_count', 0) + 1,
                compressed=metadata['compressed'],
                size_bytes=metadata['size_bytes'],
                quality_score=metadata.get('quality_score', 100.0),
                source=metadata.get('source', 'unknown')
            )
            
            # Update access statistics
            await self.redis_client.hincrby(f"fd:{key}", "access_count", 1)
            
            return data, entry
            
        except Exception as e:
            logger.warning(f"Redis get error for {key}: {e}")
            return None

    async def _put_to_redis(self, key: str, entry: CacheEntry) -> bool:
        """Store data in Redis L2 cache"""
        try:
            # Serialize data
            data_bytes = self._serialize_data(entry.data)
            if entry.compressed:
                data_bytes = gzip.compress(data_bytes)
            
            # Create metadata
            expires_at = entry.timestamp + entry.ttl
            metadata = {
                'timestamp': entry.timestamp.timestamp(),
                'data_type': entry.data_type.value,
                'symbol': entry.symbol,
                'ttl_seconds': int(entry.ttl.total_seconds()),
                'access_count': entry.access_count,
                'compressed': entry.compressed,
                'size_bytes': entry.size_bytes,
                'quality_score': entry.quality_score,
                'source': entry.source,
                'expires_at': expires_at.timestamp()
            }
            
            # Store in Redis with TTL
            pipe = self.redis_client.pipeline()
            pipe.hset(f"fd:{key}", "data", data_bytes)
            pipe.hset(f"fd:{key}", "metadata", json.dumps(metadata))
            pipe.expire(f"fd:{key}", int(entry.ttl.total_seconds()))
            await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.warning(f"Redis put error for {key}: {e}")
            return False

    async def _get_from_database(self, key: str, symbol: str = None) -> Optional[Tuple[Any, CacheEntry]]:
        """Get data from database L3 cache"""
        if self.postgres_pool:
            return await self._get_from_postgres(key, symbol)
        elif self.sqlite_conn:
            return await self._get_from_sqlite(key, symbol)
        return None

    async def _get_from_sqlite(self, key: str, symbol: str = None) -> Optional[Tuple[Any, CacheEntry]]:
        """Get data from SQLite database"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Query with TTL check
            cursor.execute("""
                SELECT data, symbol, data_type, timestamp, compressed, size_bytes,
                       quality_score, source, access_count, ttl_expires
                FROM financial_data 
                WHERE cache_key = ? AND ttl_expires > ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (key, int(datetime.now().timestamp())))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            (data_bytes, symbol, data_type, timestamp, compressed, size_bytes,
             quality_score, source, access_count, ttl_expires) = result
            
            # Deserialize data
            if compressed:
                data_bytes = gzip.decompress(data_bytes)
            
            data = self._deserialize_data(data_bytes)
            
            # Update access count
            cursor.execute("""
                UPDATE financial_data 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE cache_key = ?
            """, (int(datetime.now().timestamp()), key))
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=datetime.fromtimestamp(timestamp),
                data_type=DataType(data_type),
                symbol=symbol,
                ttl=timedelta(seconds=ttl_expires - timestamp),
                access_count=access_count + 1,
                compressed=compressed,
                size_bytes=size_bytes,
                quality_score=quality_score,
                source=source
            )
            
            return data, entry
            
        except Exception as e:
            logger.warning(f"SQLite get error for {key}: {e}")
            return None

    async def _put_to_sqlite(self, key: str, entry: CacheEntry, compressed_data: bytes) -> bool:
        """Store data in SQLite database"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Extract date from data for partitioning-like queries
            data_date = entry.timestamp.date()
            ttl_expires = int((entry.timestamp + entry.ttl).timestamp())
            
            cursor.execute("""
                INSERT OR REPLACE INTO financial_data 
                (cache_key, symbol, data_type, timestamp, data_date, data,
                 compressed, size_bytes, quality_score, source, access_count,
                 last_accessed, created_at, ttl_expires)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key, entry.symbol, entry.data_type.value,
                int(entry.timestamp.timestamp()), data_date, compressed_data,
                entry.compressed, entry.size_bytes, entry.quality_score,
                entry.source, entry.access_count,
                int(entry.last_accessed.timestamp()),
                int(datetime.now().timestamp()), ttl_expires
            ))
            
            return True
            
        except Exception as e:
            logger.warning(f"SQLite put error for {key}: {e}")
            return False

    async def _put_to_postgres(self, key: str, entry: CacheEntry, compressed_data: bytes) -> bool:
        """Store data in PostgreSQL database"""
        try:
            async with self.postgres_pool.acquire() as conn:
                data_date = entry.timestamp.date()
                ttl_expires = entry.timestamp + entry.ttl
                
                await conn.execute("""
                    INSERT INTO financial_data 
                    (cache_key, symbol, data_type, timestamp, data_date, data,
                     compressed, size_bytes, quality_score, source, access_count,
                     last_accessed, created_at, ttl_expires)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (cache_key) DO UPDATE SET
                    data = EXCLUDED.data,
                    timestamp = EXCLUDED.timestamp,
                    compressed = EXCLUDED.compressed,
                    size_bytes = EXCLUDED.size_bytes,
                    quality_score = EXCLUDED.quality_score,
                    access_count = EXCLUDED.access_count,
                    last_accessed = EXCLUDED.last_accessed,
                    ttl_expires = EXCLUDED.ttl_expires
                """, key, entry.symbol, entry.data_type.value,
                     entry.timestamp, data_date, compressed_data,
                     entry.compressed, entry.size_bytes, entry.quality_score,
                     entry.source, entry.access_count, entry.last_accessed,
                     datetime.now(), ttl_expires)
            
            return True
            
        except Exception as e:
            logger.warning(f"PostgreSQL put error for {key}: {e}")
            return False

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > entry.timestamp + entry.ttl

    def _serialize_data(self, data: Any) -> bytes:
        """Efficiently serialize financial data"""
        if isinstance(data, pd.DataFrame):
            # Optimize pandas DataFrame serialization
            return data.to_pickle(compression='gzip')
        elif isinstance(data, np.ndarray):
            # Optimize numpy array serialization
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize_data(self, data_bytes: bytes) -> Any:
        """Efficiently deserialize financial data"""
        try:
            # Try pandas first (most common for financial data)
            return pd.read_pickle(data_bytes)
        except:
            # Fallback to general pickle
            return pickle.loads(data_bytes)

    def _compress_if_beneficial(self, data_bytes: bytes) -> Tuple[bytes, bool]:
        """Compress data if it reduces size significantly"""
        if len(data_bytes) < 1000:  # Don't compress small data
            return data_bytes, False
        
        compressed = gzip.compress(data_bytes, compresslevel=6)
        
        # Only use compression if it saves at least 25%
        if len(compressed) < len(data_bytes) * 0.75:
            return compressed, True
        else:
            return data_bytes, False

    def _update_access_time(self, start_time: datetime):
        """Update average access time statistics"""
        access_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if self.stats.avg_access_time_ms == 0:
            self.stats.avg_access_time_ms = access_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_access_time_ms = (
                alpha * access_time + 
                (1 - alpha) * self.stats.avg_access_time_ms
            )

    async def invalidate(self, pattern: str = None, symbol: str = None, 
                        data_type: DataType = None) -> int:
        """Invalidate cache entries based on pattern, symbol, or data type"""
        invalidated = 0
        
        try:
            # L1 cache invalidation
            if pattern or symbol or data_type:
                keys_to_remove = []
                for key, entry in self.l1_cache.cache.items():
                    should_invalidate = False
                    
                    if pattern and pattern in key:
                        should_invalidate = True
                    elif symbol and entry.symbol == symbol:
                        should_invalidate = True
                    elif data_type and entry.data_type == data_type:
                        should_invalidate = True
                    
                    if should_invalidate:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self.l1_cache.cache.pop(key, None)
                    invalidated += 1
            
            # L2 cache invalidation (Redis)
            if self.redis_client:
                if pattern:
                    keys = await self.redis_client.keys(f"fd:*{pattern}*")
                    if keys:
                        await self.redis_client.delete(*keys)
                        invalidated += len(keys)
            
            # L3 cache invalidation (Database)
            if symbol or data_type:
                await self._invalidate_database(symbol, data_type)
                
        except Exception as e:
            logger.error(f"Error during cache invalidation: {e}")
        
        logger.info(f"Invalidated {invalidated} cache entries")
        return invalidated

    async def _invalidate_database(self, symbol: str = None, data_type: DataType = None):
        """Invalidate database cache entries"""
        try:
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    where_conditions = []
                    params = []
                    
                    if symbol:
                        where_conditions.append(f"symbol = ${len(params) + 1}")
                        params.append(symbol)
                    
                    if data_type:
                        where_conditions.append(f"data_type = ${len(params) + 1}")
                        params.append(data_type.value)
                    
                    if where_conditions:
                        query = f"DELETE FROM financial_data WHERE {' AND '.join(where_conditions)}"
                        await conn.execute(query, *params)
            
            elif self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                where_conditions = []
                params = []
                
                if symbol:
                    where_conditions.append("symbol = ?")
                    params.append(symbol)
                
                if data_type:
                    where_conditions.append("data_type = ?")
                    params.append(data_type.value)
                
                if where_conditions:
                    query = f"DELETE FROM financial_data WHERE {' AND '.join(where_conditions)}"
                    cursor.execute(query, params)
                    
        except Exception as e:
            logger.warning(f"Database invalidation error: {e}")

    async def _cache_warming_scheduler(self):
        """Background task for cache warming before market open"""
        while True:
            try:
                now = datetime.now()
                
                # Warm cache 30 minutes before market open
                market_open_today = datetime.combine(now.date(), self.market_open)
                warm_time = market_open_today - timedelta(minutes=30)
                
                if abs((now - warm_time).total_seconds()) < 300:  # Within 5 minutes
                    await self._warm_critical_data()
                
                # Sleep for 5 minutes before checking again
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Cache warming scheduler error: {e}")
                await asyncio.sleep(300)

    async def _warm_critical_data(self):
        """Pre-warm cache with critical data before market open"""
        logger.info("Starting cache warming for critical data")
        
        # This would integrate with your existing data fetching system
        # to pre-load commonly accessed tickers
        critical_symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
        
        # Placeholder for actual warming logic
        # You would call your data manager here to fetch and cache data
        
        logger.info(f"Cache warming completed for {len(critical_symbols)} symbols")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        l1_stats = self.l1_cache.get_stats()
        
        total_requests = (self.stats.l1_hits + self.stats.l1_misses +
                         self.stats.l2_hits + self.stats.l2_misses +
                         self.stats.l3_hits + self.stats.l3_misses)
        
        total_hits = self.stats.l1_hits + self.stats.l2_hits + self.stats.l3_hits
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'overall': {
                'total_requests': total_requests,
                'total_hits': total_hits,
                'hit_rate_percent': round(hit_rate, 2),
                'avg_access_time_ms': round(self.stats.avg_access_time_ms, 2)
            },
            'l1_memory': l1_stats,
            'l2_redis': {
                'hits': self.stats.l2_hits,
                'misses': self.stats.l2_misses,
                'hit_rate': round((self.stats.l2_hits / max(1, self.stats.l2_hits + self.stats.l2_misses)) * 100, 2),
                'available': self.redis_client is not None
            },
            'l3_database': {
                'hits': self.stats.l3_hits,
                'misses': self.stats.l3_misses,
                'hit_rate': round((self.stats.l3_hits / max(1, self.stats.l3_hits + self.stats.l3_misses)) * 100, 2),
                'type': 'PostgreSQL' if self.postgres_pool else 'SQLite'
            }
        }

    async def cleanup(self):
        """Clean up all cache resources"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        if self.sqlite_conn:
            self.sqlite_conn.close()
        
        self.l1_cache.clear()

# Factory function for easy integration
async def create_financial_cache(redis_url: str = None, postgres_dsn: str = None) -> FinancialDataCache:
    """Create and initialize a financial data cache"""
    cache = FinancialDataCache(
        redis_url=redis_url or "redis://localhost:6379",
        postgres_dsn=postgres_dsn
    )
    
    await cache.initialize()
    return cache