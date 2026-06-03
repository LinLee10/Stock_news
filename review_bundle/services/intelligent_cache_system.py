"""
Intelligent Caching System with Redis and SQLite

Multi-tiered caching strategy:
- L1: Redis (hot data, sub-second access)
- L2: SQLite (warm data, local persistence)  
- L3: File system (cold data, long-term storage)

Features:
- Smart cache eviction based on access patterns
- Data compression for large datasets
- Cache warming for predictable requests
- Quality-aware cache TTL
- Cross-validation between sources
"""

import asyncio
import logging
import json
import gzip
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
import aioredis
from multi_source_data_manager import DataSource, DataQuality

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    L1_REDIS = "l1_redis"      # Hot cache - millisecond access
    L2_SQLITE = "l2_sqlite"    # Warm cache - 10-50ms access
    L3_FILE = "l3_file"        # Cold cache - 100ms+ access

class CacheStrategy(Enum):
    WRITE_THROUGH = "write_through"    # Write to cache and storage simultaneously
    WRITE_BEHIND = "write_behind"      # Write to cache first, storage later
    WRITE_AROUND = "write_around"      # Write only to storage, bypass cache

@dataclass
class CacheEntry:
    key: str
    data: Any
    source: DataSource
    quality: DataQuality
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl: timedelta = field(default_factory=lambda: timedelta(hours=6))
    compressed: bool = False
    size_bytes: int = 0

@dataclass
class CacheStatistics:
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    total_requests: int = 0
    average_response_time: float = 0.0
    cache_size_mb: float = 0.0
    evictions: int = 0

class IntelligentCacheSystem:
    """Multi-tiered intelligent caching system with Redis, SQLite, and file storage"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 sqlite_path: str = "data/intelligent_cache.db",
                 file_cache_dir: str = "data/cache"):
        
        self.redis_url = redis_url
        self.sqlite_path = sqlite_path
        self.file_cache_dir = Path(file_cache_dir)
        
        # Cache connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        
        # Cache configuration
        self.l1_max_size = 100_000_000  # 100MB Redis cache
        self.l2_max_size = 1_000_000_000  # 1GB SQLite cache
        self.l3_max_size = 10_000_000_000  # 10GB file cache
        
        # Quality-based TTL configuration
        self.quality_ttl = {
            DataQuality.PREMIUM: timedelta(hours=1),
            DataQuality.STANDARD: timedelta(hours=6),
            DataQuality.BASIC: timedelta(hours=24),
            DataQuality.FALLBACK: timedelta(days=7)
        }
        
        # Statistics tracking
        self.stats = CacheStatistics()
        self.strategy = CacheStrategy.WRITE_THROUGH
        
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize all storage layers"""
        # Create cache directories
        self.file_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite
        self._initialize_sqlite()

    def _initialize_sqlite(self):
        """Initialize SQLite cache database"""
        sqlite_dir = Path(self.sqlite_path).parent
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        
        self.sqlite_conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        cursor = self.sqlite_conn.cursor()
        
        # Main cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                source TEXT NOT NULL,
                quality TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed INTEGER NOT NULL,
                ttl_seconds INTEGER NOT NULL,
                compressed BOOLEAN DEFAULT 0,
                size_bytes INTEGER DEFAULT 0
            )
        """)
        
        # Cache statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_stats (
                date TEXT PRIMARY KEY,
                l1_hits INTEGER DEFAULT 0,
                l1_misses INTEGER DEFAULT 0,
                l2_hits INTEGER DEFAULT 0,
                l2_misses INTEGER DEFAULT 0,
                l3_hits INTEGER DEFAULT 0,
                l3_misses INTEGER DEFAULT 0,
                total_requests INTEGER DEFAULT 0,
                average_response_time REAL DEFAULT 0.0,
                evictions INTEGER DEFAULT 0
            )
        """)
        
        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON cache_entries(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_last_accessed ON cache_entries(last_accessed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_quality ON cache_entries(quality)")
        
        self.sqlite_conn.commit()

    async def initialize(self) -> bool:
        """Initialize the cache system"""
        try:
            # Initialize Redis connection
            self.redis_client = aioredis.from_url(
                self.redis_url,
                decode_responses=False,  # Keep binary data
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            
            # Clean expired entries
            await self._clean_expired_entries()
            
            logger.info("Intelligent cache system initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Redis unavailable, using SQLite+File only: {e}")
            self.redis_client = None
            return True  # Continue without Redis

    async def get(self, key: str) -> Optional[Tuple[Any, CacheEntry]]:
        """Get data from cache with intelligent tier selection"""
        start_time = datetime.now()
        self.stats.total_requests += 1
        
        try:
            # L1 Cache: Redis (fastest)
            if self.redis_client:
                l1_result = await self._get_from_redis(key)
                if l1_result:
                    self.stats.l1_hits += 1
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._update_response_time(response_time)
                    return l1_result
                else:
                    self.stats.l1_misses += 1
            
            # L2 Cache: SQLite (medium speed)
            l2_result = await self._get_from_sqlite(key)
            if l2_result:
                self.stats.l2_hits += 1
                
                # Promote to L1 if Redis available and data is hot
                if self.redis_client and l2_result[1].access_count > 5:
                    await self._promote_to_redis(key, l2_result[0], l2_result[1])
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_response_time(response_time)
                return l2_result
            else:
                self.stats.l2_misses += 1
            
            # L3 Cache: File system (slowest)
            l3_result = await self._get_from_file(key)
            if l3_result:
                self.stats.l3_hits += 1
                
                # Promote to L2 if accessed frequently
                if l3_result[1].access_count > 2:
                    await self._promote_to_sqlite(key, l3_result[0], l3_result[1])
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_response_time(response_time)
                return l3_result
            else:
                self.stats.l3_misses += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def put(self, key: str, data: Any, source: DataSource, quality: DataQuality, 
                  ttl: Optional[timedelta] = None) -> bool:
        """Store data in cache with intelligent tier placement"""
        try:
            # Determine TTL based on quality if not specified
            if ttl is None:
                ttl = self.quality_ttl.get(quality, timedelta(hours=6))
            
            # Create cache entry
            serialized_data = self._serialize_data(data)
            compressed_data, is_compressed = self._compress_if_beneficial(serialized_data)
            
            entry = CacheEntry(
                key=key,
                data=data,
                source=source,
                quality=quality,
                timestamp=datetime.now(),
                ttl=ttl,
                compressed=is_compressed,
                size_bytes=len(compressed_data)
            )
            
            # Write strategy implementation
            if self.strategy == CacheStrategy.WRITE_THROUGH:
                # Write to all applicable tiers
                await self._write_to_all_tiers(key, compressed_data, entry)
                
            elif self.strategy == CacheStrategy.WRITE_BEHIND:
                # Write to L1 first, L2/L3 later
                if self.redis_client:
                    await self._put_to_redis(key, compressed_data, entry)
                asyncio.create_task(self._write_to_lower_tiers(key, compressed_data, entry))
                
            elif self.strategy == CacheStrategy.WRITE_AROUND:
                # Skip L1, write to L2/L3
                await self._put_to_sqlite(key, compressed_data, entry)
                await self._put_to_file(key, compressed_data, entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache put error: {e}")
            return False

    async def _get_from_redis(self, key: str) -> Optional[Tuple[Any, CacheEntry]]:
        """Get data from Redis L1 cache"""
        try:
            # Get data and metadata
            pipe = self.redis_client.pipeline()
            pipe.hget(f"data:{key}", "payload")
            pipe.hget(f"data:{key}", "metadata")
            results = await pipe.execute()
            
            if not all(results):
                return None
            
            data_bytes, metadata_bytes = results
            
            # Deserialize metadata
            metadata = json.loads(metadata_bytes.decode())
            
            # Check TTL
            timestamp = datetime.fromisoformat(metadata['timestamp'])
            ttl = timedelta(seconds=metadata['ttl_seconds'])
            
            if datetime.now() - timestamp > ttl:
                # Expired, remove from cache
                await self.redis_client.delete(f"data:{key}")
                return None
            
            # Deserialize data
            if metadata['compressed']:
                data_bytes = gzip.decompress(data_bytes)
            
            data = self._deserialize_data(data_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                source=DataSource(metadata['source']),
                quality=DataQuality(metadata['quality']),
                timestamp=timestamp,
                access_count=metadata.get('access_count', 0) + 1,
                ttl=ttl,
                compressed=metadata['compressed']
            )
            
            # Update access count
            await self.redis_client.hincrby(f"data:{key}", "access_count", 1)
            await self.redis_client.hset(f"data:{key}", "last_accessed", int(datetime.now().timestamp()))
            
            return data, entry
            
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None

    async def _get_from_sqlite(self, key: str) -> Optional[Tuple[Any, CacheEntry]]:
        """Get data from SQLite L2 cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            cursor.execute("""
                SELECT data, source, quality, timestamp, access_count, ttl_seconds, compressed
                FROM cache_entries WHERE key = ?
            """, (key,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            data_bytes, source, quality, timestamp, access_count, ttl_seconds, compressed = result
            
            # Check TTL
            entry_time = datetime.fromtimestamp(timestamp)
            ttl = timedelta(seconds=ttl_seconds)
            
            if datetime.now() - entry_time > ttl:
                # Expired, remove from cache
                cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                self.sqlite_conn.commit()
                return None
            
            # Deserialize data
            if compressed:
                data_bytes = gzip.decompress(data_bytes)
            
            data = self._deserialize_data(data_bytes)
            
            # Update access statistics
            cursor.execute("""
                UPDATE cache_entries 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE key = ?
            """, (int(datetime.now().timestamp()), key))
            
            self.sqlite_conn.commit()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                source=DataSource(source),
                quality=DataQuality(quality),
                timestamp=entry_time,
                access_count=access_count + 1,
                ttl=ttl,
                compressed=compressed
            )
            
            return data, entry
            
        except Exception as e:
            logger.warning(f"SQLite get error: {e}")
            return None

    async def _get_from_file(self, key: str) -> Optional[Tuple[Any, CacheEntry]]:
        """Get data from file system L3 cache"""
        try:
            # Hash key to create filename
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            data_file = self.file_cache_dir / f"{key_hash}.data"
            meta_file = self.file_cache_dir / f"{key_hash}.meta"
            
            if not data_file.exists() or not meta_file.exists():
                return None
            
            # Load metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Check TTL
            timestamp = datetime.fromisoformat(metadata['timestamp'])
            ttl = timedelta(seconds=metadata['ttl_seconds'])
            
            if datetime.now() - timestamp > ttl:
                # Expired, remove files
                data_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)
                return None
            
            # Load data
            with open(data_file, 'rb') as f:
                data_bytes = f.read()
            
            # Decompress if needed
            if metadata['compressed']:
                data_bytes = gzip.decompress(data_bytes)
            
            data = self._deserialize_data(data_bytes)
            
            # Update access count in metadata
            metadata['access_count'] += 1
            metadata['last_accessed'] = int(datetime.now().timestamp())
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                source=DataSource(metadata['source']),
                quality=DataQuality(metadata['quality']),
                timestamp=timestamp,
                access_count=metadata['access_count'],
                ttl=ttl,
                compressed=metadata['compressed']
            )
            
            return data, entry
            
        except Exception as e:
            logger.warning(f"File cache get error: {e}")
            return None

    async def _put_to_redis(self, key: str, data_bytes: bytes, entry: CacheEntry):
        """Store data in Redis L1 cache"""
        try:
            metadata = {
                'source': entry.source.value,
                'quality': entry.quality.value,
                'timestamp': entry.timestamp.isoformat(),
                'access_count': entry.access_count,
                'last_accessed': int(entry.last_accessed.timestamp()),
                'ttl_seconds': int(entry.ttl.total_seconds()),
                'compressed': entry.compressed
            }
            
            # Store data and metadata
            pipe = self.redis_client.pipeline()
            pipe.hset(f"data:{key}", "payload", data_bytes)
            pipe.hset(f"data:{key}", "metadata", json.dumps(metadata))
            pipe.expire(f"data:{key}", int(entry.ttl.total_seconds()))
            await pipe.execute()
            
        except Exception as e:
            logger.warning(f"Redis put error: {e}")

    async def _put_to_sqlite(self, key: str, data_bytes: bytes, entry: CacheEntry):
        """Store data in SQLite L2 cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (key, data, source, quality, timestamp, access_count, last_accessed, 
                 ttl_seconds, compressed, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key,
                data_bytes,
                entry.source.value,
                entry.quality.value,
                int(entry.timestamp.timestamp()),
                entry.access_count,
                int(entry.last_accessed.timestamp()),
                int(entry.ttl.total_seconds()),
                entry.compressed,
                len(data_bytes)
            ))
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            logger.warning(f"SQLite put error: {e}")

    async def _put_to_file(self, key: str, data_bytes: bytes, entry: CacheEntry):
        """Store data in file system L3 cache"""
        try:
            # Hash key to create filename
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            data_file = self.file_cache_dir / f"{key_hash}.data"
            meta_file = self.file_cache_dir / f"{key_hash}.meta"
            
            # Store data
            with open(data_file, 'wb') as f:
                f.write(data_bytes)
            
            # Store metadata
            metadata = {
                'key': key,
                'source': entry.source.value,
                'quality': entry.quality.value,
                'timestamp': entry.timestamp.isoformat(),
                'access_count': entry.access_count,
                'last_accessed': int(entry.last_accessed.timestamp()),
                'ttl_seconds': int(entry.ttl.total_seconds()),
                'compressed': entry.compressed,
                'size_bytes': len(data_bytes)
            }
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            logger.warning(f"File cache put error: {e}")

    async def _write_to_all_tiers(self, key: str, data_bytes: bytes, entry: CacheEntry):
        """Write to all applicable cache tiers"""
        tasks = []
        
        # L1: Redis (if available and data is small enough)
        if self.redis_client and len(data_bytes) < 1_000_000:  # 1MB limit for Redis
            tasks.append(self._put_to_redis(key, data_bytes, entry))
        
        # L2: SQLite
        tasks.append(self._put_to_sqlite(key, data_bytes, entry))
        
        # L3: File system (for larger data or long-term storage)
        if entry.quality in [DataQuality.PREMIUM, DataQuality.STANDARD] or len(data_bytes) > 100_000:
            tasks.append(self._put_to_file(key, data_bytes, entry))
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _write_to_lower_tiers(self, key: str, data_bytes: bytes, entry: CacheEntry):
        """Write to L2 and L3 tiers (for write-behind strategy)"""
        await asyncio.gather(
            self._put_to_sqlite(key, data_bytes, entry),
            self._put_to_file(key, data_bytes, entry),
            return_exceptions=True
        )

    async def _promote_to_redis(self, key: str, data: Any, entry: CacheEntry):
        """Promote frequently accessed data to Redis L1 cache"""
        if self.redis_client:
            serialized_data = self._serialize_data(data)
            compressed_data, _ = self._compress_if_beneficial(serialized_data)
            await self._put_to_redis(key, compressed_data, entry)

    async def _promote_to_sqlite(self, key: str, data: Any, entry: CacheEntry):
        """Promote frequently accessed data to SQLite L2 cache"""
        serialized_data = self._serialize_data(data)
        compressed_data, _ = self._compress_if_beneficial(serialized_data)
        await self._put_to_sqlite(key, compressed_data, entry)

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage"""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize_data(self, data_bytes: bytes) -> Any:
        """Deserialize data from storage"""
        return pickle.loads(data_bytes)

    def _compress_if_beneficial(self, data_bytes: bytes) -> Tuple[bytes, bool]:
        """Compress data if it reduces size significantly"""
        if len(data_bytes) < 1000:  # Don't compress small data
            return data_bytes, False
        
        compressed = gzip.compress(data_bytes, compresslevel=6)
        
        # Only use compression if it saves at least 20%
        if len(compressed) < len(data_bytes) * 0.8:
            return compressed, True
        else:
            return data_bytes, False

    def _update_response_time(self, response_time_ms: float):
        """Update average response time statistics"""
        if self.stats.total_requests == 1:
            self.stats.average_response_time = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.average_response_time = (
                alpha * response_time_ms + 
                (1 - alpha) * self.stats.average_response_time
            )

    async def _clean_expired_entries(self):
        """Clean expired entries from all cache tiers"""
        try:
            # SQLite cleanup
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                DELETE FROM cache_entries 
                WHERE timestamp + ttl_seconds < ?
            """, (int(datetime.now().timestamp()),))
            
            expired_count = cursor.rowcount
            self.sqlite_conn.commit()
            
            # File system cleanup
            file_cleaned = await self._clean_expired_files()
            
            if expired_count > 0 or file_cleaned > 0:
                logger.info(f"Cleaned {expired_count} SQLite entries and {file_cleaned} files")
                
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")

    async def _clean_expired_files(self) -> int:
        """Clean expired files from L3 cache"""
        cleaned = 0
        
        try:
            for meta_file in self.file_cache_dir.glob("*.meta"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    
                    timestamp = datetime.fromisoformat(metadata['timestamp'])
                    ttl = timedelta(seconds=metadata['ttl_seconds'])
                    
                    if datetime.now() - timestamp > ttl:
                        # Remove data and metadata files
                        data_file = meta_file.with_suffix('.data')
                        meta_file.unlink(missing_ok=True)
                        data_file.unlink(missing_ok=True)
                        cleaned += 1
                        
                except Exception as e:
                    logger.warning(f"Error cleaning file {meta_file}: {e}")
                    
        except Exception as e:
            logger.warning(f"File cleanup error: {e}")
            
        return cleaned

    async def invalidate(self, key: str) -> bool:
        """Invalidate cache entry across all tiers"""
        try:
            # Redis
            if self.redis_client:
                await self.redis_client.delete(f"data:{key}")
            
            # SQLite
            cursor = self.sqlite_conn.cursor()
            cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            self.sqlite_conn.commit()
            
            # File system
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            data_file = self.file_cache_dir / f"{key_hash}.data"
            meta_file = self.file_cache_dir / f"{key_hash}.meta"
            data_file.unlink(missing_ok=True)
            meta_file.unlink(missing_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return False

    async def warm_cache(self, keys: List[str], data_fetcher) -> int:
        """Pre-populate cache with commonly accessed data"""
        warmed = 0
        
        for key in keys:
            try:
                # Check if already cached
                if await self.get(key):
                    continue
                
                # Fetch and cache data
                data, source, quality = await data_fetcher(key)
                if data:
                    await self.put(key, data, source, quality)
                    warmed += 1
                    
            except Exception as e:
                logger.warning(f"Cache warming error for {key}: {e}")
        
        logger.info(f"Warmed {warmed} cache entries")
        return warmed

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_hits = self.stats.l1_hits + self.stats.l2_hits + self.stats.l3_hits
        total_misses = self.stats.l1_misses + self.stats.l2_misses + self.stats.l3_misses
        hit_rate = (total_hits / (total_hits + total_misses)) * 100 if (total_hits + total_misses) > 0 else 0
        
        return {
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': self.stats.total_requests,
            'average_response_time_ms': round(self.stats.average_response_time, 2),
            'l1_redis': {
                'hits': self.stats.l1_hits,
                'misses': self.stats.l1_misses,
                'hit_rate': round((self.stats.l1_hits / max(1, self.stats.l1_hits + self.stats.l1_misses)) * 100, 2)
            },
            'l2_sqlite': {
                'hits': self.stats.l2_hits,
                'misses': self.stats.l2_misses,
                'hit_rate': round((self.stats.l2_hits / max(1, self.stats.l2_hits + self.stats.l2_misses)) * 100, 2)
            },
            'l3_file': {
                'hits': self.stats.l3_hits,
                'misses': self.stats.l3_misses,
                'hit_rate': round((self.stats.l3_hits / max(1, self.stats.l3_hits + self.stats.l3_misses)) * 100, 2)
            },
            'evictions': self.stats.evictions,
            'cache_strategy': self.strategy.value
        }

    async def cleanup(self):
        """Clean up cache resources"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.sqlite_conn:
            self.sqlite_conn.close()

# Utility functions
def generate_cache_key(ticker: str, data_type: str, period: str = None, **kwargs) -> str:
    """Generate a consistent cache key for market data"""
    key_parts = [ticker, data_type]
    
    if period:
        key_parts.append(period)
    
    # Add other parameters in sorted order for consistency
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    return "|".join(key_parts)

async def get_cached_market_data(cache: IntelligentCacheSystem, ticker: str, 
                                data_type: str, **kwargs) -> Optional[Tuple[Any, CacheEntry]]:
    """Convenience function to get cached market data"""
    cache_key = generate_cache_key(ticker, data_type, **kwargs)
    return await cache.get(cache_key)

async def cache_market_data(cache: IntelligentCacheSystem, ticker: str, data_type: str, 
                          data: Any, source: DataSource, quality: DataQuality, **kwargs) -> bool:
    """Convenience function to cache market data"""
    cache_key = generate_cache_key(ticker, data_type, **kwargs)
    return await cache.put(cache_key, data, source, quality)