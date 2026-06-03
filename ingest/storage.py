"""
Storage adapters for Redis and PostgreSQL.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


class RedisStorage:
    """Redis storage adapter for caching and queuing"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._pool: Optional[redis.ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None
    
    async def connect(self):
        """Establish Redis connection"""
        self._pool = redis.ConnectionPool.from_url(self.redis_url)
        self._redis = redis.Redis(connection_pool=self._pool)
        
        # Test connection
        try:
            await self._redis.ping()
            logger.info("Redis connection established", url=self.redis_url)
        except Exception as e:
            logger.error("Redis connection failed", error=str(e))
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.aclose()
        if self._pool:
            await self._pool.aclose()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            value = await self._redis.get(key)
            return value.decode('utf-8') if value else None
        except Exception as e:
            logger.error("Redis get failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with optional TTL"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            if ttl:
                await self._redis.setex(key, ttl, value)
            else:
                await self._redis.set(key, value)
            return True
        except Exception as e:
            logger.error("Redis set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            result = await self._redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error("Redis delete failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            result = await self._redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error("Redis exists check failed", key=key, error=str(e))
            return False
    
    # Queue operations
    async def push_job(self, queue: str, job_data: str) -> bool:
        """Push job to queue"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            await self._redis.lpush(queue, job_data)
            return True
        except Exception as e:
            logger.error("Redis push failed", queue=queue, error=str(e))
            return False
    
    async def pop_job(self, queue: str, timeout: int = 1) -> Optional[str]:
        """Pop job from queue with timeout"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            result = await self._redis.brpop(queue, timeout=timeout)
            if result:
                _, job_data = result
                return job_data.decode('utf-8')
            return None
        except Exception as e:
            logger.error("Redis pop failed", queue=queue, error=str(e))
            return None
    
    async def queue_length(self, queue: str) -> int:
        """Get queue length"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            return await self._redis.llen(queue)
        except Exception as e:
            logger.error("Redis queue length check failed", queue=queue, error=str(e))
            return 0
    
    # Set operations for deduplication
    async def add_to_set(self, set_key: str, value: str) -> bool:
        """Add value to set"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            result = await self._redis.sadd(set_key, value)
            return result > 0
        except Exception as e:
            logger.error("Redis set add failed", set_key=set_key, error=str(e))
            return False
    
    async def is_in_set(self, set_key: str, value: str) -> bool:
        """Check if value is in set"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            result = await self._redis.sismember(set_key, value)
            return bool(result)
        except Exception as e:
            logger.error("Redis set check failed", set_key=set_key, error=str(e))
            return False
    
    async def set_expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key"""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        try:
            result = await self._redis.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error("Redis expire failed", key=key, error=str(e))
            return False


class PostgreSQLStorage:
    """PostgreSQL storage adapter for persistent data"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._pool = None
    
    async def connect(self):
        """Establish PostgreSQL connection pool"""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool established")
        except ImportError:
            logger.warning("asyncpg not installed, PostgreSQL storage disabled")
            raise
        except Exception as e:
            logger.error("PostgreSQL connection failed", error=str(e))
            raise
    
    async def disconnect(self):
        """Close PostgreSQL connection pool"""
        if self._pool:
            await self._pool.close()
    
    async def create_tables(self):
        """Create necessary tables"""
        if not self._pool:
            raise RuntimeError("PostgreSQL not connected")
        
        create_articles_table = """
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            canonical_url TEXT,
            title TEXT NOT NULL,
            authors TEXT[],
            published_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            paywall_status TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            embedding vector(384)  -- Optional for vector similarity
        );
        """
        
        create_extractions_table = """
        CREATE TABLE IF NOT EXISTS extractions (
            id SERIAL PRIMARY KEY,
            url TEXT NOT NULL,
            success BOOLEAN NOT NULL,
            strategy TEXT NOT NULL,
            extraction_time_ms INTEGER NOT NULL,
            error_message TEXT,
            selectors_used TEXT[],
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        create_quality_decisions_table = """
        CREATE TABLE IF NOT EXISTS quality_decisions (
            id SERIAL PRIMARY KEY,
            url TEXT NOT NULL,
            keep BOOLEAN NOT NULL,
            reasons TEXT[],
            scores JSONB,
            duplicate_of TEXT,
            evaluation_time_ms INTEGER NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        create_sources_table = """
        CREATE TABLE IF NOT EXISTS sources (
            id SERIAL PRIMARY KEY,
            domain TEXT UNIQUE NOT NULL,
            credibility_score REAL NOT NULL DEFAULT 0.5,
            allowed BOOLEAN NOT NULL DEFAULT TRUE,
            crawl_delay_seconds REAL NOT NULL DEFAULT 12.0,
            max_concurrency INTEGER NOT NULL DEFAULT 1,
            robots_txt_url TEXT,
            robots_last_checked TIMESTAMPTZ,
            custom_selectors JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(create_articles_table)
                await conn.execute(create_extractions_table)
                await conn.execute(create_quality_decisions_table)
                await conn.execute(create_sources_table)
                
                # Create indexes
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_extractions_url ON extractions(url);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_url ON quality_decisions(url);")
                
                logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create tables", error=str(e))
            raise
    
    async def save_article(self, article: Dict[str, Any]) -> Optional[int]:
        """Save article to database"""
        if not self._pool:
            raise RuntimeError("PostgreSQL not connected")
        
        insert_query = """
        INSERT INTO articles (
            url, canonical_url, title, authors, published_at, updated_at,
            source, text, word_count, paywall_status, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (url) DO UPDATE SET
            canonical_url = EXCLUDED.canonical_url,
            title = EXCLUDED.title,
            authors = EXCLUDED.authors,
            published_at = EXCLUDED.published_at,
            updated_at = EXCLUDED.updated_at,
            text = EXCLUDED.text,
            word_count = EXCLUDED.word_count,
            paywall_status = EXCLUDED.paywall_status,
            metadata = EXCLUDED.metadata
        RETURNING id;
        """
        
        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(
                    insert_query,
                    article['url'],
                    article.get('canonical_url'),
                    article['title'],
                    article.get('authors', []),
                    article.get('published_at'),
                    article.get('updated_at'),
                    article['source'],
                    article['text'],
                    article['word_count'],
                    article['paywall_status'],
                    json.dumps(article.get('metadata', {}))
                )
                return result
        except Exception as e:
            logger.error("Failed to save article", url=article.get('url'), error=str(e))
            return None
    
    async def get_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Get article by URL"""
        if not self._pool:
            raise RuntimeError("PostgreSQL not connected")
        
        query = "SELECT * FROM articles WHERE url = $1;"
        
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, url)
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error("Failed to get article", url=url, error=str(e))
            return None
    
    async def save_extraction_log(self, extraction: Dict[str, Any]) -> bool:
        """Save extraction log"""
        if not self._pool:
            raise RuntimeError("PostgreSQL not connected")
        
        insert_query = """
        INSERT INTO extractions (
            url, success, strategy, extraction_time_ms, error_message, selectors_used
        ) VALUES ($1, $2, $3, $4, $5, $6);
        """
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    extraction['url'],
                    extraction['success'],
                    extraction['strategy'],
                    extraction['extraction_time_ms'],
                    extraction.get('error_message'),
                    extraction.get('selectors_used', [])
                )
                return True
        except Exception as e:
            logger.error("Failed to save extraction log", error=str(e))
            return False
    
    async def save_quality_decision(self, decision: Dict[str, Any]) -> bool:
        """Save quality decision"""
        if not self._pool:
            raise RuntimeError("PostgreSQL not connected")
        
        insert_query = """
        INSERT INTO quality_decisions (
            url, keep, reasons, scores, duplicate_of, evaluation_time_ms
        ) VALUES ($1, $2, $3, $4, $5, $6);
        """
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    decision['url'],
                    decision['keep'],
                    decision.get('reasons', []),
                    json.dumps(decision.get('scores', {})),
                    decision.get('duplicate_of'),
                    decision['evaluation_time_ms']
                )
                return True
        except Exception as e:
            logger.error("Failed to save quality decision", error=str(e))
            return False
    
    async def get_recent_articles(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent articles"""
        if not self._pool:
            raise RuntimeError("PostgreSQL not connected")
        
        query = """
        SELECT * FROM articles 
        WHERE created_at >= NOW() - INTERVAL '%s hours'
        ORDER BY created_at DESC 
        LIMIT $1;
        """ % hours
        
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, limit)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to get recent articles", error=str(e))
            return []