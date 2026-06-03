#!/usr/bin/env python3
"""
Vector search store for semantic news similarity
Supports multiple backends: Qdrant, pgvector, CSV fallback
Optional embeddings generation with sentence-transformers placeholder
"""

import os
import csv
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Protocol
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import numpy as np

from config.feature_flags import is_vector_search_enabled

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """News item for vector storage"""
    headline: str
    url: str
    published_at: datetime
    symbol: str
    embedding: Optional[List[float]] = None
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        """Generate content hash for deduplication"""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.url.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """Vector search result"""
    news_item: NewsItem
    similarity_score: float
    distance: float


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the vector store"""
        pass
    
    @abstractmethod
    async def put_news(self, items: List[NewsItem]) -> int:
        """Store news items with embeddings"""
        pass
    
    @abstractmethod
    async def search_similar(self, query: str, symbol: Optional[str] = None, limit: int = 10) -> List[SearchResult]:
        """Search for similar news items"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources"""
        pass


class EmbeddingGenerator:
    """Placeholder embedding generator using sentence-transformers"""
    
    def __init__(self):
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight model
        self.embedding_dim = 384
        
    async def initialize(self) -> bool:
        """Initialize embedding model if available"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, using placeholder embeddings")
            return True
            
        try:
            # In a real implementation, load the model
            # self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model {self.model_name} ready (placeholder)")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        if self.model and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Real implementation would use:
            # embeddings = self.model.encode(texts)
            # return embeddings.tolist()
            pass
            
        # Placeholder: generate random normalized vectors
        embeddings = []
        for text in texts:
            # Create deterministic "embedding" from text hash for consistency
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            # Convert hash to float values and normalize
            hash_ints = [int(text_hash[i:i+2], 16) for i in range(0, min(len(text_hash), self.embedding_dim * 2), 2)]
            hash_ints = hash_ints[:self.embedding_dim]  # Ensure exact dimension
            
            # Pad if needed
            while len(hash_ints) < self.embedding_dim:
                hash_ints.append(128)  # Midpoint value
            
            # Normalize to [-1, 1] range
            normalized = [(x - 128) / 128.0 for x in hash_ints]
            
            # L2 normalize
            norm = sum(x * x for x in normalized) ** 0.5
            if norm > 0:
                normalized = [x / norm for x in normalized]
            
            embeddings.append(normalized)
        
        return embeddings


class QdrantVectorStore(VectorStoreInterface):
    """Qdrant vector database implementation"""
    
    def __init__(self):
        self.client = None
        self.collection_name = "news_embeddings"
        self.url = os.getenv('QDRANT_URL')
        self.api_key = os.getenv('QDRANT_KEY')
        self.embedding_dim = 384
        
    async def initialize(self) -> bool:
        """Initialize Qdrant client"""
        if not QDRANT_AVAILABLE:
            logger.warning("qdrant-client not available")
            return False
            
        if not self.url:
            logger.debug("No QDRANT_URL configured")
            return False
            
        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30
            )
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            
            logger.info("Qdrant vector store initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            return False
    
    async def put_news(self, items: List[NewsItem]) -> int:
        """Store news items in Qdrant"""
        if not self.client or not items:
            return 0
            
        points = []
        for item in items:
            if not item.embedding:
                continue
                
            point = PointStruct(
                id=item.content_hash,
                vector=item.embedding,
                payload={
                    "headline": item.headline,
                    "url": item.url,
                    "published_at": item.published_at.isoformat(),
                    "symbol": item.symbol,
                    "content_hash": item.content_hash
                }
            )
            points.append(point)
        
        if not points:
            return 0
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.debug(f"Stored {len(points)} news items in Qdrant")
            return len(points)
            
        except Exception as e:
            logger.error(f"Failed to store in Qdrant: {e}")
            return 0
    
    async def search_similar(self, query: str, symbol: Optional[str] = None, limit: int = 10) -> List[SearchResult]:
        """Search for similar news in Qdrant"""
        if not self.client:
            return []
            
        # Generate query embedding (placeholder)
        embedding_gen = EmbeddingGenerator()
        query_embeddings = await embedding_gen.generate_embeddings([query])
        query_vector = query_embeddings[0]
        
        try:
            # Build filter if symbol provided
            search_filter = None
            if symbol:
                search_filter = {
                    "must": [
                        {"key": "symbol", "match": {"value": symbol}}
                    ]
                }
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit
            )
            
            search_results = []
            for result in results:
                news_item = NewsItem(
                    headline=result.payload["headline"],
                    url=result.payload["url"],
                    published_at=datetime.fromisoformat(result.payload["published_at"]),
                    symbol=result.payload["symbol"],
                    content_hash=result.payload["content_hash"]
                )
                
                search_result = SearchResult(
                    news_item=news_item,
                    similarity_score=result.score,
                    distance=1.0 - result.score
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics"""
        if not self.client:
            return {"backend": "qdrant", "available": False}
            
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "backend": "qdrant",
                "available": True,
                "vector_count": collection_info.points_count,
                "collection_name": self.collection_name,
                "embedding_dim": self.embedding_dim
            }
        except Exception as e:
            logger.error(f"Failed to get Qdrant stats: {e}")
            return {"backend": "qdrant", "available": False, "error": str(e)}
    
    async def cleanup(self):
        """Clean up Qdrant resources"""
        if self.client:
            # Qdrant client doesn't require explicit cleanup
            pass


class PgVectorStore(VectorStoreInterface):
    """pgvector PostgreSQL implementation"""
    
    def __init__(self):
        self.pool = None
        self.dsn = os.getenv('PG_DSN')
        self.table_name = "news_vectors"
        self.embedding_dim = 384
        
    async def initialize(self) -> bool:
        """Initialize pgvector connection"""
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available for pgvector")
            return False
            
        if not self.dsn:
            logger.debug("No PG_DSN configured for pgvector")
            return False
            
        try:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            
            # Create pgvector extension and table
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        content_hash VARCHAR(16) PRIMARY KEY,
                        headline TEXT NOT NULL,
                        url TEXT NOT NULL,
                        published_at TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(10) NOT NULL,
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Create index for vector similarity search
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding 
                    ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                """)
                
                # Create index for symbol filtering
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_symbol 
                    ON {self.table_name} (symbol)
                """)
            
            logger.info("pgvector store initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pgvector: {e}")
            return False
    
    async def put_news(self, items: List[NewsItem]) -> int:
        """Store news items in pgvector"""
        if not self.pool or not items:
            return 0
            
        stored_count = 0
        async with self.pool.acquire() as conn:
            for item in items:
                if not item.embedding:
                    continue
                    
                try:
                    await conn.execute(f"""
                        INSERT INTO {self.table_name} 
                        (content_hash, headline, url, published_at, symbol, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (content_hash) DO UPDATE SET
                            headline = EXCLUDED.headline,
                            url = EXCLUDED.url,
                            published_at = EXCLUDED.published_at,
                            symbol = EXCLUDED.symbol,
                            embedding = EXCLUDED.embedding
                    """, item.content_hash, item.headline, item.url, 
                         item.published_at, item.symbol, item.embedding)
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to store news item: {e}")
        
        logger.debug(f"Stored {stored_count} news items in pgvector")
        return stored_count
    
    async def search_similar(self, query: str, symbol: Optional[str] = None, limit: int = 10) -> List[SearchResult]:
        """Search for similar news in pgvector"""
        if not self.pool:
            return []
            
        # Generate query embedding (placeholder)
        embedding_gen = EmbeddingGenerator()
        query_embeddings = await embedding_gen.generate_embeddings([query])
        query_vector = query_embeddings[0]
        
        try:
            async with self.pool.acquire() as conn:
                if symbol:
                    results = await conn.fetch(f"""
                        SELECT content_hash, headline, url, published_at, symbol,
                               1 - (embedding <=> $1) as similarity_score
                        FROM {self.table_name}
                        WHERE symbol = $2
                        ORDER BY embedding <=> $1
                        LIMIT $3
                    """, query_vector, symbol, limit)
                else:
                    results = await conn.fetch(f"""
                        SELECT content_hash, headline, url, published_at, symbol,
                               1 - (embedding <=> $1) as similarity_score
                        FROM {self.table_name}
                        ORDER BY embedding <=> $1
                        LIMIT $2
                    """, query_vector, limit)
                
                search_results = []
                for row in results:
                    news_item = NewsItem(
                        headline=row['headline'],
                        url=row['url'],
                        published_at=row['published_at'],
                        symbol=row['symbol'],
                        content_hash=row['content_hash']
                    )
                    
                    search_result = SearchResult(
                        news_item=news_item,
                        similarity_score=row['similarity_score'],
                        distance=1.0 - row['similarity_score']
                    )
                    search_results.append(search_result)
                
                return search_results
                
        except Exception as e:
            logger.error(f"Failed to search pgvector: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pgvector table statistics"""
        if not self.pool:
            return {"backend": "pgvector", "available": False}
            
        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name}")
                return {
                    "backend": "pgvector",
                    "available": True,
                    "vector_count": count,
                    "table_name": self.table_name,
                    "embedding_dim": self.embedding_dim
                }
        except Exception as e:
            logger.error(f"Failed to get pgvector stats: {e}")
            return {"backend": "pgvector", "available": False, "error": str(e)}
    
    async def cleanup(self):
        """Clean up pgvector resources"""
        if self.pool:
            await self.pool.close()


class CSVFallbackStore(VectorStoreInterface):
    """CSV fallback implementation for development/testing"""
    
    def __init__(self):
        self.csv_path = Path("data/vector_store.csv")
        self.embedding_dim = 384
        self._items = []  # In-memory cache
        
    async def initialize(self) -> bool:
        """Initialize CSV store"""
        try:
            # Create data directory
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing data
            if self.csv_path.exists():
                await self._load_from_csv()
            
            logger.info(f"CSV fallback store initialized: {self.csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CSV store: {e}")
            return False
    
    async def _load_from_csv(self):
        """Load data from CSV file"""
        self._items = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                embedding = None
                if row.get('embedding'):
                    try:
                        embedding = json.loads(row['embedding'])
                    except:
                        pass
                
                item = NewsItem(
                    headline=row['headline'],
                    url=row['url'],
                    published_at=datetime.fromisoformat(row['published_at']),
                    symbol=row['symbol'],
                    embedding=embedding,
                    content_hash=row['content_hash']
                )
                self._items.append(item)
    
    async def _save_to_csv(self):
        """Save data to CSV file"""
        with open(self.csv_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['content_hash', 'headline', 'url', 'published_at', 'symbol', 'embedding']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in self._items:
                writer.writerow({
                    'content_hash': item.content_hash,
                    'headline': item.headline,
                    'url': item.url,
                    'published_at': item.published_at.isoformat(),
                    'symbol': item.symbol,
                    'embedding': json.dumps(item.embedding) if item.embedding else ''
                })
    
    async def put_news(self, items: List[NewsItem]) -> int:
        """Store news items in CSV"""
        if not items:
            return 0
            
        # Dedup by content_hash
        existing_hashes = {item.content_hash for item in self._items}
        new_items = [item for item in items if item.content_hash not in existing_hashes]
        
        self._items.extend(new_items)
        await self._save_to_csv()
        
        logger.debug(f"Stored {len(new_items)} news items in CSV")
        return len(new_items)
    
    async def search_similar(self, query: str, symbol: Optional[str] = None, limit: int = 10) -> List[SearchResult]:
        """Search for similar news in CSV (simple text matching fallback)"""
        # Generate query embedding for consistency
        embedding_gen = EmbeddingGenerator()
        query_embeddings = await embedding_gen.generate_embeddings([query])
        query_vector = query_embeddings[0]
        
        results = []
        query_lower = query.lower()
        
        for item in self._items:
            # Filter by symbol if provided
            if symbol and item.symbol != symbol:
                continue
            
            # Simple similarity: text overlap + embedding cosine similarity
            headline_lower = item.headline.lower()
            text_similarity = len(set(query_lower.split()) & set(headline_lower.split())) / max(len(set(query_lower.split())), 1)
            
            # Embedding similarity if available
            embedding_similarity = 0.0
            if item.embedding and len(item.embedding) == len(query_vector):
                try:
                    # Cosine similarity
                    dot_product = sum(a * b for a, b in zip(query_vector, item.embedding))
                    norm_a = sum(a * a for a in query_vector) ** 0.5
                    norm_b = sum(b * b for b in item.embedding) ** 0.5
                    if norm_a > 0 and norm_b > 0:
                        embedding_similarity = dot_product / (norm_a * norm_b)
                except:
                    pass
            
            # Combined similarity score
            combined_similarity = 0.3 * text_similarity + 0.7 * embedding_similarity
            
            if combined_similarity > 0:
                search_result = SearchResult(
                    news_item=item,
                    similarity_score=combined_similarity,
                    distance=1.0 - combined_similarity
                )
                results.append(search_result)
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:limit]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get CSV store statistics"""
        return {
            "backend": "csv_fallback",
            "available": True,
            "vector_count": len(self._items),
            "csv_path": str(self.csv_path),
            "embedding_dim": self.embedding_dim
        }
    
    async def cleanup(self):
        """Clean up CSV resources"""
        # Save final state
        if self._items:
            await self._save_to_csv()


class VectorStore:
    """Main vector store interface with automatic backend selection"""
    
    def __init__(self):
        self.backend: Optional[VectorStoreInterface] = None
        self.embedding_generator = EmbeddingGenerator()
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize vector store with best available backend"""
        if not is_vector_search_enabled():
            logger.debug("Vector search disabled by feature flag")
            return False
            
        # Try backends in priority order
        backends = [
            ("qdrant", QdrantVectorStore),
            ("pgvector", PgVectorStore),
            ("csv_fallback", CSVFallbackStore)
        ]
        
        for backend_name, backend_class in backends:
            try:
                backend = backend_class()
                success = await backend.initialize()
                if success:
                    self.backend = backend
                    await self.embedding_generator.initialize()
                    self._initialized = True
                    logger.info(f"Vector store initialized with {backend_name} backend")
                    return True
            except Exception as e:
                logger.warning(f"Failed to initialize {backend_name}: {e}")
        
        logger.error("All vector store backends failed to initialize")
        return False
    
    async def put_news(self, items: List[NewsItem]) -> int:
        """Store news items with automatic embedding generation"""
        if not self._initialized or not self.backend:
            return 0
            
        # Generate embeddings for items that don't have them
        items_needing_embeddings = [item for item in items if not item.embedding]
        if items_needing_embeddings:
            texts = [item.headline for item in items_needing_embeddings]
            embeddings = await self.embedding_generator.generate_embeddings(texts)
            
            for item, embedding in zip(items_needing_embeddings, embeddings):
                item.embedding = embedding
        
        return await self.backend.put_news(items)
    
    async def search_similar(self, query: str, symbol: Optional[str] = None, limit: int = 10) -> List[SearchResult]:
        """Search for similar news items"""
        if not self._initialized or not self.backend:
            return []
            
        return await self.backend.search_similar(query, symbol, limit)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        base_stats = {
            "enabled": is_vector_search_enabled(),
            "initialized": self._initialized,
            "embedding_generator": {
                "available": SENTENCE_TRANSFORMERS_AVAILABLE,
                "model_name": self.embedding_generator.model_name,
                "embedding_dim": self.embedding_generator.embedding_dim
            }
        }
        
        if self.backend:
            backend_stats = await self.backend.get_stats()
            base_stats.update(backend_stats)
        
        return base_stats
    
    async def cleanup(self):
        """Clean up vector store resources"""
        if self.backend:
            await self.backend.cleanup()
        self._initialized = False


# Global instance
vector_store = VectorStore()


# Convenience functions
async def initialize_vector_store() -> bool:
    """Initialize the global vector store"""
    return await vector_store.initialize()


async def store_news_embeddings(items: List[NewsItem]) -> int:
    """Store news items with embeddings"""
    return await vector_store.put_news(items)


async def search_similar_news(query: str, symbol: Optional[str] = None, limit: int = 10) -> List[SearchResult]:
    """Search for similar news items"""
    return await vector_store.search_similar(query, symbol, limit)