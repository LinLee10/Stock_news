#!/usr/bin/env python3
"""
Tests for F11 Vector search feature
Validates vector store interface, embedding generation, and search functionality
"""

import pytest
import os
import json
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List

# Test imports
from services.vector_store import (
    VectorStore, NewsItem, SearchResult, EmbeddingGenerator,
    QdrantVectorStore, PgVectorStore, CSVFallbackStore,
    VectorStoreInterface, initialize_vector_store, store_news_embeddings, 
    search_similar_news, vector_store
)
from config.feature_flags import is_vector_search_enabled


class TestNewsItem:
    """Test NewsItem dataclass"""
    
    def test_news_item_creation(self):
        """Test NewsItem creation and hash generation"""
        item = NewsItem(
            headline="Apple Reports Strong Q4 Earnings",
            url="https://example.com/apple-earnings",
            published_at=datetime(2023, 12, 1),
            symbol="AAPL"
        )
        
        assert item.headline == "Apple Reports Strong Q4 Earnings"
        assert item.symbol == "AAPL"
        assert item.embedding is None
        assert item.content_hash is not None
        assert len(item.content_hash) == 16
    
    def test_news_item_hash_consistency(self):
        """Test that same URL produces same hash"""
        item1 = NewsItem("Title", "https://example.com/test", datetime.now(), "AAPL")
        item2 = NewsItem("Different Title", "https://example.com/test", datetime.now(), "MSFT")
        
        # Same URL should produce same hash
        assert item1.content_hash == item2.content_hash
    
    def test_news_item_with_embedding(self):
        """Test NewsItem with pre-set embedding"""
        embedding = [0.1, 0.2, 0.3]
        item = NewsItem(
            headline="Test headline",
            url="https://example.com/test",
            published_at=datetime.now(),
            symbol="AAPL",
            embedding=embedding
        )
        
        assert item.embedding == embedding


class TestEmbeddingGenerator:
    """Test embedding generation"""
    
    @pytest.fixture
    def generator(self):
        """Create embedding generator for testing"""
        return EmbeddingGenerator()
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, generator):
        """Test successful initialization"""
        success = await generator.initialize()
        assert success is True
    
    @pytest.mark.asyncio
    async def test_generate_placeholder_embeddings(self, generator):
        """Test placeholder embedding generation"""
        await generator.initialize()
        
        texts = ["Apple stock rises", "Tesla reports earnings", "Market volatility"]
        embeddings = await generator.generate_embeddings(texts)
        
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert len(embedding) == generator.embedding_dim
            assert isinstance(embedding, list)
            assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_deterministic_embeddings(self, generator):
        """Test that same text produces same embedding"""
        await generator.initialize()
        
        text = "Apple reports strong earnings"
        embedding1 = await generator.generate_embeddings([text])
        embedding2 = await generator.generate_embeddings([text])
        
        assert embedding1 == embedding2
    
    @pytest.mark.asyncio
    async def test_normalized_embeddings(self, generator):
        """Test that embeddings are L2 normalized"""
        await generator.initialize()
        
        embeddings = await generator.generate_embeddings(["test text"])
        embedding = embeddings[0]
        
        # Calculate L2 norm
        norm = sum(x * x for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 0.001  # Should be normalized


class TestCSVFallbackStore:
    """Test CSV fallback vector store"""
    
    @pytest.fixture
    def temp_csv_path(self):
        """Create temporary CSV file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    def csv_store(self, temp_csv_path):
        """Create CSV store with temporary file"""
        store = CSVFallbackStore()
        store.csv_path = temp_csv_path
        return store
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, csv_store):
        """Test CSV store initialization"""
        success = await csv_store.initialize()
        assert success is True
        assert csv_store.csv_path.parent.exists()
    
    @pytest.mark.asyncio
    async def test_put_and_retrieve_news(self, csv_store):
        """Test storing and retrieving news items"""
        await csv_store.initialize()
        
        items = [
            NewsItem("Apple earnings beat expectations", "https://example.com/apple1", datetime.now(), "AAPL"),
            NewsItem("Tesla stock surges on delivery numbers", "https://example.com/tesla1", datetime.now(), "TSLA")
        ]
        
        # Store items
        count = await csv_store.put_news(items)
        assert count == 2
        
        # Verify items are in memory cache
        assert len(csv_store._items) == 2
        
        # Verify CSV file was created
        assert csv_store.csv_path.exists()
    
    @pytest.mark.asyncio
    async def test_deduplication(self, csv_store):
        """Test that duplicate items are not stored twice"""
        await csv_store.initialize()
        
        item1 = NewsItem("Same news", "https://example.com/same", datetime.now(), "AAPL")
        item2 = NewsItem("Same news different title", "https://example.com/same", datetime.now(), "AAPL")
        
        # Store first item
        count1 = await csv_store.put_news([item1])
        assert count1 == 1
        
        # Store duplicate (same URL)
        count2 = await csv_store.put_news([item2])
        assert count2 == 0  # Should not store duplicate
        
        assert len(csv_store._items) == 1
    
    @pytest.mark.asyncio
    async def test_search_similar_text_matching(self, csv_store):
        """Test basic text similarity search"""
        await csv_store.initialize()
        
        items = [
            NewsItem("Apple reports strong quarterly earnings", "https://example.com/1", datetime.now(), "AAPL"),
            NewsItem("Tesla delivers record number of vehicles", "https://example.com/2", datetime.now(), "TSLA"),
            NewsItem("Apple stock price rises after earnings", "https://example.com/3", datetime.now(), "AAPL")
        ]
        
        await csv_store.put_news(items)
        
        # Search for Apple-related news
        results = await csv_store.search_similar("Apple earnings", limit=5)
        
        assert len(results) >= 2  # Should find Apple-related articles
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.similarity_score >= 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_search_with_symbol_filter(self, csv_store):
        """Test search with symbol filtering"""
        await csv_store.initialize()
        
        items = [
            NewsItem("Apple reports earnings", "https://example.com/1", datetime.now(), "AAPL"),
            NewsItem("Tesla reports earnings", "https://example.com/2", datetime.now(), "TSLA"),
            NewsItem("Microsoft reports earnings", "https://example.com/3", datetime.now(), "MSFT")
        ]
        
        await csv_store.put_news(items)
        
        # Search only for AAPL news
        results = await csv_store.search_similar("earnings", symbol="AAPL", limit=5)
        
        assert len(results) == 1
        assert results[0].news_item.symbol == "AAPL"
    
    @pytest.mark.asyncio
    async def test_get_stats(self, csv_store):
        """Test CSV store statistics"""
        await csv_store.initialize()
        
        items = [NewsItem("Test", f"https://example.com/{i}", datetime.now(), "AAPL") for i in range(5)]
        await csv_store.put_news(items)
        
        stats = await csv_store.get_stats()
        
        assert stats["backend"] == "csv_fallback"
        assert stats["available"] is True
        assert stats["vector_count"] == 5
        assert "csv_path" in stats
    
    @pytest.mark.asyncio
    async def test_persistence_across_sessions(self, csv_store):
        """Test that data persists across store instances"""
        await csv_store.initialize()
        
        # Store some items
        items = [NewsItem("Persistent news", "https://example.com/persist", datetime.now(), "AAPL")]
        await csv_store.put_news(items)
        
        # Create new store instance with same path
        new_store = CSVFallbackStore()
        new_store.csv_path = csv_store.csv_path
        await new_store.initialize()
        
        # Should load existing data
        assert len(new_store._items) == 1
        assert new_store._items[0].headline == "Persistent news"


class TestVectorStoreInterface:
    """Test main VectorStore interface"""
    
    @pytest.fixture
    def vector_store_instance(self):
        """Create VectorStore instance for testing"""
        return VectorStore()
    
    @pytest.mark.asyncio
    async def test_no_initialize_when_disabled(self, vector_store_instance):
        """Test that vector store doesn't initialize when feature disabled"""
        with patch('services.vector_store.is_vector_search_enabled', return_value=False):
            success = await vector_store_instance.initialize()
            assert success is False
            assert vector_store_instance._initialized is False
    
    @pytest.mark.asyncio
    async def test_fallback_to_csv(self, vector_store_instance):
        """Test fallback to CSV when other backends fail"""
        with patch('services.vector_store.is_vector_search_enabled', return_value=True):
            with patch('services.vector_store.QdrantVectorStore.initialize', return_value=False):
                with patch('services.vector_store.PgVectorStore.initialize', return_value=False):
                    with patch('services.vector_store.CSVFallbackStore.initialize', return_value=True):
                        success = await vector_store_instance.initialize()
                        assert success is True
                        assert isinstance(vector_store_instance.backend, CSVFallbackStore)
    
    @pytest.mark.asyncio
    async def test_automatic_embedding_generation(self, vector_store_instance):
        """Test automatic embedding generation for news items"""
        # Mock CSV backend initialization
        with patch('services.vector_store.is_vector_search_enabled', return_value=True):
            with patch('services.vector_store.QdrantVectorStore.initialize', return_value=False):
                with patch('services.vector_store.PgVectorStore.initialize', return_value=False):
                    csv_store_mock = AsyncMock()
                    csv_store_mock.initialize.return_value = True
                    csv_store_mock.put_news.return_value = 1
                    
                    with patch('services.vector_store.CSVFallbackStore', return_value=csv_store_mock):
                        await vector_store_instance.initialize()
                        
                        # Create news items without embeddings
                        items = [NewsItem("Test news", "https://example.com/test", datetime.now(), "AAPL")]
                        
                        count = await vector_store_instance.put_news(items)
                        assert count == 1
                        
                        # Verify embeddings were generated
                        assert items[0].embedding is not None
                        assert len(items[0].embedding) == 384  # Default embedding dimension
    
    @pytest.mark.asyncio
    async def test_no_op_when_not_initialized(self, vector_store_instance):
        """Test that operations are no-op when not initialized"""
        assert vector_store_instance._initialized is False
        
        # All operations should be no-op
        items = [NewsItem("Test", "https://example.com/test", datetime.now(), "AAPL")]
        count = await vector_store_instance.put_news(items)
        assert count == 0
        
        results = await vector_store_instance.search_similar("test query")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_get_stats(self, vector_store_instance):
        """Test statistics collection"""
        stats = await vector_store_instance.get_stats()
        
        assert "enabled" in stats
        assert "initialized" in stats
        assert "embedding_generator" in stats
        assert stats["embedding_generator"]["embedding_dim"] == 384


class TestQdrantVectorStore:
    """Test Qdrant vector store (mock tests)"""
    
    @pytest.fixture
    def qdrant_store(self):
        """Create Qdrant store for testing"""
        return QdrantVectorStore()
    
    @pytest.mark.asyncio
    async def test_no_initialize_without_url(self, qdrant_store):
        """Test that Qdrant doesn't initialize without URL"""
        with patch.dict(os.environ, {}, clear=True):
            success = await qdrant_store.initialize()
            assert success is False
    
    @pytest.mark.asyncio
    async def test_no_initialize_without_client(self, qdrant_store):
        """Test that Qdrant doesn't initialize without client library"""
        with patch('services.vector_store.QDRANT_AVAILABLE', False):
            success = await qdrant_store.initialize()
            assert success is False
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'QDRANT_URL': 'http://localhost:6333'})
    @patch('services.vector_store.QDRANT_AVAILABLE', True)
    @patch('services.vector_store.QdrantClient')
    async def test_successful_initialization(self, mock_qdrant_client, qdrant_store):
        """Test successful Qdrant initialization"""
        # Mock Qdrant client
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_client
        
        success = await qdrant_store.initialize()
        assert success is True
        assert qdrant_store.client is not None
        
        # Verify collection creation was called
        mock_client.create_collection.assert_called_once()


class TestPgVectorStore:
    """Test pgvector PostgreSQL store (mock tests)"""
    
    @pytest.fixture
    def pgvector_store(self):
        """Create pgvector store for testing"""
        return PgVectorStore()
    
    @pytest.mark.asyncio
    async def test_no_initialize_without_dsn(self, pgvector_store):
        """Test that pgvector doesn't initialize without DSN"""
        with patch.dict(os.environ, {}, clear=True):
            success = await pgvector_store.initialize()
            assert success is False
    
    @pytest.mark.asyncio
    async def test_no_initialize_without_asyncpg(self, pgvector_store):
        """Test that pgvector doesn't initialize without asyncpg"""
        with patch('services.vector_store.ASYNCPG_AVAILABLE', False):
            success = await pgvector_store.initialize()
            assert success is False
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'PG_DSN': 'postgres://localhost/test'})
    @patch('services.vector_store.ASYNCPG_AVAILABLE', True)
    @patch('services.vector_store.asyncpg')
    async def test_successful_initialization(self, mock_asyncpg, pgvector_store):
        """Test successful pgvector initialization"""
        # Mock asyncpg pool and connection
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_asyncpg.create_pool.return_value = mock_pool
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None
        
        success = await pgvector_store.initialize()
        assert success is True
        assert pgvector_store.pool is not None
        
        # Verify table and index creation
        assert mock_conn.execute.call_count >= 3  # Extension + table + indexes


class TestFeatureFlagIntegration:
    """Test feature flag integration"""
    
    @patch('config.feature_flags.feature_flags.is_enabled')
    def test_feature_flag_disabled(self, mock_is_enabled):
        """Test behavior when feature flag is disabled"""
        mock_is_enabled.return_value = False
        
        assert is_vector_search_enabled() is False
    
    @patch('config.feature_flags.feature_flags.is_enabled')  
    def test_feature_flag_enabled(self, mock_is_enabled):
        """Test behavior when feature flag is enabled"""
        mock_is_enabled.return_value = True
        
        assert is_vector_search_enabled() is True


class TestConvenienceFunctions:
    """Test convenience functions and global vector store"""
    
    @pytest.fixture(autouse=True)
    def reset_global_vector_store(self):
        """Reset global vector store state before each test"""
        vector_store._initialized = False
        vector_store.backend = None
        yield
        # Cleanup after test
        asyncio.run(vector_store.cleanup())
    
    @pytest.mark.asyncio
    async def test_initialize_vector_store_disabled(self):
        """Test initialize_vector_store when disabled"""
        with patch('services.vector_store.is_vector_search_enabled', return_value=False):
            result = await initialize_vector_store()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_convenience_functions_no_op(self):
        """Test that convenience functions are no-op when not initialized"""
        # Vector store not initialized
        assert vector_store._initialized is False
        
        # Create sample data
        items = [NewsItem("Test headline", "https://example.com/test", datetime.now(), "AAPL")]
        
        # Should return 0 (no-op)
        result = await store_news_embeddings(items)
        assert result == 0
        
        # Should return empty list (no-op)
        results = await search_similar_news("test query")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_convenience_functions_with_csv_backend(self):
        """Test convenience functions with CSV backend"""
        with patch('services.vector_store.is_vector_search_enabled', return_value=True):
            with patch('services.vector_store.QdrantVectorStore.initialize', return_value=False):
                with patch('services.vector_store.PgVectorStore.initialize', return_value=False):
                    # Use a temporary file for CSV store
                    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                        temp_path = Path(f.name)
                    
                    try:
                        with patch('services.vector_store.CSVFallbackStore') as mock_csv_class:
                            mock_csv_store = AsyncMock()
                            mock_csv_store.initialize.return_value = True
                            mock_csv_store.put_news.return_value = 1
                            mock_csv_store.search_similar.return_value = []
                            mock_csv_class.return_value = mock_csv_store
                            
                            # Initialize global vector store
                            success = await initialize_vector_store()
                            assert success is True
                            
                            # Test convenience functions
                            items = [NewsItem("Test", "https://example.com/test", datetime.now(), "AAPL")]
                            count = await store_news_embeddings(items)
                            assert count == 1
                            
                            results = await search_similar_news("test query")
                            assert isinstance(results, list)
                    
                    finally:
                        if temp_path.exists():
                            temp_path.unlink()


class TestSearchResult:
    """Test SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test SearchResult creation"""
        news_item = NewsItem("Test headline", "https://example.com/test", datetime.now(), "AAPL")
        result = SearchResult(
            news_item=news_item,
            similarity_score=0.85,
            distance=0.15
        )
        
        assert result.news_item == news_item
        assert result.similarity_score == 0.85
        assert result.distance == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])