"""
Tests for content quality filtering.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ..models import ExtractedArticle, PaywallStatus, QualityDecision
from ..quality import (
    FinancialRelevanceScorer, DuplicationDetector, CredibilityScorer, 
    ContentQualityFilter
)
from ..storage import RedisStorage


class TestFinancialRelevanceScorer:
    """Test financial relevance scoring"""
    
    def test_keyword_scoring(self):
        """Test keyword-based scoring"""
        scorer = FinancialRelevanceScorer()
        
        # High relevance financial text
        financial_text = """
        Apple reported strong quarterly earnings today, with revenue beating 
        analyst expectations. The stock price rose 5% in after-hours trading 
        as investors welcomed the positive results. CEO Tim Cook highlighted 
        strong performance in the services division.
        """
        
        score = scorer._calculate_keyword_score(financial_text.lower())
        assert score > 0.3  # Should be high relevance
        
        # Low relevance non-financial text
        non_financial_text = """
        The weather was beautiful today. I went to the park and enjoyed 
        a nice walk. The flowers were blooming and the birds were singing.
        This has nothing to do with finance or markets.
        """
        
        score = scorer._calculate_keyword_score(non_financial_text.lower())
        assert score < 0.2  # Should be low relevance


class TestCredibilityScorer:
    """Test source credibility scoring"""
    
    def test_known_sources(self):
        """Test scoring for known financial sources"""
        scorer = CredibilityScorer()
        
        # High credibility sources
        assert scorer.get_credibility_score("Reuters", "reuters.com") == 0.95
        assert scorer.get_credibility_score("Bloomberg", "bloomberg.com") == 0.95
        assert scorer.get_credibility_score("Wall Street Journal", "wsj.com") == 0.95
        
        # Medium credibility sources
        assert scorer.get_credibility_score("CNBC", "cnbc.com") == 0.90
        assert scorer.get_credibility_score("MarketWatch", "marketwatch.com") == 0.90
        
        # Lower credibility sources
        assert scorer.get_credibility_score("Seeking Alpha", "seekingalpha.com") == 0.70
    
    def test_unknown_source(self):
        """Test scoring for unknown sources"""
        scorer = CredibilityScorer()
        
        # Unknown source should get default score
        score = scorer.get_credibility_score("Unknown Blog", "unknown-blog.com")
        assert score == 0.5


class TestDuplicationDetector:
    """Test duplicate content detection"""
    
    @pytest.fixture
    def mock_storage(self):
        """Mock Redis storage"""
        storage = AsyncMock(spec=RedisStorage)
        storage.get.return_value = None
        storage.set.return_value = True
        return storage
    
    def test_create_minhash(self, mock_storage):
        """Test MinHash creation"""
        detector = DuplicationDetector(mock_storage)
        
        text1 = "The stock market closed higher today on positive earnings news"
        text2 = "Stock market closes up on good earnings reports today"
        text3 = "Weather forecast shows sunny skies for the weekend"
        
        minhash1 = detector._create_minhash(text1)
        minhash2 = detector._create_minhash(text2)
        minhash3 = detector._create_minhash(text3)
        
        # Similar content should have similar hashes
        similarity12 = minhash1.jaccard(minhash2)
        similarity13 = minhash1.jaccard(minhash3)
        
        assert similarity12 > 0.3  # Similar financial content
        assert similarity13 < 0.2  # Different topics
    
    def test_normalize_text(self, mock_storage):
        """Test text normalization"""
        detector = DuplicationDetector(mock_storage)
        
        text = "The STOCK market CLOSED higher today, with strong trading volume!"
        normalized = detector._normalize_text(text)
        
        # Should be lowercase, no punctuation, no stop words
        assert normalized.islower()
        assert "!" not in normalized
        assert "stock" in normalized
        assert "market" in normalized
        assert "closed" in normalized


class TestContentQualityFilter:
    """Test the main quality filter"""
    
    @pytest.fixture
    def mock_storage(self):
        """Mock Redis storage"""
        storage = AsyncMock(spec=RedisStorage)
        storage.get.return_value = None
        storage.set.return_value = True
        return storage
    
    @pytest.fixture
    def quality_filter(self, mock_storage):
        """Create quality filter with mocked storage"""
        config = {
            'min_word_count': 100,
            'min_relevance_score': 0.3,
            'min_credibility_score': 0.4,
            'duplicate_threshold': 0.8
        }
        
        filter_obj = ContentQualityFilter(mock_storage, config)
        # Mock the FinBERT model to avoid loading in tests
        filter_obj.relevance_scorer._calculate_finbert_score = MagicMock(return_value=0.7)
        
        return filter_obj
    
    @pytest.mark.asyncio
    async def test_evaluate_high_quality_article(self, quality_filter):
        """Test evaluation of high-quality article"""
        article = ExtractedArticle(
            url="https://reuters.com/article",
            title="Apple Reports Record Quarterly Earnings",
            text=" ".join(["Strong financial performance with record revenue and profit margins."] * 30),  # 150+ words
            word_count=150,
            source="reuters.com",
            paywall_status=PaywallStatus.FREE,
            authors=["Business Reporter"],
            published_at=None,
            updated_at=None
        )
        
        # Mock duplication check to return no duplicate
        quality_filter.duplication_detector.check_duplicate = AsyncMock(return_value=None)
        
        decision = await quality_filter.evaluate_quality(article)
        
        assert decision.keep is True
        assert "Passed all quality checks" in decision.reasons
        assert decision.scores['financial_relevance'] > 0.5
        assert decision.scores['credibility'] >= 0.95  # Reuters high credibility
    
    @pytest.mark.asyncio
    async def test_evaluate_low_quality_article(self, quality_filter):
        """Test evaluation of low-quality article"""
        article = ExtractedArticle(
            url="https://unknown-blog.com/article",
            title="Short Title",
            text="This is very short content.",  # Too short
            word_count=5,
            source="unknown-blog.com",
            paywall_status=PaywallStatus.FREE,
            authors=[],
            published_at=None,
            updated_at=None
        )
        
        decision = await quality_filter.evaluate_quality(article)
        
        assert decision.keep is False
        assert any("Too short" in reason for reason in decision.reasons)
        assert any("Low financial relevance" in reason for reason in decision.reasons)
    
    @pytest.mark.asyncio
    async def test_evaluate_paywalled_article(self, quality_filter):
        """Test evaluation of paywalled article"""
        article = ExtractedArticle(
            url="https://wsj.com/premium",
            title="Premium Market Analysis",
            text="[PAYWALLED CONTENT]",
            word_count=0,
            source="wsj.com",
            paywall_status=PaywallStatus.PAYWALLED,
            authors=["Premium Author"],
            published_at=None,
            updated_at=None
        )
        
        decision = await quality_filter.evaluate_quality(article)
        
        assert decision.keep is False
        assert "Content is paywalled" in decision.reasons
    
    @pytest.mark.asyncio
    async def test_evaluate_duplicate_article(self, quality_filter):
        """Test evaluation of duplicate article"""
        article = ExtractedArticle(
            url="https://cnbc.com/duplicate",
            title="Market Update: Stocks Rise",
            text=" ".join(["Comprehensive market analysis with detailed insights."] * 40),  # 200+ words
            word_count=200,
            source="cnbc.com",
            paywall_status=PaywallStatus.FREE,
            authors=["Market Reporter"],
            published_at=None,
            updated_at=None
        )
        
        # Mock duplication check to return a duplicate
        quality_filter.duplication_detector.check_duplicate = AsyncMock(
            return_value="https://original-article.com"
        )
        
        decision = await quality_filter.evaluate_quality(article)
        
        assert decision.keep is False
        assert any("Duplicate of" in reason for reason in decision.reasons)
        assert decision.duplicate_of == "https://original-article.com"


if __name__ == '__main__':
    pytest.main([__file__])