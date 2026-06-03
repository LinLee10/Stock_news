"""
ContentQualityFilter - AI-powered content quality assessment.

This module provides content quality filtering using FinBERT for financial
relevance scoring, deduplication, and credibility assessment.
"""

import hashlib
import re
import time
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

import structlog
from datasketch import MinHashLSH, MinHash
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

from .models import ExtractedArticle, QualityDecision
from .storage import RedisStorage

logger = structlog.get_logger(__name__)


class FinancialRelevanceScorer:
    """Scores content for financial relevance using FinBERT"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._classifier = None
        
        # Financial keywords for additional scoring
        self.financial_keywords = {
            'companies': [
                'earnings', 'revenue', 'profit', 'loss', 'quarterly', 'annual',
                'ceo', 'cfo', 'dividend', 'shares', 'stock', 'equity',
                'merger', 'acquisition', 'ipo', 'buyback', 'split'
            ],
            'markets': [
                'nasdaq', 'nyse', 'dow', 'sp500', 's&p', 'market', 'trading',
                'bull', 'bear', 'volatility', 'futures', 'options',
                'index', 'sector', 'commodity', 'currency', 'forex'
            ],
            'economics': [
                'fed', 'federal reserve', 'interest rate', 'inflation',
                'recession', 'gdp', 'unemployment', 'economic', 'fiscal',
                'monetary', 'policy', 'treasury', 'bond', 'yield'
            ],
            'crypto': [
                'bitcoin', 'cryptocurrency', 'crypto', 'blockchain',
                'ethereum', 'defi', 'nft', 'mining', 'wallet', 'exchange'
            ]
        }
    
    def _load_model(self):
        """Lazy load the FinBERT model"""
        if self._classifier is None:
            try:
                logger.info("Loading FinBERT model", model=self.model_name, device=self.device)
                
                # Load tokenizer and model
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                # Create pipeline
                self._classifier = pipeline(
                    "sentiment-analysis",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                
                logger.info("FinBERT model loaded successfully")
                
            except Exception as e:
                logger.error("Failed to load FinBERT model", error=str(e))
                # Fallback to a lighter model
                self._classifier = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                logger.info("Using fallback sentiment model")
    
    def score_financial_relevance(self, text: str, title: str = "") -> float:
        """
        Score text for financial relevance.
        
        Returns:
            Score between 0 and 1, where 1 is highly relevant
        """
        if not text:
            return 0.0
        
        # Combined text for analysis
        combined_text = f"{title} {text}".strip()
        
        # Keyword-based scoring
        keyword_score = self._calculate_keyword_score(combined_text.lower())
        
        # FinBERT-based scoring (if available)
        finbert_score = self._calculate_finbert_score(combined_text[:512])  # Limit length
        
        # Weighted combination
        final_score = (keyword_score * 0.4) + (finbert_score * 0.6)
        
        return min(1.0, max(0.0, final_score))
    
    def _calculate_keyword_score(self, text_lower: str) -> float:
        """Calculate financial relevance based on keywords"""
        total_score = 0.0
        word_count = len(text_lower.split())
        
        if word_count == 0:
            return 0.0
        
        for category, keywords in self.financial_keywords.items():
            category_matches = sum(1 for keyword in keywords if keyword in text_lower)
            category_score = min(1.0, category_matches / len(keywords) * 3)  # Scale up
            
            # Weight different categories
            if category == 'companies':
                total_score += category_score * 0.4
            elif category == 'markets':
                total_score += category_score * 0.3
            elif category == 'economics':
                total_score += category_score * 0.2
            elif category == 'crypto':
                total_score += category_score * 0.1
        
        return min(1.0, total_score)
    
    def _calculate_finbert_score(self, text: str) -> float:
        """Calculate financial relevance using FinBERT"""
        if not text.strip():
            return 0.0
        
        try:
            self._load_model()
            
            # Get sentiment scores (FinBERT outputs positive/negative/neutral)
            results = self._classifier(text[:512])  # Limit token length
            
            if results and len(results) > 0:
                scores = results[0]  # First result
                
                # For financial relevance, we care about having strong sentiment
                # (either positive or negative) which indicates financial impact
                pos_score = next((s['score'] for s in scores if 'positive' in s['label'].lower()), 0.0)
                neg_score = next((s['score'] for s in scores if 'negative' in s['label'].lower()), 0.0)
                
                # Strong sentiment (either direction) indicates financial relevance
                sentiment_strength = max(pos_score, neg_score)
                
                # Additional boost if it seems like financial content
                financial_boost = 1.0
                if any(term in text.lower() for term in ['stock', 'market', 'earnings', 'revenue', 'profit']):
                    financial_boost = 1.2
                
                return min(1.0, sentiment_strength * financial_boost)
        
        except Exception as e:
            logger.warning("FinBERT scoring failed, using fallback", error=str(e))
            return self._calculate_keyword_score(text.lower())
        
        return 0.0


class DuplicationDetector:
    """Detects duplicate content using MinHash LSH"""
    
    def __init__(self, storage: RedisStorage, threshold: float = 0.8):
        self.storage = storage
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self._initialized = False
    
    async def initialize(self):
        """Initialize LSH index from Redis"""
        if self._initialized:
            return
        
        try:
            # Load existing hashes from Redis
            existing_hashes = await self._load_existing_hashes()
            for url, minhash_data in existing_hashes.items():
                minhash = self._deserialize_minhash(minhash_data)
                if minhash:
                    self.lsh.insert(url, minhash)
            
            self._initialized = True
            logger.info("Duplication detector initialized", existing_count=len(existing_hashes))
            
        except Exception as e:
            logger.error("Failed to initialize duplication detector", error=str(e))
    
    async def check_duplicate(self, article: ExtractedArticle) -> Optional[str]:
        """
        Check if article is a duplicate.
        
        Returns:
            URL of duplicate article if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        # Create MinHash for the article
        minhash = self._create_minhash(article.text, article.title)
        
        # Query LSH for similar articles
        similar_urls = self.lsh.query(minhash)
        
        if similar_urls:
            # Return the first duplicate found
            duplicate_url = similar_urls[0]
            logger.info("Duplicate content detected", 
                       original=duplicate_url, 
                       duplicate=article.url,
                       similarity_threshold=self.threshold)
            return duplicate_url
        
        # Store this article's hash
        await self._store_minhash(article.url, minhash)
        self.lsh.insert(article.url, minhash)
        
        return None
    
    def _create_minhash(self, text: str, title: str = "") -> MinHash:
        """Create MinHash from article text"""
        minhash = MinHash(num_perm=128)
        
        # Normalize text
        normalized_text = self._normalize_text(f"{title} {text}")
        
        # Create shingles (n-grams)
        shingles = self._create_shingles(normalized_text, n=3)
        
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        
        return minhash
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove common stop words that don't affect content similarity
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def _create_shingles(self, text: str, n: int = 3) -> Set[str]:
        """Create n-gram shingles from text"""
        words = text.split()
        shingles = set()
        
        for i in range(len(words) - n + 1):
            shingle = ' '.join(words[i:i + n])
            shingles.add(shingle)
        
        return shingles
    
    def _serialize_minhash(self, minhash: MinHash) -> str:
        """Serialize MinHash for storage"""
        return ','.join(str(x) for x in minhash.hashvalues)
    
    def _deserialize_minhash(self, data: str) -> Optional[MinHash]:
        """Deserialize MinHash from storage"""
        try:
            values = [int(x) for x in data.split(',')]
            minhash = MinHash(num_perm=len(values))
            minhash.hashvalues = values
            return minhash
        except Exception:
            return None
    
    async def _load_existing_hashes(self) -> Dict[str, str]:
        """Load existing MinHashes from Redis"""
        # This would load from Redis with a pattern like "dup:minhash:*"
        # For now, return empty dict as we haven't implemented Redis pattern matching
        return {}
    
    async def _store_minhash(self, url: str, minhash: MinHash):
        """Store MinHash in Redis"""
        key = f"dup:minhash:{hashlib.md5(url.encode()).hexdigest()}"
        data = self._serialize_minhash(minhash)
        await self.storage.set(key, data, ttl=86400 * 30)  # 30 days


class CredibilityScorer:
    """Scores source credibility"""
    
    def __init__(self):
        # Tier-based credibility scores
        self.credibility_scores = {
            # Tier 1: Major financial news outlets
            'reuters.com': 0.95,
            'bloomberg.com': 0.95,
            'wsj.com': 0.95,
            'ft.com': 0.95,
            'cnbc.com': 0.90,
            'marketwatch.com': 0.90,
            'investing.com': 0.85,
            'yahoo.com': 0.80,  # Yahoo Finance
            
            # Tier 2: Business publications
            'forbes.com': 0.85,
            'businessinsider.com': 0.80,
            'fortune.com': 0.85,
            'economist.com': 0.90,
            
            # Tier 3: General news with financial sections
            'nytimes.com': 0.85,
            'washingtonpost.com': 0.85,
            'cnn.com': 0.75,
            'bbc.com': 0.85,
            
            # Lower credibility sources
            'seekingalpha.com': 0.70,
            'fool.com': 0.65,  # Motley Fool
        }
    
    def get_credibility_score(self, source: str, domain: str) -> float:
        """Get credibility score for a source"""
        # Clean domain
        domain = domain.replace('www.', '').lower()
        
        # Direct lookup
        if domain in self.credibility_scores:
            return self.credibility_scores[domain]
        
        # Pattern matching for subdomains
        for scored_domain, score in self.credibility_scores.items():
            if domain.endswith(scored_domain):
                return score
        
        # Default credibility for unknown sources
        return 0.5


class ContentQualityFilter:
    """
    Main content quality filter that combines multiple scoring methods
    to determine if content should be kept.
    """
    
    def __init__(self, storage: RedisStorage, config: Dict[str, Any] = None):
        self.storage = storage
        self.config = config or {}
        
        # Initialize components
        self.relevance_scorer = FinancialRelevanceScorer(
            model_name=self.config.get('finbert_model', 'ProsusAI/finbert'),
            device=self.config.get('device', 'auto')
        )
        self.duplication_detector = DuplicationDetector(
            storage, 
            threshold=self.config.get('duplicate_threshold', 0.8)
        )
        self.credibility_scorer = CredibilityScorer()
        
        # Thresholds
        self.min_word_count = self.config.get('min_word_count', 200)
        self.min_relevance_score = self.config.get('min_relevance_score', 0.3)
        self.min_credibility_score = self.config.get('min_credibility_score', 0.4)
    
    async def evaluate_quality(self, article: ExtractedArticle) -> QualityDecision:
        """
        Evaluate article quality and return decision.
        
        Args:
            article: ExtractedArticle to evaluate
            
        Returns:
            QualityDecision with keep/reject decision and reasons
        """
        start_time = time.time()
        reasons = []
        scores = {}
        keep = True
        
        try:
            # Word count check
            if article.word_count < self.min_word_count:
                keep = False
                reasons.append(f"Too short: {article.word_count} words (min: {self.min_word_count})")
            
            scores['word_count'] = article.word_count
            
            # Financial relevance scoring
            relevance_score = self.relevance_scorer.score_financial_relevance(
                article.text, article.title
            )
            scores['financial_relevance'] = relevance_score
            
            if relevance_score < self.min_relevance_score:
                keep = False
                reasons.append(f"Low financial relevance: {relevance_score:.3f} (min: {self.min_relevance_score})")
            
            # Source credibility
            domain = article.source.replace('www.', '').lower()
            credibility_score = self.credibility_scorer.get_credibility_score(
                article.source, domain
            )
            scores['credibility'] = credibility_score
            
            if credibility_score < self.min_credibility_score:
                keep = False
                reasons.append(f"Low source credibility: {credibility_score:.3f} (min: {self.min_credibility_score})")
            
            # Duplicate detection
            duplicate_url = await self.duplication_detector.check_duplicate(article)
            if duplicate_url:
                keep = False
                reasons.append(f"Duplicate of: {duplicate_url}")
            
            # Paywall check
            if article.paywall_status.value in ['paywalled']:
                keep = False
                reasons.append("Content is paywalled")
            
            # Additional quality checks
            if not article.title or len(article.title) < 10:
                keep = False
                reasons.append("Invalid or too short title")
            
            # If everything passed, add positive reasons
            if keep and not reasons:
                reasons.append("Passed all quality checks")
            
            evaluation_time = int((time.time() - start_time) * 1000)
            
            return QualityDecision(
                keep=keep,
                reasons=reasons,
                scores=scores,
                duplicate_of=duplicate_url,
                evaluation_time_ms=evaluation_time
            )
            
        except Exception as e:
            logger.error("Quality evaluation failed", url=article.url, error=str(e))
            
            evaluation_time = int((time.time() - start_time) * 1000)
            
            return QualityDecision(
                keep=False,
                reasons=[f"Evaluation error: {str(e)}"],
                scores=scores,
                evaluation_time_ms=evaluation_time
            )