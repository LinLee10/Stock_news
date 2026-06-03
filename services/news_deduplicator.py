"""
NewsDeduplicator - Advanced deduplication system for financial news articles.

This module implements multiple deduplication strategies:
1. SHA-256 hashing for exact duplicates
2. SimHash for semantic similarity detection
3. Content fingerprinting for near-duplicates
4. Fast lookup using Redis-backed hash sets
"""

import hashlib
import re
import time
from typing import Dict, List, Set, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    import redis
    import structlog
    from simhash import Simhash, SimhashIndex
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    REDIS_AVAILABLE = True
    logger = structlog.get_logger(__name__)
except ImportError:
    REDIS_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz
except ImportError:
    class _FuzzFallback:
        @staticmethod
        def ratio(left: str, right: str) -> float:
            return SequenceMatcher(None, left, right).ratio() * 100

    fuzz = _FuzzFallback()

# BEGIN F11 - Vector search integration
try:
    from services.vector_store import NewsItem, vector_store
    from config.feature_flags import is_vector_search_enabled
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    logger.warning("Vector store not available")
# END F11


class DuplicateResult(NamedTuple):
    """Result of duplicate detection"""
    is_duplicate: bool
    duplicate_type: str  # 'exact', 'semantic', 'near_duplicate'
    similarity_score: float
    original_url: Optional[str] = None
    processing_time_ms: int = 0


@dataclass
class ContentFingerprint:
    """Content fingerprint for deduplication"""
    url: str
    title: str
    content_hash: str
    simhash: int
    key_phrases: List[str]
    word_count: int
    publish_date: Optional[datetime] = None
    symbol: Optional[str] = None  # F11: Add symbol for vector search
    

class NewsDeduplicator:
    """
    Advanced news deduplication system with multiple detection strategies.
    
    Features:
    - SHA-256 exact duplicate detection
    - SimHash semantic similarity (85% threshold)
    - Content fingerprinting for near-duplicates
    - Redis-backed fast lookups
    - Real-time and batch processing modes
    """
    
    def __init__(self, redis_client: Optional[object] = None, similarity_threshold: float = 0.85):
        if REDIS_AVAILABLE:
            self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        else:
            self.redis_client = None
        self.similarity_threshold = similarity_threshold
        
        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not found, downloading...")
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
        
        # SimHash index for efficient similarity search
        self.simhash_index = SimhashIndex([], k=3)  # k=3 allows for ~85% similarity
        
        # Redis keys
        self.EXACT_HASH_KEY = "news:exact_hashes"
        self.SIMHASH_KEY = "news:simhashes"
        self.URL_FINGERPRINT_KEY = "news:fingerprints"
        self.STATS_KEY = "news:dedup_stats"
        
        # Initialize statistics
        self._init_stats()
        
    def _init_stats(self):
        """Initialize deduplication statistics"""
        if not self.redis_client.exists(self.STATS_KEY):
            stats = {
                'total_processed': 0,
                'exact_duplicates': 0,
                'semantic_duplicates': 0,
                'near_duplicates': 0,
                'unique_articles': 0
            }
            self.redis_client.hset(self.STATS_KEY, mapping=stats)
    
    def create_content_fingerprint(self, url: str, title: str, content: str, 
                                 publish_date: Optional[datetime] = None,
                                 symbol: Optional[str] = None) -> ContentFingerprint:
        """
        Create a comprehensive content fingerprint for deduplication.
        
        Args:
            url: Article URL
            title: Article title
            content: Full article content
            publish_date: Publication date
            
        Returns:
            ContentFingerprint object
        """
        start_time = time.time()
        
        try:
            # Normalize content for consistent hashing
            normalized_content = self._normalize_content(content)
            normalized_title = self._normalize_content(title)
            
            # Create exact content hash (SHA-256)
            content_hash = hashlib.sha256(
                f"{normalized_title}|{normalized_content}".encode('utf-8')
            ).hexdigest()
            
            # Create SimHash for semantic similarity
            combined_text = f"{title} {content}"
            simhash_value = Simhash(self._extract_features(combined_text)).value
            
            # Extract key phrases for additional matching
            key_phrases = self._extract_key_phrases(combined_text)
            
            # Calculate word count
            word_count = len(content.split())
            
            processing_time = int((time.time() - start_time) * 1000)
            
            fingerprint = ContentFingerprint(
                url=url,
                title=title,
                content_hash=content_hash,
                simhash=simhash_value,
                key_phrases=key_phrases,
                word_count=word_count,
                publish_date=publish_date,
                symbol=symbol  # F11: Include symbol for vector search
            )
            
            logger.debug("Content fingerprint created", 
                        url=url, 
                        processing_time_ms=processing_time,
                        word_count=word_count)
            
            return fingerprint
            
        except Exception as e:
            logger.error("Failed to create content fingerprint", 
                        url=url, error=str(e))
            raise
    
    def check_duplicate(self, fingerprint: ContentFingerprint) -> DuplicateResult:
        """
        Check if content is a duplicate using multiple strategies.
        
        Args:
            fingerprint: Content fingerprint to check
            
        Returns:
            DuplicateResult indicating if content is duplicate
        """
        start_time = time.time()
        
        try:
            # Strategy 1: Exact hash duplicate detection
            if self.redis_client.sismember(self.EXACT_HASH_KEY, fingerprint.content_hash):
                # Get original URL for exact duplicate
                original_url = self._get_original_url_by_hash(fingerprint.content_hash)
                
                self._increment_stat('exact_duplicates')
                processing_time = int((time.time() - start_time) * 1000)
                
                return DuplicateResult(
                    is_duplicate=True,
                    duplicate_type='exact',
                    similarity_score=1.0,
                    original_url=original_url,
                    processing_time_ms=processing_time
                )
            
            # Strategy 2: Semantic similarity using SimHash
            similar_hashes = self.simhash_index.get_near_dups(Simhash(fingerprint.simhash))
            
            if similar_hashes:
                # Calculate similarity score with best match
                best_similarity = 0.0
                best_original_url = None
                
                for similar_hash in similar_hashes[:3]:  # Check top 3 matches
                    similarity = 1.0 - (bin(fingerprint.simhash ^ similar_hash).count('1') / 64.0)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_original_url = self._get_original_url_by_simhash(similar_hash)
                
                if best_similarity >= self.similarity_threshold:
                    self._increment_stat('semantic_duplicates')
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    return DuplicateResult(
                        is_duplicate=True,
                        duplicate_type='semantic',
                        similarity_score=best_similarity,
                        original_url=best_original_url,
                        processing_time_ms=processing_time
                    )
            
            # Strategy 3: Key phrase matching for near-duplicates
            near_duplicate_result = self._check_near_duplicate(fingerprint)
            if near_duplicate_result.is_duplicate:
                self._increment_stat('near_duplicates')
                return near_duplicate_result
            
            # Not a duplicate - it's unique content
            self._increment_stat('unique_articles')
            processing_time = int((time.time() - start_time) * 1000)
            
            return DuplicateResult(
                is_duplicate=False,
                duplicate_type='unique',
                similarity_score=0.0,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error("Error checking duplicate", 
                        url=fingerprint.url, error=str(e))
            processing_time = int((time.time() - start_time) * 1000)
            
            return DuplicateResult(
                is_duplicate=False,
                duplicate_type='error',
                similarity_score=0.0,
                processing_time_ms=processing_time
            )
    
    def store_fingerprint(self, fingerprint: ContentFingerprint) -> bool:
        """
        Store content fingerprint for future duplicate detection.
        
        Args:
            fingerprint: Content fingerprint to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Store exact hash
            self.redis_client.sadd(self.EXACT_HASH_KEY, fingerprint.content_hash)
            
            # Store SimHash in index
            self.simhash_index.add(fingerprint.url, Simhash(fingerprint.simhash))
            
            # Store complete fingerprint data
            fingerprint_data = {
                'url': fingerprint.url,
                'title': fingerprint.title,
                'content_hash': fingerprint.content_hash,
                'simhash': str(fingerprint.simhash),
                'key_phrases': ','.join(fingerprint.key_phrases),
                'word_count': fingerprint.word_count,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            if fingerprint.publish_date:
                fingerprint_data['publish_date'] = fingerprint.publish_date.isoformat()
            
            self.redis_client.hset(
                f"{self.URL_FINGERPRINT_KEY}:{fingerprint.content_hash}",
                mapping=fingerprint_data
            )
            
            # Set expiration (30 days)
            self.redis_client.expire(
                f"{self.URL_FINGERPRINT_KEY}:{fingerprint.content_hash}",
                30 * 24 * 3600
            )
            
            self._increment_stat('total_processed')
            
            # BEGIN F11 - Store embeddings if vector search enabled
            if VECTOR_STORE_AVAILABLE and is_vector_search_enabled():
                try:
                    # Extract symbol from fingerprint data if available
                    symbol = getattr(fingerprint, 'symbol', 'UNKNOWN')
                    
                    news_item = NewsItem(
                        headline=fingerprint.title,
                        url=fingerprint.url,
                        published_at=fingerprint.publish_date or datetime.now(timezone.utc),
                        symbol=symbol,
                        content_hash=fingerprint.content_hash[:16]  # Truncate for vector store
                    )
                    
                    # Store asynchronously (fire and forget)
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If we're already in an async context, schedule for later
                            asyncio.create_task(vector_store.put_news([news_item]))
                        else:
                            # If no event loop, run in new loop
                            asyncio.run(vector_store.put_news([news_item]))
                    except RuntimeError:
                        # Fallback: just log for now
                        logger.debug("Could not store vector embedding immediately", url=fingerprint.url)
                        
                except Exception as e:
                    logger.warning("Failed to store vector embedding", url=fingerprint.url, error=str(e))
            # END F11
            
            logger.debug("Fingerprint stored", url=fingerprint.url)
            return True
            
        except Exception as e:
            logger.error("Failed to store fingerprint", 
                        url=fingerprint.url, error=str(e))
            return False
    
    def _normalize_content(self, text: str) -> str:
        """
        Normalize content for consistent comparison.
        
        Args:
            text: Raw text content
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common boilerplate patterns
        patterns_to_remove = [
            r'subscribe.*newsletter',
            r'follow\s+us\s+on',
            r'share\s+this\s+article',
            r'©\s*\d{4}.*',
            r'all\s+rights\s+reserved',
            r'terms\s+of\s+service',
            r'privacy\s+policy'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def _extract_features(self, text: str) -> List[str]:
        """
        Extract features for SimHash calculation.
        
        Args:
            text: Input text
            
        Returns:
            List of features (words/phrases)
        """
        try:
            # Tokenize and remove stop words
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            
            # Create n-grams for better semantic matching
            features = []
            
            # Unigrams
            features.extend(words)
            
            # Bigrams
            for i in range(len(words) - 1):
                features.append(f"{words[i]}_{words[i+1]}")
            
            # Trigrams for financial terms
            financial_trigrams = []
            for i in range(len(words) - 2):
                trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
                # Boost financial terms
                if any(term in trigram for term in ['earnings', 'revenue', 'profit', 'stock', 'market']):
                    financial_trigrams.extend([trigram] * 3)  # Triple weight
                else:
                    financial_trigrams.append(trigram)
            
            features.extend(financial_trigrams)
            
            return features
            
        except Exception as e:
            logger.warning("Feature extraction failed", error=str(e))
            return text.split()  # Fallback to simple word splitting
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from content for additional matching.
        
        Args:
            text: Input text
            
        Returns:
            List of key phrases
        """
        try:
            # Financial keywords that are important for matching
            financial_terms = [
                'earnings per share', 'revenue growth', 'market cap', 'price target',
                'quarterly results', 'profit margin', 'dividend yield', 'book value',
                'return on equity', 'debt to equity', 'free cash flow', 'operating income'
            ]
            
            key_phrases = []
            text_lower = text.lower()
            
            # Extract financial terms
            for term in financial_terms:
                if term in text_lower:
                    key_phrases.append(term)
            
            # Extract quoted text (often important statements)
            quotes = re.findall(r'"([^"]*)"', text)
            key_phrases.extend([quote.lower() for quote in quotes if len(quote) > 20])
            
            # Extract numbers with context (prices, percentages, etc.)
            number_contexts = re.findall(r'(\w+\s+)?\$?(\d+(?:\.\d+)?)\s*(?:%|billion|million|thousand)?(\s+\w+)?', text)
            for match in number_contexts:
                context = f"{match[0]}{match[1]}{match[2]}".strip()
                if context:
                    key_phrases.append(context.lower())
            
            return key_phrases[:20]  # Limit to top 20 phrases
            
        except Exception as e:
            logger.warning("Key phrase extraction failed", error=str(e))
            return []
    
    def _check_near_duplicate(self, fingerprint: ContentFingerprint) -> DuplicateResult:
        """
        Check for near-duplicates using key phrase matching and fuzzy string similarity.
        
        Args:
            fingerprint: Content fingerprint to check
            
        Returns:
            DuplicateResult for near-duplicate check
        """
        start_time = time.time()
        
        try:
            # Get recent fingerprints (last 7 days worth)
            pattern = f"{self.URL_FINGERPRINT_KEY}:*"
            keys = self.redis_client.keys(pattern)
            
            best_similarity = 0.0
            best_original_url = None
            
            # Check against recent articles
            for key in keys[:100]:  # Limit to 100 recent articles for performance
                stored_data = self.redis_client.hgetall(key)
                if not stored_data:
                    continue
                
                stored_title = stored_data.get(b'title', b'').decode('utf-8')
                stored_phrases = stored_data.get(b'key_phrases', b'').decode('utf-8').split(',')
                stored_url = stored_data.get(b'url', b'').decode('utf-8')
                
                # Skip self-comparison
                if stored_url == fingerprint.url:
                    continue
                
                # Title similarity
                title_similarity = fuzz.ratio(fingerprint.title.lower(), stored_title.lower()) / 100.0
                
                # Key phrase overlap
                phrase_overlap = len(set(fingerprint.key_phrases) & set(stored_phrases))
                phrase_similarity = phrase_overlap / max(1, len(fingerprint.key_phrases))
                
                # Combined similarity score
                combined_similarity = (title_similarity * 0.6) + (phrase_similarity * 0.4)
                
                if combined_similarity > best_similarity:
                    best_similarity = combined_similarity
                    best_original_url = stored_url
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Threshold for near-duplicates (lower than semantic similarity)
            if best_similarity >= 0.75:  # 75% similarity threshold
                return DuplicateResult(
                    is_duplicate=True,
                    duplicate_type='near_duplicate',
                    similarity_score=best_similarity,
                    original_url=best_original_url,
                    processing_time_ms=processing_time
                )
            
            return DuplicateResult(
                is_duplicate=False,
                duplicate_type='unique',
                similarity_score=best_similarity,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error("Near-duplicate check failed", error=str(e))
            processing_time = int((time.time() - start_time) * 1000)
            
            return DuplicateResult(
                is_duplicate=False,
                duplicate_type='error',
                similarity_score=0.0,
                processing_time_ms=processing_time
            )
    
    def _get_original_url_by_hash(self, content_hash: str) -> Optional[str]:
        """Get original URL by content hash"""
        try:
            data = self.redis_client.hgetall(f"{self.URL_FINGERPRINT_KEY}:{content_hash}")
            if data:
                return data.get(b'url', b'').decode('utf-8')
        except Exception as e:
            logger.warning("Failed to get original URL by hash", error=str(e))
        return None
    
    def _get_original_url_by_simhash(self, simhash_value: int) -> Optional[str]:
        """Get original URL by simhash value"""
        try:
            # This would require maintaining a reverse index in practice
            # For now, return None as it's complex to implement efficiently
            return None
        except Exception as e:
            logger.warning("Failed to get original URL by simhash", error=str(e))
        return None
    
    def _increment_stat(self, stat_name: str) -> None:
        """Increment a statistics counter"""
        try:
            self.redis_client.hincrby(self.STATS_KEY, stat_name, 1)
        except Exception as e:
            logger.warning("Failed to increment stat", stat=stat_name, error=str(e))
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics"""
        try:
            stats_data = self.redis_client.hgetall(self.STATS_KEY)
            return {
                key.decode('utf-8'): int(value.decode('utf-8'))
                for key, value in stats_data.items()
            }
        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return {}
    
    def cleanup_old_entries(self, days_to_keep: int = 30) -> int:
        """
        Clean up old fingerprint entries to prevent memory bloat.
        
        Args:
            days_to_keep: Number of days to keep entries
            
        Returns:
            Number of entries cleaned up
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            pattern = f"{self.URL_FINGERPRINT_KEY}:*"
            keys = self.redis_client.keys(pattern)
            
            cleaned_count = 0
            
            for key in keys:
                data = self.redis_client.hgetall(key)
                if not data:
                    continue
                
                timestamp_str = data.get(b'timestamp', b'').decode('utf-8')
                if timestamp_str:
                    try:
                        entry_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if entry_date < cutoff_date:
                            # Remove from all stores
                            content_hash = data.get(b'content_hash', b'').decode('utf-8')
                            if content_hash:
                                self.redis_client.srem(self.EXACT_HASH_KEY, content_hash)
                            
                            self.redis_client.delete(key)
                            cleaned_count += 1
                            
                    except (ValueError, TypeError):
                        # Invalid timestamp, remove the entry
                        self.redis_client.delete(key)
                        cleaned_count += 1
            
            logger.info("Cleanup completed", cleaned_entries=cleaned_count)
            return cleaned_count
            
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
            return 0


# Utility functions for easy integration
def detect_duplicate_article(url: str, title: str, content: str, 
                           redis_client: Optional[object] = None) -> DuplicateResult:
    """
    Convenience function to detect duplicate articles.
    
    Args:
        url: Article URL
        title: Article title
        content: Article content
        redis_client: Optional Redis client
        
    Returns:
        DuplicateResult indicating if article is duplicate
    """
    deduplicator = NewsDeduplicator(redis_client)
    fingerprint = deduplicator.create_content_fingerprint(url, title, content)
    result = deduplicator.check_duplicate(fingerprint)
    
    if not result.is_duplicate:
        deduplicator.store_fingerprint(fingerprint)
    
    return result


def batch_deduplicate_articles(articles: List[Dict], 
                             redis_client: Optional[object] = None) -> List[Tuple[Dict, DuplicateResult]]:
    """
    Batch deduplication for multiple articles.
    
    Args:
        articles: List of article dictionaries with 'url', 'title', 'content' keys
        redis_client: Optional Redis client
        
    Returns:
        List of (article, DuplicateResult) tuples
    """
    deduplicator = NewsDeduplicator(redis_client)
    results = []
    
    for article in articles:
        try:
            fingerprint = deduplicator.create_content_fingerprint(
                article['url'], 
                article['title'], 
                article['content']
            )
            
            duplicate_result = deduplicator.check_duplicate(fingerprint)
            
            if not duplicate_result.is_duplicate:
                deduplicator.store_fingerprint(fingerprint)
            
            results.append((article, duplicate_result))
            
        except Exception as e:
            logger.error("Batch deduplication failed for article", 
                        url=article.get('url', 'unknown'), error=str(e))
            results.append((article, DuplicateResult(
                is_duplicate=False,
                duplicate_type='error',
                similarity_score=0.0
            )))
    
    return results


if __name__ == "__main__":
    # Test the deduplicator
    import redis
    
    # Test articles
    articles = [
        {
            'url': 'https://example.com/article1',
            'title': 'Apple Reports Strong Quarterly Earnings',
            'content': 'Apple Inc. reported strong quarterly earnings today, beating analyst expectations with revenue of $89.5 billion...'
        },
        {
            'url': 'https://example.com/article2', 
            'title': 'Apple Q4 Earnings Beat Expectations',
            'content': 'Apple exceeded analyst forecasts in Q4 with revenue reaching $89.5 billion, driven by iPhone sales...'
        },
        {
            'url': 'https://example.com/article3',
            'title': 'Tesla Stock Jumps on Production News',
            'content': 'Tesla shares rose 5% after the company announced record production numbers for the quarter...'
        }
    ]
    
    # Test batch deduplication
    if REDIS_AVAILABLE:
        redis_client = redis.Redis(host='localhost', port=6379, db=1)  # Use test DB
    else:
        redis_client = None
    results = batch_deduplicate_articles(articles, redis_client)
    
    for article, result in results:
        print(f"\nArticle: {article['title']}")
        print(f"Duplicate: {result.is_duplicate} ({result.duplicate_type})")
        print(f"Similarity: {result.similarity_score:.3f}")
        if result.original_url:
            print(f"Original: {result.original_url}")
    
    # Print stats
    deduplicator = NewsDeduplicator(redis_client)
    stats = deduplicator.get_stats()
    print(f"\nDeduplication Stats: {stats}")


# BEGIN F03 - Simple deduplication without Redis dependency
def canonicalize_url(url: str) -> str:
    """
    Canonicalize URL for deduplication by removing tracking parameters
    and normalizing format
    """
    if not url:
        return ""
    
    try:
        parsed = urlparse(url.strip())
        
        # Remove common tracking parameters
        tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term',
            'gclid', 'fbclid', 'ref', '_source', 'source', 'campaign',
            'medium', 'content', 'term', 'mc_cid', 'mc_eid'
        }

        filtered_params = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in tracking_params
        ]
        filtered_params.sort(key=lambda item: (item[0].lower(), item[1]))
        new_query = urlencode(filtered_params, doseq=True)

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if scheme == "http" and netloc.endswith(":80"):
            netloc = netloc[:-3]
        elif scheme == "https" and netloc.endswith(":443"):
            netloc = netloc[:-4]
        
        return urlunparse((scheme, netloc, parsed.path, "", new_query, ""))
        
    except Exception:
        # Fallback to simple normalization
        return url.strip().split('?')[0].lower() if '?' in url else url.strip().lower()


def dedupe_headlines_simple(headlines: List[Tuple[str, str, datetime]], threshold: float = 0.8) -> List[Tuple[str, str, datetime]]:
    """
    F03: Simple deduplication using URL canonicalization and RapidFuzz similarity
    
    Args:
        headlines: List of (title, url, datetime) tuples
        threshold: Similarity threshold for title matching (0.0-1.0)
        
    Returns:
        Deduplicated list of headlines
    """
    if not headlines:
        return []
        
    seen_urls = set()
    seen_titles = []
    deduplicated = []
    
    for title, url, date in headlines:
        # Step 1: URL-based deduplication
        canonical_url = canonicalize_url(url)
        
        if canonical_url in seen_urls:
            continue  # Skip exact URL duplicate
            
        # Step 2: Title similarity-based deduplication
        title_lower = title.lower().strip()
        is_similar = False
        
        for existing_title in seen_titles:
            similarity = fuzz.ratio(title_lower, existing_title.lower()) / 100.0
            
            if similarity >= threshold:
                is_similar = True
                break
        
        if not is_similar:
            # Not a duplicate - add to results
            seen_urls.add(canonical_url)
            seen_titles.append(title)
            deduplicated.append((title, url, date))
    
    # Calculate dedupe ratio for logging
    if len(headlines) > 0:
        dedupe_ratio = ((len(headlines) - len(deduplicated)) / len(headlines)) * 100
        logger.info(f"F03: Deduplication complete: {len(headlines)} → {len(deduplicated)} headlines ({dedupe_ratio:.1f}% removed)")
    
    return deduplicated
# END F03
