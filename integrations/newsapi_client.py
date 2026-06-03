#!/usr/bin/env python3
"""
F03 - NewsAPI client for multisource news ingestion

Provides NewsAPI.org integration with quota management and fallback handling.
Maps NewsAPI responses to the existing news scraper schema for seamless integration.
"""

import os
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from config.config import NEWSAPI_KEY
from config.feature_flags import is_newsapi_ingestion_enabled

logger = logging.getLogger(__name__)

@dataclass
class NewsAPIConfig:
    """NewsAPI configuration and rate limits"""
    base_url: str = "https://newsapi.org/v2"
    free_tier_daily_limit: int = 100
    request_timeout: int = 30
    rate_limit_window: int = 3600  # 1 hour in seconds
    max_requests_per_hour: int = 1000  # Conservative estimate

@dataclass 
class NewsAPIResult:
    """Result container for NewsAPI fetch operations"""
    success: bool
    articles: List[Tuple[str, str, datetime]]  # (title, url, published_at)
    total_results: int
    quota_remaining: Optional[int]
    error_message: Optional[str] = None
    fallback_triggered: bool = False

class NewsAPIQuotaManager:
    """Simple quota tracking for NewsAPI"""
    
    def __init__(self):
        self.daily_calls = 0
        self.last_reset_date = datetime.now().date()
        self.config = NewsAPIConfig()
        
    def can_make_request(self) -> bool:
        """Check if we can make a request within quota limits"""
        self._check_daily_reset()
        return self.daily_calls < self.config.free_tier_daily_limit
        
    def record_request(self, quota_remaining: Optional[int] = None):
        """Record that a request was made"""
        self.daily_calls += 1
        if quota_remaining is not None:
            # Update our count based on actual API response
            used_today = self.config.free_tier_daily_limit - quota_remaining
            self.daily_calls = max(self.daily_calls, used_today)
            
    def get_status(self) -> Dict[str, Any]:
        """Get current quota status"""
        self._check_daily_reset()
        return {
            'calls_made_today': self.daily_calls,
            'calls_remaining': max(0, self.config.free_tier_daily_limit - self.daily_calls),
            'daily_limit': self.config.free_tier_daily_limit,
            'reset_date': self.last_reset_date.isoformat()
        }
        
    def _check_daily_reset(self):
        """Reset daily counter if needed"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            logger.info(f"NewsAPI quota reset: {self.last_reset_date} → {today}")
            self.daily_calls = 0
            self.last_reset_date = today

class NewsAPIClient:
    """
    NewsAPI.org client with quota management and schema mapping
    
    Features:
    - Free tier quota management (100 requests/day)
    - Rate limiting and error handling
    - Maps to existing news scraper schema: (title, url, datetime)
    - Graceful fallback when quota exceeded
    """
    
    def __init__(self):
        self.config = NewsAPIConfig()
        self.quota_manager = NewsAPIQuotaManager()
        
        if not NEWSAPI_KEY:
            logger.warning("NewsAPI key not configured, NewsAPI ingestion will be disabled")
            
        logger.info("F03: NewsAPI client initialized")
        
    def fetch(self, symbols: List[str], days: int = 7) -> NewsAPIResult:
        """
        Fetch news for symbols from NewsAPI
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'NVDA'])
            days: Number of days to look back
            
        Returns:
            NewsAPIResult with articles in (title, url, datetime) format
        """
        if not is_newsapi_ingestion_enabled():
            logger.debug("F03: NewsAPI ingestion disabled via feature flag")
            return NewsAPIResult(
                success=False,
                articles=[],
                total_results=0,
                quota_remaining=None,
                error_message="NewsAPI ingestion disabled",
                fallback_triggered=True
            )
            
        if not NEWSAPI_KEY:
            logger.warning("F03: NewsAPI key not configured")
            return NewsAPIResult(
                success=False,
                articles=[],
                total_results=0,
                quota_remaining=None,
                error_message="NewsAPI key not configured",
                fallback_triggered=True
            )
            
        # Check quota before making request
        if not self.quota_manager.can_make_request():
            quota_status = self.quota_manager.get_status()
            logger.warning(f"F03: NewsAPI quota exhausted - {quota_status['calls_made_today']}/{quota_status['daily_limit']} calls used")
            return NewsAPIResult(
                success=False,
                articles=[],
                total_results=0,
                quota_remaining=quota_status['calls_remaining'],
                error_message="NewsAPI quota exhausted",
                fallback_triggered=True
            )
            
        # Prepare search query
        query = self._build_query(symbols)
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        try:
            logger.info(f"F03: Fetching NewsAPI articles for {len(symbols)} symbols, {days} days back")
            
            # Make API request
            response = self._make_request(query, from_date)
            
            if response.status_code == 429:
                # Rate limited
                logger.warning("F03: NewsAPI rate limit hit")
                return NewsAPIResult(
                    success=False,
                    articles=[],
                    total_results=0,
                    quota_remaining=0,
                    error_message="Rate limit exceeded",
                    fallback_triggered=True
                )
                
            response.raise_for_status()
            data = response.json()
            
            # Track quota usage
            quota_remaining = self._extract_quota_remaining(response)
            self.quota_manager.record_request(quota_remaining)
            
            # Parse articles to expected format
            articles = self._parse_articles(data.get('articles', []))
            total_results = data.get('totalResults', len(articles))
            
            logger.info(f"F03: Retrieved {len(articles)} articles from NewsAPI")
            
            return NewsAPIResult(
                success=True,
                articles=articles,
                total_results=total_results,
                quota_remaining=quota_remaining,
                fallback_triggered=False
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"F03: NewsAPI request failed: {e}")
            return NewsAPIResult(
                success=False,
                articles=[],
                total_results=0,
                quota_remaining=None,
                error_message=str(e),
                fallback_triggered=True
            )
        except Exception as e:
            logger.error(f"F03: NewsAPI parsing failed: {e}")
            return NewsAPIResult(
                success=False,
                articles=[],
                total_results=0,
                quota_remaining=None,
                error_message=str(e),
                fallback_triggered=True
            )
    
    def _build_query(self, symbols: List[str]) -> str:
        """Build NewsAPI search query from stock symbols"""
        # Use symbol names for better matching
        symbol_terms = []
        for symbol in symbols:
            # Add both symbol and common company name variations
            symbol_terms.append(symbol)
            
            # Add common company name mappings
            company_names = {
                'AAPL': 'Apple Inc',
                'NVDA': 'NVIDIA', 
                'TSLA': 'Tesla',
                'GOOGL': 'Alphabet Google',
                'MSFT': 'Microsoft',
                'AMZN': 'Amazon',
                'META': 'Meta Facebook',
                'PFE': 'Pfizer',
                'RTX': 'Raytheon'
            }
            
            if symbol in company_names:
                symbol_terms.append(company_names[symbol])
        
        # Create OR query for NewsAPI
        query = ' OR '.join(f'"{term}"' for term in symbol_terms[:10])  # Limit to avoid too long query
        return query
    
    def _make_request(self, query: str, from_date: str) -> requests.Response:
        """Make HTTP request to NewsAPI"""
        url = f"{self.config.base_url}/everything"
        
        params = {
            'q': query,
            'from': from_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,  # Maximum allowed
            'apiKey': NEWSAPI_KEY
        }
        
        return requests.get(
            url,
            params=params,
            timeout=self.config.request_timeout
        )
    
    def _extract_quota_remaining(self, response: requests.Response) -> Optional[int]:
        """Extract quota remaining from response headers"""
        # NewsAPI includes X-API-Key-Requests-Remaining header
        remaining_header = response.headers.get('X-API-Key-Requests-Remaining')
        if remaining_header:
            try:
                return int(remaining_header)
            except ValueError:
                pass
        return None
    
    def _parse_articles(self, articles: List[Dict[str, Any]]) -> List[Tuple[str, str, datetime]]:
        """Parse NewsAPI articles to expected format (title, url, datetime)"""
        parsed = []
        
        for article in articles:
            try:
                title = article.get('title', '').strip()
                url = article.get('url', '').strip()
                published_at = article.get('publishedAt', '')
                
                if not title or not url:
                    continue
                    
                # Parse ISO datetime
                if published_at:
                    # NewsAPI returns ISO format: "2023-10-25T14:30:00Z"
                    dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                else:
                    dt = datetime.now()
                
                parsed.append((title, url, dt))
                
            except Exception as e:
                logger.warning(f"F03: Failed to parse article: {e}")
                continue
                
        return parsed
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get current NewsAPI quota status"""
        return self.quota_manager.get_status()

# Global instance for easy import
newsapi_client = NewsAPIClient()