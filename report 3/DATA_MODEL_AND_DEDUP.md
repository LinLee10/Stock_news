# Data Model & Deduplication Strategy

## Canonical Data Schemas

### Price Data Schema
```python
@dataclass
class PriceDataPoint:
    """Canonical OHLCV data point"""
    provider: str                    # "yfinance", "twelve_data", "alpha_vantage"
    symbol: str                      # "AAPL", "NVDA" 
    timestamp: datetime              # UTC timezone required
    interval: str                    # "1d", "1h", "5m"
    open: Decimal
    high: Decimal  
    low: Decimal
    close: Decimal
    volume: int
    adjusted_close: Optional[Decimal] = None
    split_factor: Optional[Decimal] = None
    dividend: Optional[Decimal] = None
    
    @property
    def canonical_id(self) -> str:
        """Unique identifier for deduplication"""
        return f"{self.provider}|{self.symbol}|{self.interval}|{self.timestamp.isoformat()}"
```

### News Article Schema
```python  
@dataclass
class NewsArticle:
    """Canonical news article"""
    provider: str                    # "newsapi", "finnhub", "gnews"
    title: str
    url: str                        # Original source URL
    canonical_url: str              # Normalized for deduplication  
    content: Optional[str]          # Full article text if available
    summary: Optional[str]          # Article summary/description
    published_at: datetime          # UTC timezone required
    symbols_mentioned: List[str]    # ["AAPL", "NVDA"] - extracted tickers
    sentiment_score: Optional[float] = None  # -1.0 to 1.0
    content_hash: str = ""          # SHA-256 for deduplication
    
    def __post_init__(self):
        """Generate content hash after initialization"""
        content_for_hash = f"{self.title}|{self.canonical_url}|{self.summary or ''}"
        self.content_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()
    
    @property  
    def canonical_id(self) -> str:
        """Unique identifier for deduplication"""
        return f"{self.content_hash}|{self.published_at.date()}"
```

### Economic Indicator Schema
```python
@dataclass  
class EconomicIndicator:
    """Canonical economic/macro data point"""
    provider: str                    # "fred", "bea", "bls"
    series_id: str                  # "UNRATE", "GDP", "CPIAUCSL"
    name: str                       # "Unemployment Rate"
    timestamp: datetime             # Data point date
    value: Decimal
    unit: str                       # "Percent", "Billions", "Index"
    frequency: str                  # "Monthly", "Quarterly", "Annual"
    seasonal_adjustment: Optional[str] = None
    
    @property
    def canonical_id(self) -> str:
        return f"{self.provider}|{self.series_id}|{self.timestamp.date()}"
```

## Deduplication Strategies

### Price Data Deduplication
```python
class PriceDeduplicator:
    """Handle price data conflicts across providers"""
    
    PROVIDER_PRECEDENCE = {
        "yfinance": 1,        # Highest quality for daily data
        "alpaca": 2,          # Best for US intraday  
        "twelve_data": 3,     # Good for non-US and indicators
        "alpha_vantage": 4,   # Fallback only
        "polygon": 5          # Premium alternative
    }
    
    def merge_price_data(self, data_points: List[PriceDataPoint]) -> PriceDataPoint:
        """Merge duplicate price points using provider precedence"""
        if len(data_points) == 1:
            return data_points[0]
            
        # Group by canonical timestamp/symbol/interval
        grouped = {}
        for point in data_points:
            key = f"{point.symbol}|{point.interval}|{point.timestamp}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(point)
        
        merged_points = []
        for key, points in grouped.items():
            if len(points) == 1:
                merged_points.append(points[0])
            else:
                # Use provider precedence
                best_point = min(points, key=lambda p: self.PROVIDER_PRECEDENCE.get(p.provider, 99))
                merged_points.append(best_point)
                
                # Log conflict resolution
                providers = [p.provider for p in points]
                logger.info(f"Price conflict resolved: {key}, providers={providers}, chosen={best_point.provider}")
        
        return merged_points[0] if len(merged_points) == 1 else merged_points
```

### News Article Deduplication  
```python
class NewsDeduplicator:
    """Handle news article duplicates across providers"""
    
    def __init__(self):
        self.url_normalizer = URLNormalizer()
        self.content_similarity = ContentSimilarityMatcher()
    
    def deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate news articles"""
        unique_articles = {}
        duplicates_found = 0
        
        for article in articles:
            # Primary dedup: content hash
            if article.content_hash in unique_articles:
                existing = unique_articles[article.content_hash]
                merged = self._merge_articles(existing, article)
                unique_articles[article.content_hash] = merged
                duplicates_found += 1
                continue
            
            # Secondary dedup: URL canonicalization  
            canonical_url = self.url_normalizer.normalize(article.url)
            url_hash = hashlib.sha256(canonical_url.encode()).hexdigest()
            
            url_match = None
            for existing_article in unique_articles.values():
                if existing_article.content_hash == url_hash:
                    url_match = existing_article
                    break
            
            if url_match:
                merged = self._merge_articles(url_match, article)  
                unique_articles[url_match.content_hash] = merged
                duplicates_found += 1
                continue
            
            # Tertiary dedup: content similarity (expensive)
            similar_article = self.content_similarity.find_similar(
                article, list(unique_articles.values()), threshold=0.85
            )
            
            if similar_article:
                merged = self._merge_articles(similar_article, article)
                unique_articles[similar_article.content_hash] = merged  
                duplicates_found += 1
            else:
                unique_articles[article.content_hash] = article
        
        logger.info(f"News deduplication: {len(articles)} → {len(unique_articles)} (-{duplicates_found} dupes)")
        return list(unique_articles.values())
    
    def _merge_articles(self, primary: NewsArticle, secondary: NewsArticle) -> NewsArticle:
        """Merge two duplicate articles, preserving best data"""
        # Use primary article as base
        merged = dataclasses.replace(primary)
        
        # Enhance with secondary article data
        if not merged.content and secondary.content:
            merged.content = secondary.content
            
        if not merged.sentiment_score and secondary.sentiment_score:
            merged.sentiment_score = secondary.sentiment_score
            
        # Merge symbol mentions
        all_symbols = set(merged.symbols_mentioned + secondary.symbols_mentioned)
        merged.symbols_mentioned = sorted(list(all_symbols))
        
        return merged
```

### URL Normalization
```python
class URLNormalizer:
    """Normalize URLs for consistent deduplication"""
    
    def __init__(self):
        self.utm_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term']
        self.tracking_params = ['ref', 'source', 'fbclid', 'gclid', 'msclkid']
    
    def normalize(self, url: str) -> str:
        """Normalize URL by removing tracking parameters and standardizing format"""
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Remove tracking parameters
            query_params = urllib.parse.parse_qs(parsed.query)
            clean_params = {
                k: v for k, v in query_params.items() 
                if k not in self.utm_params + self.tracking_params
            }
            
            # Rebuild URL
            clean_query = urllib.parse.urlencode(clean_params, doseq=True)
            normalized = urllib.parse.urlunparse((
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip('/'),
                parsed.params,
                clean_query,
                ''  # Remove fragment
            ))
            
            return normalized
            
        except Exception as e:
            logger.warning(f"URL normalization failed for {url}: {e}")
            return url
```

## Merge Precedence Rules

### News Provider Priority
```python
NEWS_PROVIDER_PRECEDENCE = {
    "finnhub": 1,      # Highest quality, social sentiment  
    "newsapi": 2,      # Good coverage, reliable
    "gnews": 3,        # Broad coverage, less detailed
    "alpha_vantage": 4 # Basic news, fallback only
}
```

### Price Provider Priority by Use Case
```python
PRICE_PROVIDER_PRIORITY = {
    "daily_ohlc": ["yfinance", "twelve_data", "alpha_vantage"],
    "intraday_us": ["alpaca", "twelve_data", "polygon"], 
    "intraday_intl": ["twelve_data", "alpha_vantage"],
    "historical": ["yfinance", "tiingo", "alpha_vantage"],
    "real_time": ["alpaca", "finnhub", "polygon"]
}
```

## Storage & Indexing Strategy

### Primary Keys  
```sql
-- Price data table
CREATE TABLE price_data (
    canonical_id VARCHAR(255) PRIMARY KEY,  -- provider|symbol|interval|timestamp
    provider VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL, 
    timestamp TIMESTAMP NOT NULL,
    interval VARCHAR(10) NOT NULL,
    -- OHLCV columns...
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_provider_timestamp (provider, timestamp)
);

-- News articles table  
CREATE TABLE news_articles (
    canonical_id VARCHAR(255) PRIMARY KEY,  -- content_hash|date
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    canonical_url VARCHAR(500) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    -- Article columns...
    INDEX idx_content_hash (content_hash),
    INDEX idx_published_symbols (published_at, symbols_mentioned),
    FULLTEXT idx_content_search (title, summary, content)
);
```

### Deduplication Performance
- **Price Data**: O(1) lookup by canonical_id
- **News Articles**: O(1) content hash + O(n) similarity for edge cases  
- **Batch Processing**: Process in 1000-record chunks for memory efficiency
- **Cache TTL**: 24h for price data, 1h for news articles

## Data Quality Validation

### Price Data Validation
```python
def validate_price_data(point: PriceDataPoint) -> List[str]:
    """Return list of validation errors"""
    errors = []
    
    if point.high < point.low:
        errors.append("High price less than low price")
        
    if point.open < 0 or point.close < 0:
        errors.append("Negative price values")
        
    if not (point.low <= point.open <= point.high and point.low <= point.close <= point.high):
        errors.append("OHLC values inconsistent")
        
    if point.volume < 0:
        errors.append("Negative volume")
        
    return errors
```

### News Article Validation  
```python
def validate_news_article(article: NewsArticle) -> List[str]:
    """Return list of validation errors"""
    errors = []
    
    if len(article.title) < 10:
        errors.append("Title too short")
        
    if not article.url.startswith(('http://', 'https://')):
        errors.append("Invalid URL format")
        
    if article.published_at > datetime.now(timezone.utc):
        errors.append("Future publish date")
        
    if article.sentiment_score and not -1.0 <= article.sentiment_score <= 1.0:
        errors.append("Sentiment score out of range")
        
    return errors
```

**Implementation Priority**: Content hash deduplication (Week 1), URL normalization (Week 2), similarity matching (Week 3).