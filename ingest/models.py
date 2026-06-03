"""
Data models for the compliant ingestion system.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PaywallStatus(str, Enum):
    """Paywall detection status"""
    FREE = "free"
    PAYWALLED = "paywalled"
    PREVIEW = "preview"  # Partial content available
    UNKNOWN = "unknown"


class FetchStrategy(str, Enum):
    """Strategy used to fetch content"""
    API = "api"
    RSS = "rss"
    PLAYWRIGHT = "playwright"
    AIOHTTP = "aiohttp"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RenderResult(BaseModel):
    """Result from browser rendering or HTTP fetch"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    url: str
    final_url: str
    html: str
    status_code: int
    strategy: FetchStrategy
    render_time_ms: int
    success: bool
    error_message: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)


class ExtractedArticle(BaseModel):
    """Extracted article content and metadata"""
    
    url: str
    canonical_url: Optional[str] = None
    title: str
    authors: List[str] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    source: str
    text: str
    word_count: int
    paywall_status: PaywallStatus = PaywallStatus.UNKNOWN
    selectors_used: List[str] = Field(default_factory=list)
    extraction_time_ms: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QualityDecision(BaseModel):
    """Content quality assessment result"""
    
    keep: bool
    reasons: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)
    duplicate_of: Optional[str] = None
    evaluation_time_ms: int = 0


class DomainPolicy(BaseModel):
    """Per-domain crawling policy"""
    
    domain: str
    allowed: bool = True
    crawl_delay_seconds: float = 12.0
    max_concurrency: int = 1
    user_agent_required: bool = True
    robots_txt_url: Optional[str] = None
    robots_last_checked: Optional[datetime] = None
    robots_disallowed_paths: List[str] = Field(default_factory=list)
    credibility_score: float = 0.5
    preferred_strategy: FetchStrategy = FetchStrategy.PLAYWRIGHT
    custom_selectors: Dict[str, str] = Field(default_factory=dict)


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status for a domain"""
    
    domain: str
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None
    success_count: int = 0


class IngestionJob(BaseModel):
    """Job for ingesting content from a URL"""
    
    job_id: str
    url: str
    domain: str
    priority: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attempts: int = 0
    max_attempts: int = 3
    strategy: Optional[FetchStrategy] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Result of processing an ingestion job"""
    
    job_id: str
    url: str
    success: bool
    article: Optional[ExtractedArticle] = None
    quality_decision: Optional[QualityDecision] = None
    error_message: Optional[str] = None
    processing_time_ms: int = 0
    strategy_used: Optional[FetchStrategy] = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RateLimitStatus(BaseModel):
    """Rate limiting status for a domain"""
    
    domain: str
    tokens_remaining: int
    tokens_per_period: int
    period_seconds: int
    last_request_time: Optional[datetime] = None
    next_allowed_time: Optional[datetime] = None