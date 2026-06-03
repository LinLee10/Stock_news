"""
AlphaVantageManager - Intelligent Alpha Vantage API management with rate limiting and resource allocation.

This module implements sophisticated Alpha Vantage API management:
- Token bucket rate limiter (5 calls/minute, 25/day)
- Smart request queuing with priority levels
- Exponential backoff on rate limit errors
- Resource allocation for portfolio vs watchlist tickers
- Intelligent retry mechanisms
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import os

import requests
import structlog
import redis

logger = structlog.get_logger(__name__)


class RequestPriority(Enum):
    """Request priority levels for resource allocation"""
    CRITICAL = 1      # Emergency/manual requests
    PORTFOLIO = 2     # Portfolio tickers (highest scheduled priority)
    WATCHLIST = 3     # Watchlist tickers
    RESEARCH = 4      # Research/analysis requests
    BACKGROUND = 5    # Background refresh/preloading


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    calls_per_minute: int = 5
    calls_per_day: int = 25
    min_delay_seconds: float = 12.0  # 12 seconds between calls minimum
    backoff_base: float = 2.0        # Exponential backoff base
    max_backoff_seconds: float = 300.0  # Max 5 minutes backoff
    retry_attempts: int = 3


@dataclass 
class APIRequest:
    """API request with metadata"""
    ticker: str
    function: str
    priority: RequestPriority
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    backoff_until: Optional[datetime] = None
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value < other.priority.value


class APIResponse(NamedTuple):
    """API response container"""
    success: bool
    data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    response_time_ms: int
    cached: bool = False


class TokenBucket:
    """Token bucket for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float, refill_period: float = 60.0):
        """
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per refill period
            refill_period: Refill period in seconds (default 60s for per-minute rates)
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.refill_period = refill_period
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        async with self.lock:
            await self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed >= self.refill_period:
            periods_elapsed = elapsed / self.refill_period
            tokens_to_add = int(periods_elapsed * self.refill_rate)
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    async def time_until_token_available(self) -> float:
        """Get seconds until next token is available"""
        async with self.lock:
            await self._refill()
            
            if self.tokens > 0:
                return 0.0
            
            # Calculate when next refill will happen
            now = time.time()
            time_since_refill = now - self.last_refill
            time_to_next_refill = self.refill_period - time_since_refill
            return max(0.0, time_to_next_refill)


class AlphaVantageManager:
    """
    Intelligent Alpha Vantage API manager with sophisticated rate limiting and resource allocation.
    
    Features:
    - Token bucket rate limiting (5 calls/minute, 25/day)
    - Priority-based request queuing
    - Smart resource allocation for different ticker types
    - Exponential backoff and retry logic
    - Request scheduling and batching
    - Usage analytics and optimization
    """
    
    def __init__(self, 
                 api_key: str, 
                 redis_client: Optional[redis.Redis] = None,
                 config: Optional[RateLimitConfig] = None):
        
        self.api_key = api_key
        self.redis_client = redis_client
        self.config = config or RateLimitConfig()
        
        # Rate limiting
        self.minute_bucket = TokenBucket(
            capacity=self.config.calls_per_minute,
            refill_rate=self.config.calls_per_minute,
            refill_period=60.0
        )
        self.daily_bucket = TokenBucket(
            capacity=self.config.calls_per_day,
            refill_rate=self.config.calls_per_day, 
            refill_period=86400.0  # 24 hours
        )
        
        # Request management
        self.request_queue = asyncio.PriorityQueue()
        self.active_requests: Dict[str, APIRequest] = {}
        self.last_request_time = 0.0
        
        # Resource allocation tracking
        self.daily_allocations = {
            RequestPriority.CRITICAL: 1,      # 1 call reserved for emergencies
            RequestPriority.PORTFOLIO: 20,   # 20 calls for portfolio tickers
            RequestPriority.WATCHLIST: 4     # 4 calls for watchlist tickers
        }
        self.allocation_used = {priority: 0 for priority in self.daily_allocations}
        
        # Analytics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'rate_limited_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'avg_response_time_ms': 0.0,
            'allocation_usage': {}
        }
        
        # Worker control
        self.is_running = False
        self.worker_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the API manager worker"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        
        # Load daily allocations from Redis if available
        await self._load_daily_state()
        
        logger.info("AlphaVantage manager started", 
                   daily_limit=self.config.calls_per_day,
                   minute_limit=self.config.calls_per_minute)
    
    async def stop(self):
        """Stop the API manager worker"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        # Save daily state to Redis
        await self._save_daily_state()
        
        logger.info("AlphaVantage manager stopped")
    
    async def get_daily_data(self, ticker: str, 
                           priority: RequestPriority = RequestPriority.RESEARCH,
                           timeout: float = 30.0) -> APIResponse:
        """
        Get daily time series data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            priority: Request priority level
            timeout: Request timeout in seconds
            
        Returns:
            APIResponse with data or error information
        """
        request = APIRequest(
            ticker=ticker,
            function="TIME_SERIES_DAILY",
            priority=priority,
            params={
                "symbol": ticker,
                "outputsize": "full"
            }
        )
        
        return await self._submit_request(request, timeout)
    
    async def get_intraday_data(self, ticker: str,
                              interval: str = "5min",
                              priority: RequestPriority = RequestPriority.RESEARCH,
                              timeout: float = 30.0) -> APIResponse:
        """
        Get intraday data for a ticker.
        
        Args:
            ticker: Stock ticker symbol  
            interval: Data interval (1min, 5min, 15min, 30min, 60min)
            priority: Request priority level
            timeout: Request timeout in seconds
            
        Returns:
            APIResponse with data or error information
        """
        request = APIRequest(
            ticker=ticker,
            function="TIME_SERIES_INTRADAY",
            priority=priority,
            params={
                "symbol": ticker,
                "interval": interval,
                "outputsize": "full"
            }
        )
        
        return await self._submit_request(request, timeout)
    
    async def get_quote(self, ticker: str,
                       priority: RequestPriority = RequestPriority.RESEARCH,
                       timeout: float = 15.0) -> APIResponse:
        """
        Get real-time quote for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            priority: Request priority level  
            timeout: Request timeout in seconds
            
        Returns:
            APIResponse with quote data
        """
        request = APIRequest(
            ticker=ticker,
            function="GLOBAL_QUOTE",
            priority=priority,
            params={"symbol": ticker}
        )
        
        return await self._submit_request(request, timeout)
    
    async def _submit_request(self, request: APIRequest, timeout: float) -> APIResponse:
        """Submit a request and wait for response"""
        request_id = f"{request.ticker}_{request.function}_{int(time.time())}"
        
        # Check allocation limits
        if not await self._check_allocation_limit(request.priority):
            return APIResponse(
                success=False,
                data=None,
                error_message=f"Daily allocation limit reached for {request.priority.name}",
                response_time_ms=0
            )
        
        # Add to queue
        await self.request_queue.put(request)
        self.active_requests[request_id] = request
        
        # Wait for completion
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                if request_id not in self.active_requests:
                    # Request completed, get result from Redis cache or return error
                    break
                await asyncio.sleep(0.1)
            
            # Check if request was completed
            if request_id in self.active_requests:
                # Request timed out
                del self.active_requests[request_id]
                return APIResponse(
                    success=False,
                    data=None,
                    error_message="Request timeout",
                    response_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Try to get result from cache
            if self.redis_client:
                cached_result = await self._get_cached_response(request)
                if cached_result:
                    return cached_result
            
            return APIResponse(
                success=False,
                data=None, 
                error_message="Request completed but result not found",
                response_time_ms=int((time.time() - start_time) * 1000)
            )
            
        except Exception as e:
            logger.error("Request submission failed", error=str(e))
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                response_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _worker_loop(self):
        """Main worker loop for processing API requests"""
        logger.info("AlphaVantage worker started")
        
        while self.is_running:
            try:
                # Get next request from queue (with timeout to allow shutdown)
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                await self._process_request(request)
                
            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.error("Worker error", error=str(e))
                await asyncio.sleep(1)
        
        logger.info("AlphaVantage worker stopped")
    
    async def _process_request(self, request: APIRequest):
        """Process a single API request with rate limiting and retry logic"""
        
        # Check if request needs to wait due to backoff
        if request.backoff_until and datetime.now(timezone.utc) < request.backoff_until:
            backoff_seconds = (request.backoff_until - datetime.now(timezone.utc)).total_seconds()
            logger.debug("Request waiting for backoff", 
                        ticker=request.ticker, 
                        backoff_seconds=backoff_seconds)
            # Re-queue for later
            await asyncio.sleep(min(backoff_seconds, 60))  # Don't sleep more than 1 minute
            await self.request_queue.put(request)
            return
        
        # Rate limiting checks
        await self._enforce_rate_limits()
        
        # Make the API call
        start_time = time.time()
        
        try:
            response = await self._make_api_call(request)
            processing_time = int((time.time() - start_time) * 1000)
            
            # Handle response
            if response.success:
                self.stats['successful_requests'] += 1
                await self._increment_allocation_usage(request.priority)
                
                # Cache successful response
                if self.redis_client:
                    await self._cache_response(request, response)
                
                logger.debug("API request successful", 
                           ticker=request.ticker,
                           function=request.function,
                           response_time_ms=processing_time)
            else:
                # Handle failure
                await self._handle_request_failure(request, response)
            
            # Update stats
            self.stats['total_requests'] += 1
            self._update_avg_response_time(processing_time)
            
        except Exception as e:
            logger.error("API request processing failed", 
                        ticker=request.ticker, error=str(e))
            
            # Handle as failure
            error_response = APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                response_time_ms=int((time.time() - start_time) * 1000)
            )
            await self._handle_request_failure(request, error_response)
        
        finally:
            # Remove from active requests
            request_id = f"{request.ticker}_{request.function}_{int(request.created_at.timestamp())}"
            self.active_requests.pop(request_id, None)
    
    async def _make_api_call(self, request: APIRequest) -> APIResponse:
        """Make the actual API call to Alpha Vantage"""
        start_time = time.time()
        
        # Prepare parameters
        params = {
            "function": request.function,
            "apikey": self.api_key,
            **request.params
        }
        
        # Make request with timeout
        try:
            response = requests.get(
                "https://www.alphavantage.co/query",
                params=params,
                timeout=(5, 15)  # 5s connect, 15s read timeout
            )
            response.raise_for_status()
            
            data = response.json()
            processing_time = int((time.time() - start_time) * 1000)
            
            # Check for API errors
            if "Error Message" in data:
                return APIResponse(
                    success=False,
                    data=None,
                    error_message=data["Error Message"],
                    response_time_ms=processing_time
                )
            
            # Check for rate limit messages
            info_msg = data.get("Information", "")
            note_msg = data.get("Note", "")
            
            if ("rate limit" in info_msg.lower() or 
                "rate limit" in note_msg.lower() or
                "Thank you for using Alpha Vantage" in note_msg):
                
                self.stats['rate_limited_requests'] += 1
                return APIResponse(
                    success=False,
                    data=None,
                    error_message="API rate limit reached",
                    response_time_ms=processing_time
                )
            
            # Check for empty data
            expected_keys = {
                "TIME_SERIES_DAILY": "Time Series (Daily)",
                "TIME_SERIES_INTRADAY": f"Time Series ({request.params.get('interval', '5min')})",
                "GLOBAL_QUOTE": "Global Quote"
            }
            
            expected_key = expected_keys.get(request.function)
            if expected_key and expected_key not in data:
                return APIResponse(
                    success=False,
                    data=None,
                    error_message=f"No data returned for {request.ticker}",
                    response_time_ms=processing_time
                )
            
            return APIResponse(
                success=True,
                data=data,
                error_message=None,
                response_time_ms=processing_time
            )
            
        except requests.RequestException as e:
            processing_time = int((time.time() - start_time) * 1000)
            return APIResponse(
                success=False,
                data=None,
                error_message=f"Request failed: {str(e)}",
                response_time_ms=processing_time
            )
    
    async def _enforce_rate_limits(self):
        """Enforce rate limiting before making requests"""
        # Check minute bucket
        while not await self.minute_bucket.consume():
            wait_time = await self.minute_bucket.time_until_token_available()
            logger.debug("Rate limited - waiting for minute bucket", wait_seconds=wait_time)
            await asyncio.sleep(wait_time + 1)  # Add small buffer
        
        # Check daily bucket  
        while not await self.daily_bucket.consume():
            wait_time = await self.daily_bucket.time_until_token_available()
            logger.warning("Daily rate limit reached", wait_hours=wait_time/3600)
            await asyncio.sleep(min(wait_time, 3600))  # Don't wait more than 1 hour at a time
        
        # Enforce minimum delay between requests
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.config.min_delay_seconds:
            wait_time = self.config.min_delay_seconds - time_since_last
            logger.debug("Enforcing minimum delay", wait_seconds=wait_time)
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def _handle_request_failure(self, request: APIRequest, response: APIResponse):
        """Handle failed API requests with retry logic"""
        request.attempts += 1
        request.last_attempt_at = datetime.now(timezone.utc)
        
        # Check if we should retry
        if request.attempts < self.config.retry_attempts:
            # Calculate backoff time
            backoff_seconds = min(
                self.config.backoff_base ** request.attempts,
                self.config.max_backoff_seconds
            )
            
            # Add jitter to prevent thundering herd
            jitter = backoff_seconds * 0.1 * (0.5 - random.random())
            backoff_seconds += jitter
            
            request.backoff_until = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
            
            logger.info("Retrying failed request", 
                       ticker=request.ticker,
                       attempt=request.attempts,
                       backoff_seconds=backoff_seconds)
            
            # Re-queue for retry
            await self.request_queue.put(request)
        else:
            logger.error("Request failed permanently after retries",
                        ticker=request.ticker,
                        attempts=request.attempts,
                        error=response.error_message)
            
            self.stats['failed_requests'] += 1
            
            # Cache the failure to prevent immediate retry
            if self.redis_client:
                await self._cache_failure(request, response)
    
    async def _check_allocation_limit(self, priority: RequestPriority) -> bool:
        """Check if priority level has remaining allocation"""
        if priority not in self.daily_allocations:
            return True  # No limit for this priority
        
        allocated = self.daily_allocations[priority]
        used = self.allocation_used.get(priority, 0)
        
        if used >= allocated:
            logger.warning("Allocation limit reached", 
                          priority=priority.name,
                          used=used,
                          allocated=allocated)
            return False
        
        return True
    
    async def _increment_allocation_usage(self, priority: RequestPriority):
        """Increment allocation usage for a priority level"""
        if priority in self.daily_allocations:
            self.allocation_used[priority] = self.allocation_used.get(priority, 0) + 1
            
            # Save to Redis
            if self.redis_client:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                key = f"av_allocation:{today}:{priority.name}"
                try:
                    await self.redis_client.set(key, self.allocation_used[priority], ex=86400)
                except Exception as e:
                    logger.warning("Failed to save allocation usage", error=str(e))
    
    def _update_avg_response_time(self, response_time_ms: int):
        """Update average response time statistics"""
        if self.stats['total_requests'] == 0:
            self.stats['avg_response_time_ms'] = response_time_ms
        else:
            # Moving average
            current_avg = self.stats['avg_response_time_ms']
            total = self.stats['total_requests']
            self.stats['avg_response_time_ms'] = ((current_avg * (total - 1)) + response_time_ms) / total
    
    async def _cache_response(self, request: APIRequest, response: APIResponse):
        """Cache successful API response"""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"av_response:{request.function}:{request.ticker}"
            cache_data = {
                'data': response.data,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'response_time_ms': response.response_time_ms
            }
            
            # Cache for different durations based on data type
            ttl = 300  # 5 minutes default
            if request.function == "TIME_SERIES_DAILY":
                ttl = 3600  # 1 hour for daily data
            elif request.function == "GLOBAL_QUOTE":
                ttl = 60   # 1 minute for quotes
            
            await self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(cache_data, default=str)
            )
            
        except Exception as e:
            logger.warning("Failed to cache response", error=str(e))
    
    async def _get_cached_response(self, request: APIRequest) -> Optional[APIResponse]:
        """Get cached API response"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"av_response:{request.function}:{request.ticker}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                cache_obj = json.loads(cached_data)
                self.stats['cache_hits'] += 1
                
                return APIResponse(
                    success=True,
                    data=cache_obj['data'],
                    error_message=None,
                    response_time_ms=0,  # Cached response
                    cached=True
                )
        
        except Exception as e:
            logger.warning("Failed to get cached response", error=str(e))
        
        return None
    
    async def _cache_failure(self, request: APIRequest, response: APIResponse):
        """Cache request failure to prevent immediate retry"""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"av_failure:{request.function}:{request.ticker}"
            failure_data = {
                'error_message': response.error_message,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'attempts': request.attempts
            }
            
            # Cache failures for 1 hour to prevent retry storms
            await self.redis_client.setex(
                cache_key,
                3600,
                json.dumps(failure_data)
            )
            
        except Exception as e:
            logger.warning("Failed to cache failure", error=str(e))
    
    async def _load_daily_state(self):
        """Load daily allocation state from Redis"""
        if not self.redis_client:
            return
        
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            for priority in self.daily_allocations:
                key = f"av_allocation:{today}:{priority.name}"
                used = await self.redis_client.get(key)
                if used:
                    self.allocation_used[priority] = int(used)
        
        except Exception as e:
            logger.warning("Failed to load daily state", error=str(e))
    
    async def _save_daily_state(self):
        """Save daily allocation state to Redis"""
        if not self.redis_client:
            return
        
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            for priority, used in self.allocation_used.items():
                if used > 0:
                    key = f"av_allocation:{today}:{priority.name}"
                    await self.redis_client.setex(key, 86400, used)
        
        except Exception as e:
            logger.warning("Failed to save daily state", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        stats = self.stats.copy()
        stats['allocation_usage'] = {
            priority.name: {
                'used': self.allocation_used.get(priority, 0),
                'allocated': limit,
                'remaining': limit - self.allocation_used.get(priority, 0)
            }
            for priority, limit in self.daily_allocations.items()
        }
        
        return stats
    
    async def get_remaining_capacity(self) -> Dict[str, int]:
        """Get remaining API capacity"""
        minute_capacity = await self._get_bucket_capacity(self.minute_bucket)
        daily_capacity = await self._get_bucket_capacity(self.daily_bucket)
        
        return {
            'minute_remaining': minute_capacity,
            'daily_remaining': daily_capacity,
            'allocation_remaining': {
                priority.name: limit - self.allocation_used.get(priority, 0)
                for priority, limit in self.daily_allocations.items()
            }
        }
    
    async def _get_bucket_capacity(self, bucket: TokenBucket) -> int:
        """Get remaining tokens in bucket"""
        async with bucket.lock:
            await bucket._refill()
            return bucket.tokens


# Utility function for easy initialization
async def create_alpha_vantage_manager(api_key: str, 
                                     redis_url: Optional[str] = None) -> AlphaVantageManager:
    """Create and start an Alpha Vantage manager"""
    redis_client = None
    if redis_url:
        import redis.asyncio as redis
        redis_client = redis.from_url(redis_url)
    
    manager = AlphaVantageManager(api_key, redis_client)
    await manager.start()
    return manager


if __name__ == "__main__":
    import os
    import asyncio
    from dotenv import load_dotenv
    
    # Test the Alpha Vantage manager
    async def test_manager():
        load_dotenv("../config/secrets.env")
        api_key = os.getenv("ALPHA_VANTAGE_KEY")
        
        if not api_key:
            print("Missing ALPHA_VANTAGE_KEY")
            return
        
        manager = await create_alpha_vantage_manager(api_key)
        
        try:
            # Test portfolio ticker (high priority)
            print("Testing portfolio ticker...")
            response = await manager.get_daily_data("AAPL", RequestPriority.PORTFOLIO)
            print(f"AAPL response: success={response.success}, cached={response.cached}")
            
            # Test watchlist ticker (medium priority)  
            print("Testing watchlist ticker...")
            response = await manager.get_daily_data("NVDA", RequestPriority.WATCHLIST)
            print(f"NVDA response: success={response.success}, cached={response.cached}")
            
            # Get stats
            stats = manager.get_stats()
            print(f"Manager stats: {stats}")
            
            # Get capacity
            capacity = await manager.get_remaining_capacity()
            print(f"Remaining capacity: {capacity}")
            
        finally:
            await manager.stop()
    
    asyncio.run(test_manager())