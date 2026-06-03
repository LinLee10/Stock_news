# Alpha Vantage Infinite Loop - Analysis & Fix

## Problem Identification

### Root Cause Analysis
**Location**: `services/alpha_vantage_manager.py:394-402` and `577-583`

**Issue**: The current re-queueing logic can create infinite loops:

1. **Backoff Re-queue (Lines 394-402)**:
```python
if request.backoff_until and datetime.now(timezone.utc) < request.backoff_until:
    # ... 
    await asyncio.sleep(min(backoff_seconds, 60))  # Capped at 1 minute
    await self.request_queue.put(request)  # Re-queues immediately
    return
```

2. **Retry Re-queue (Lines 577-583)**:
```python
if request.attempts < self.config.retry_attempts:
    # Calculate backoff...
    request.backoff_until = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
    await self.request_queue.put(request)  # Re-queues for retry
```

### Infinite Loop Scenarios

#### Scenario 1: Daily Quota Exhausted + Continuous Re-queueing
1. Daily quota (25 calls) exhausted
2. New request comes in → gets processed → fails with quota exceeded
3. Request gets retried with exponential backoff
4. After backoff expires, request processed again → fails again
5. **LOOP**: Process continues indefinitely, consuming CPU

#### Scenario 2: Rate Limit + Short Backoff Loop
1. Minute quota (5 calls) exhausted  
2. Request waits for backoff (max 60 seconds from line 400)
3. After 60 seconds, request processed → may still hit rate limit
4. **LOOP**: Tight 60-second cycles if rate limit persists

## Minimal Reproduction Test

```python
# tests/unit/test_alpha_vantage_loop_fix.py
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

from services.alpha_vantage_manager import (
    AlphaVantageManager, APIRequest, RequestPriority, APIResponse
)

class TestAlphaVantageLoopFix:
    
    @pytest.mark.asyncio
    async def test_daily_quota_exhausted_stops_enqueuing(self):
        """Test that daily exhaustion sets closed-for-day flag"""
        manager = AlphaVantageManager("test-key")
        
        # Mock daily bucket as exhausted
        manager.daily_bucket.tokens = 0
        manager._daily_closed = True  # NEW: This flag should exist
        
        request = APIRequest("AAPL", "TIME_SERIES_DAILY", RequestPriority.PORTFOLIO)
        
        # Should reject immediately without queuing
        response = await manager._submit_request(request, timeout=5.0)
        
        assert not response.success
        assert "daily quota exhausted" in response.error_message.lower()
        # Should not have been added to queue
        assert manager.request_queue.empty()
    
    @pytest.mark.asyncio  
    async def test_backoff_request_advances_state(self):
        """Test that requests with backoff don't re-queue in tight loop"""
        manager = AlphaVantageManager("test-key")
        await manager.start()
        
        try:
            # Create request with future backoff
            request = APIRequest("AAPL", "TIME_SERIES_DAILY", RequestPriority.PORTFOLIO)
            request.backoff_until = datetime.now(timezone.utc) + timedelta(minutes=5)
            request.attempts = 1
            
            # Process request (should NOT re-queue immediately)  
            process_start = datetime.now(timezone.utc)
            await manager._process_request(request)
            process_end = datetime.now(timezone.utc)
            
            # Should have waited at least some time, not returned immediately
            elapsed = (process_end - process_start).total_seconds()
            assert elapsed > 1.0, "Request should wait during backoff, not return immediately"
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_429_response_bounded_retry(self):
        """Test that 429/Note responses don't cause CPU spin"""
        manager = AlphaVantageManager("test-key")
        
        # Mock API to always return rate limit error
        mock_response = APIResponse(
            success=False,
            data=None, 
            error_message="API rate limit reached",
            response_time_ms=100
        )
        
        with patch.object(manager, '_make_api_call', return_value=mock_response):
            request = APIRequest("AAPL", "TIME_SERIES_DAILY", RequestPriority.PORTFOLIO)
            
            start_time = datetime.now(timezone.utc)
            await manager._process_request(request)
            end_time = datetime.now(timezone.utc)
            
            # Should not complete immediately (should apply backoff)
            elapsed = (end_time - start_time).total_seconds()
            assert elapsed > 0.5, "Rate limited request should not complete immediately"
            
            # Should not exceed maximum retry attempts
            assert request.attempts <= manager.config.retry_attempts
```

## Proposed Fix

### 1. Add Daily Exhaustion Guard

```python
# Add to AlphaVantageManager.__init__
self._daily_closed = False  # NEW: Track daily quota exhaustion
self._daily_closed_until = None  # NEW: When to reset the flag

async def _check_daily_exhaustion_guard(self) -> bool:
    """Check if we should accept new requests (daily quota guard)"""
    # Reset flag if new day
    now = datetime.now(timezone.utc)
    if (self._daily_closed_until and 
        now >= self._daily_closed_until):
        self._daily_closed = False
        self._daily_closed_until = None
        logger.info("Daily quota guard reset - accepting new requests")
    
    return not self._daily_closed

async def _set_daily_exhaustion_guard(self):
    """Set daily exhaustion flag to prevent new requests"""
    if not self._daily_closed:
        self._daily_closed = True
        # Reset at next UTC midnight (when quotas typically reset)
        tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
        self._daily_closed_until = datetime.combine(tomorrow, datetime.min.time()).replace(tzinfo=timezone.utc)
        
        logger.warning(f"Daily quota exhausted - blocking new requests until {self._daily_closed_until}")
```

### 2. Fix Backoff Re-queueing Logic

```python
async def _process_request(self, request: APIRequest):
    """Process a single API request with rate limiting and retry logic"""
    
    # NEW: Check daily exhaustion guard first
    if not await self._check_daily_exhaustion_guard():
        # Don't process or re-queue if daily quota exhausted
        logger.debug(f"Request blocked by daily exhaustion guard: {request.ticker}")
        return
    
    # Check if request needs to wait due to backoff
    if request.backoff_until and datetime.now(timezone.utc) < request.backoff_until:
        backoff_seconds = (request.backoff_until - datetime.now(timezone.utc)).total_seconds()
        logger.debug("Request waiting for backoff", 
                    ticker=request.ticker, 
                    backoff_seconds=backoff_seconds)
        
        # FIXED: Don't re-queue immediately, use proper bounded wait
        if backoff_seconds > 0:
            wait_time = min(backoff_seconds, 300)  # Max 5 minutes
            logger.debug(f"Waiting {wait_time}s for backoff on {request.ticker}")
            await asyncio.sleep(wait_time)
            
            # Clear backoff and process (don't re-queue)
            request.backoff_until = None
            # Fall through to process the request
    
    # Rate limiting checks  
    try:
        await self._enforce_rate_limits()
    except DailyQuotaExhaustedError:  # NEW exception
        await self._set_daily_exhaustion_guard()
        return
    
    # ... rest of method unchanged
```

### 3. Bound Rate Limit Enforcement

```python
class DailyQuotaExhaustedError(Exception):
    """Raised when daily quota is exhausted and should not retry"""
    pass

async def _enforce_rate_limits(self):
    """Enforce rate limiting before making requests"""
    # Check minute bucket
    minute_attempts = 0
    while not await self.minute_bucket.consume():
        wait_time = await self.minute_bucket.time_until_token_available()
        if wait_time > 300:  # Don't wait more than 5 minutes
            logger.warning(f"Minute rate limit wait too long ({wait_time}s), aborting request")
            raise DailyQuotaExhaustedError("Minute rate limit wait excessive")
            
        logger.debug("Rate limited - waiting for minute bucket", wait_seconds=wait_time)
        await asyncio.sleep(wait_time + 1)  # Add small buffer
        
        minute_attempts += 1
        if minute_attempts > 3:  # Prevent infinite minute-bucket loops
            logger.warning("Too many minute rate limit retries, aborting")
            raise DailyQuotaExhaustedError("Excessive minute rate limit retries")
    
    # Check daily bucket  
    if not await self.daily_bucket.consume():
        logger.warning("Daily rate limit reached")
        raise DailyQuotaExhaustedError("Daily quota exhausted")
    
    # ... rest unchanged
```

### 4. Fix Retry Re-queueing

```python  
async def _handle_request_failure(self, request: APIRequest, response: APIResponse):
    """Handle failed API requests with retry logic"""
    request.attempts += 1
    request.last_attempt_at = datetime.now(timezone.utc)
    
    # Check for daily quota exhaustion in response
    if ("rate limit" in response.error_message.lower() and 
        "daily" in response.error_message.lower()):
        logger.warning("Daily quota exhausted per API response")
        await self._set_daily_exhaustion_guard()
        self.stats['failed_requests'] += 1
        return  # Don't retry
    
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
        
        # FIXED: Re-queue with backoff, will be processed by worker loop naturally
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
```

## Implementation Steps

1. **Week 1**: Add daily exhaustion guard and exception handling
2. **Week 1**: Fix backoff re-queueing to use bounded waits  
3. **Week 1**: Add comprehensive unit tests for loop scenarios
4. **Week 2**: Deploy with monitoring to verify fix effectiveness

**Risk Mitigation**: All changes are backwards compatible and fail-safe (block requests rather than loop infinitely).