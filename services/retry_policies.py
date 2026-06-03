#!/usr/bin/env python3
"""
Retry policies and circuit breaker utilities for external API calls.

Provides exponential backoff with jitter and basic circuit breaker patterns
to handle rate limiting and temporary failures gracefully.

F09 additions: Async I/O support with asyncio gather, semaphore limits,
exponential backoff, circuit breakers, and local result caching.
"""

import asyncio
import aiohttp
import time
import random
import logging
import hashlib
import json
from typing import Callable, Tuple, Type, Any, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def retry_with_backoff(
    fn: Callable,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    max_retries: int = 4,
    base_delay: float = 2.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    debug: bool = False
) -> Any:
    """
    Execute a function with exponential backoff on specified exceptions.
    
    Args:
        fn: Function to execute
        retry_on: Tuple of exception types to retry on
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds before first retry
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Add random jitter to prevent thundering herd
        debug: Enable detailed retry logging
        
    Returns:
        Result of fn() if successful
        
    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            result = fn()
            if debug and attempt > 0:
                logger.info(f"Function succeeded on attempt {attempt + 1}")
            return result
            
        except retry_on as e:
            last_exception = e
            
            if attempt == max_retries:
                if debug:
                    logger.error(f"Function failed after {max_retries + 1} attempts: {e}")
                break
                
            # Calculate delay with exponential backoff
            delay = base_delay * (backoff_multiplier ** attempt)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay += random.uniform(0, delay * 0.1)  # 10% jitter
                
            if debug:
                logger.warning(f"Attempt {attempt + 1} failed ({e}), retrying in {delay:.1f}s...")
            
            # Injectable sleep for tests
            import os
            if os.getenv("YF_TEST_FAST_BACKOFF", "0") == "1":
                pass  # Skip sleep in tests
            else:
                time.sleep(delay)
            
        except Exception as e:
            # Don't retry on non-retriable exceptions
            if debug:
                logger.error(f"Non-retriable exception occurred: {e}")
            raise
    
    # Re-raise the last exception if we've exhausted retries
    raise last_exception


class CircuitBreaker:
    """
    Simple circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing fast, not calling the function
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, fn: Callable) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            fn: Function to execute
            
        Returns:
            Result of fn() if successful
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
                
        try:
            result = fn()
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        
    def _on_success(self):
        """Handle successful function execution."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker reset to CLOSED")
            
    def _on_failure(self):
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is in OPEN state."""
    pass


def with_circuit_breaker(
    circuit_breaker: CircuitBreaker,
    fn: Callable,
    fallback: Optional[Callable] = None
) -> Any:
    """
    Execute function with circuit breaker protection.
    
    Args:
        circuit_breaker: CircuitBreaker instance
        fn: Function to execute
        fallback: Optional fallback function if circuit is open
        
    Returns:
        Result of fn() or fallback() if circuit is open
    """
    try:
        return circuit_breaker.call(fn)
    except CircuitBreakerOpenError:
        if fallback:
            logger.info("Circuit breaker open, using fallback")
            return fallback()
        raise


# BEGIN F09 - Async I/O enhancements

@dataclass
class AsyncRetryConfig:
    """Configuration for async retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed


@dataclass
class AsyncCircuitBreakerConfig:
    """Configuration for async circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0


@dataclass
class CacheConfig:
    """Configuration for local result caching"""
    enabled: bool = True
    ttl_seconds: int = 86400  # 24 hours
    max_entries: int = 1000


class AsyncCircuitBreaker:
    """
    Async circuit breaker implementation per host to prevent cascading failures
    """
    
    def __init__(self, config: AsyncCircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute async function through circuit breaker"""
        async with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    logger.info(f"F09: Circuit breaker {self.name}: Attempting half-open transition")
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                else:
                    raise AsyncCircuitBreakerOpenError(f"F09: Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() > self.config.recovery_timeout
    
    async def _record_success(self):
        """Record successful operation"""
        async with self.lock:
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"F09: Circuit breaker {self.name}: Closing after {self.success_count} successes")
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == "CLOSED":
                self.failure_count = 0
    
    async def _record_failure(self, exception: Exception):
        """Record failed operation"""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == "CLOSED":
                if self.failure_count >= self.config.failure_threshold:
                    logger.warning(f"F09: Circuit breaker {self.name}: Opening after {self.failure_count} failures")
                    self.state = "OPEN"
            elif self.state == "HALF_OPEN":
                logger.warning(f"F09: Circuit breaker {self.name}: Returning to OPEN from half-open")
                self.state = "OPEN"
                self.success_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class AsyncCircuitBreakerOpenError(Exception):
    """Raised when async circuit breaker is open"""
    pass


class AsyncLocalCache:
    """
    Simple in-memory cache with TTL for async HTTP results
    Idempotent with cache key = URL+params; TTL 24h default
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()
    
    def _generate_key(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> str:
        """Generate idempotent cache key from URL and parameters"""
        key_data = {
            'url': url,
            'params': params or {},
            # Exclude authorization headers from cache key for security
            'headers': {k: v for k, v in (headers or {}).items() if k.lower() != 'authorization'}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[Any]:
        """Get cached result if available and fresh"""
        if not self.config.enabled:
            return None
        
        key = self._generate_key(url, params, headers)
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.config.ttl_seconds:
                    logger.debug(f"F09: Cache hit for {url}")
                    return entry['data']
                else:
                    # Expired entry
                    del self.cache[key]
        
        return None
    
    async def set(self, url: str, data: Any, params: Optional[Dict] = None, headers: Optional[Dict] = None):
        """Store result in cache"""
        if not self.config.enabled:
            return
        
        key = self._generate_key(url, params, headers)
        async with self.lock:
            # Simple LRU: remove oldest entries when at capacity
            if len(self.cache) >= self.config.max_entries:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
                del self.cache[oldest_key]
            
            self.cache[key] = {
                'data': data,
                'timestamp': time.time(),
                'url': url
            }
            logger.debug(f"F09: Cached result for {url}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'enabled': self.config.enabled,
            'entries': len(self.cache),
            'max_entries': self.config.max_entries,
            'ttl_seconds': self.config.ttl_seconds
        }
    
    async def clear(self):
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()
            logger.info("F09: Cache cleared")


# Global async instances
_async_circuit_breakers: Dict[str, AsyncCircuitBreaker] = {}
_async_cache = AsyncLocalCache(CacheConfig())


def get_async_circuit_breaker(host: str, config: Optional[AsyncCircuitBreakerConfig] = None) -> AsyncCircuitBreaker:
    """Get or create async circuit breaker for host"""
    if host not in _async_circuit_breakers:
        breaker_config = config or AsyncCircuitBreakerConfig()
        _async_circuit_breakers[host] = AsyncCircuitBreaker(breaker_config, name=host)
        logger.info(f"F09: Created async circuit breaker for host: {host}")
    return _async_circuit_breakers[host]


async def retry_async(
    func: Callable,
    *args,
    retry_config: Optional[AsyncRetryConfig] = None,
    circuit_breaker: Optional[AsyncCircuitBreaker] = None,
    **kwargs
) -> Any:
    """
    Execute async function with retry logic and optional circuit breaker
    Implements exponential backoff with jitter and capped retries
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        retry_config: Retry configuration
        circuit_breaker: Circuit breaker instance
        **kwargs: Keyword arguments for func
    
    Returns:
        Function result
        
    Raises:
        Last exception if all retries exhausted
    """
    config = retry_config or AsyncRetryConfig()
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            if circuit_breaker:
                return await circuit_breaker.call(func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)
                
        except Exception as e:
            last_exception = e
            
            if attempt == config.max_retries:
                logger.error(f"F09: All {config.max_retries + 1} attempts failed. Last error: {e}")
                break
            
            delay = _calculate_async_delay(attempt, config)
            logger.warning(f"F09: Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)
    
    raise last_exception


def _calculate_async_delay(attempt: int, config: AsyncRetryConfig) -> float:
    """Calculate delay for async retry attempt"""
    if config.backoff_strategy == "exponential":
        delay = config.base_delay * (config.exponential_base ** attempt)
    elif config.backoff_strategy == "linear":
        delay = config.base_delay * (attempt + 1)
    else:  # fixed
        delay = config.base_delay
    
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
    
    return delay


class AsyncHTTPClient:
    """
    F09: HTTP client with built-in retry logic, circuit breakers, semaphore limits, and caching
    Resilient to transient failures with exponential backoff and circuit breaker per host
    """
    
    def __init__(
        self,
        retry_config: Optional[AsyncRetryConfig] = None,
        circuit_config: Optional[AsyncCircuitBreakerConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        semaphore_limit: int = 5  # Default MAX_CONCURRENCY
    ):
        self.retry_config = retry_config or AsyncRetryConfig()
        self.circuit_config = circuit_config or AsyncCircuitBreakerConfig()
        self.cache_config = cache_config or CacheConfig()
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking and telemetry
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'retries': 0,
            'circuit_breaker_opens': 0,
            'average_latency': 0.0,
            'total_latency': 0.0,
            'failures': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.circuit_config.timeout)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Perform resilient GET request with retry, circuit breaker, and caching
        
        Returns:
            Dict with 'status', 'data', 'from_cache', 'latency' keys
        """
        return await self._request('GET', url, params=params, headers=headers, **kwargs)
    
    async def post(self, url: str, data: Any = None, json_data: Any = None, headers: Optional[Dict] = None, **kwargs) -> Dict:
        """Perform resilient POST request with retry and circuit breaker"""
        return await self._request('POST', url, data=data, json=json_data, headers=headers, **kwargs)
    
    async def _request(self, method: str, url: str, **kwargs) -> Dict:
        """Internal request method with all F09 resilience features"""
        start_time = time.time()
        self.stats['requests'] += 1
        
        # Check cache for GET requests (idempotent)
        if method == 'GET':
            cached_result = await _async_cache.get(url, kwargs.get('params'), kwargs.get('headers'))
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return {
                    'status': 200,
                    'data': cached_result,
                    'from_cache': True,
                    'latency': 0.0
                }
        
        # Extract host for circuit breaker per host
        parsed_url = urlparse(url)
        host = f"{parsed_url.scheme}://{parsed_url.netloc}"
        circuit_breaker = get_async_circuit_breaker(host, self.circuit_config)
        
        # Semaphore limits concurrent requests (configurable MAX_CONCURRENCY)
        async with self.semaphore:
            try:
                result = await retry_async(
                    self._do_request,
                    method, url,
                    retry_config=self.retry_config,
                    circuit_breaker=circuit_breaker,
                    **kwargs
                )
                
                # Cache successful GET responses for 24h TTL
                if method == 'GET' and result['status'] == 200:
                    await _async_cache.set(url, result['data'], kwargs.get('params'), kwargs.get('headers'))
                
                return result
                
            except AsyncCircuitBreakerOpenError:
                self.stats['circuit_breaker_opens'] += 1
                logger.error(f"F09: Circuit breaker open for {host}")
                self.stats['failures'] += 1
                raise
            except Exception as e:
                self.stats['failures'] += 1
                logger.error(f"F09: Request failed after all retries: {e}")
                raise
            finally:
                latency = time.time() - start_time
                self.stats['total_latency'] += latency
                if self.stats['requests'] > 0:
                    self.stats['average_latency'] = self.stats['total_latency'] / self.stats['requests']
    
    async def _do_request(self, method: str, url: str, **kwargs) -> Dict:
        """Perform actual HTTP request"""
        if not self.session:
            raise RuntimeError("F09: AsyncHTTPClient not initialized. Use 'async with' context.")
        
        async with self.session.request(method, url, **kwargs) as response:
            # Server errors (5xx) trigger retries, client errors (4xx) do not
            if response.status >= 500:
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"F09: Server error: {response.status}"
                )
            
            # Parse response based on content type
            if response.content_type and 'application/json' in response.content_type:
                data = await response.json()
            else:
                data = await response.text()
            
            return {
                'status': response.status,
                'data': data,
                'from_cache': False,
                'headers': dict(response.headers),
                'url': str(response.url)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get F09 performance statistics and telemetry"""
        circuit_stats = [cb.get_stats() for cb in _async_circuit_breakers.values()]
        cache_stats = _async_cache.get_stats()
        
        return {
            'client_stats': self.stats.copy(),
            'circuit_breakers': circuit_stats,
            'cache': cache_stats,
            'active_circuit_breakers': len(_async_circuit_breakers),
            'timestamp': datetime.now().isoformat()
        }


async def make_resilient_requests(
    urls: List[str],
    method: str = 'GET',
    max_concurrency: int = 5,
    **kwargs
) -> List[Dict]:
    """
    F09: Make multiple resilient requests concurrently with asyncio.gather and semaphore limits
    Implements the core F09 pattern: async gather with semaphore limits for network calls
    
    Args:
        urls: List of URLs to fetch
        method: HTTP method (GET/POST)
        max_concurrency: Maximum concurrent requests (semaphore limit)
        **kwargs: Additional request parameters
    
    Returns:
        List of response dictionaries, same order as input URLs
        Failed requests return error dict instead of raising
    """
    async with AsyncHTTPClient(semaphore_limit=max_concurrency) as client:
        # Create tasks for all requests
        if method.upper() == 'GET':
            tasks = [_safe_request(client.get, url, **kwargs) for url in urls]
        elif method.upper() == 'POST':
            tasks = [_safe_request(client.post, url, **kwargs) for url in urls]
        else:
            raise ValueError(f"F09: Unsupported HTTP method: {method}")
        
        # Execute with asyncio.gather - core F09 pattern
        logger.info(f"F09: Executing {len(tasks)} requests with max_concurrency={max_concurrency}")
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return results


async def _safe_request(request_func: Callable, url: str, **kwargs) -> Dict:
    """
    Wrapper to catch exceptions and return error dict instead of raising
    Ensures daily run is resilient to transient failures (AC1)
    """
    try:
        return await request_func(url, **kwargs)
    except Exception as e:
        logger.warning(f"F09: Request failed for {url}: {e}")
        return {
            'status': 0,
            'data': None,
            'error': str(e),
            'from_cache': False,
            'url': url,
            'failed': True
        }


def load_async_config() -> Dict[str, Any]:
    """Load F09 async I/O configuration from environment"""
    import os
    
    return {
        'max_concurrency': int(os.getenv('MAX_CONCURRENCY', '5')),
        'backoff_max': int(os.getenv('BACKOFF_MAX', '60')),
        'circuit_failure_threshold': int(os.getenv('CIRCUIT_FAILURE_THRESHOLD', '5')),
        'circuit_recovery_timeout': int(os.getenv('CIRCUIT_RECOVERY_TIMEOUT', '60')),
        'cache_ttl': int(os.getenv('CACHE_TTL', '86400')),
        'request_timeout': int(os.getenv('REQUEST_TIMEOUT', '30'))
    }


def get_async_stats() -> Dict[str, Any]:
    """Get global F09 async I/O performance and telemetry stats"""
    return {
        'circuit_breakers': {name: cb.get_stats() for name, cb in _async_circuit_breakers.items()},
        'cache': _async_cache.get_stats(),
        'timestamp': datetime.now().isoformat(),
        'feature': 'F09_async_io'
    }


async def reset_async_state():
    """Reset all F09 async state (useful for testing)"""
    global _async_circuit_breakers
    _async_circuit_breakers.clear()
    await _async_cache.clear()
    logger.info("F09: Reset async I/O state")

# END F09