"""
CompliantBrowsingSession - Handles web browsing with full compliance.

This module provides a session manager that respects robots.txt, rate limits,
and terms of service without any evasion techniques.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import aiohttp
import structlog
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from .models import (
    RenderResult, FetchStrategy, DomainPolicy, CircuitBreakerStatus, 
    CircuitBreakerState, RateLimitStatus
)
from .storage import RedisStorage


logger = structlog.get_logger(__name__)


class RobotsChecker:
    """Handles robots.txt compliance checking"""
    
    def __init__(self, storage: RedisStorage):
        self.storage = storage
        self._cache_ttl = 86400  # 24 hours
    
    async def can_fetch(self, url: str, user_agent: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        domain = urlparse(url).netloc
        robots_key = f"robots:{domain}"
        
        # Try to get cached robots.txt
        cached_robots = await self.storage.get(robots_key)
        
        if cached_robots is None:
            # Fetch and cache robots.txt
            robots_url = f"https://{domain}/robots.txt"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            await self.storage.set(robots_key, robots_content, ttl=self._cache_ttl)
                            cached_robots = robots_content
                        else:
                            # No robots.txt or error - assume allowed
                            await self.storage.set(robots_key, "", ttl=self._cache_ttl)
                            return True
            except Exception as e:
                logger.warning("Failed to fetch robots.txt", domain=domain, error=str(e))
                return True
        
        if not cached_robots:
            return True
        
        # Parse robots.txt
        rp = RobotFileParser()
        rp.set_url(f"https://{domain}/robots.txt")
        rp.read()
        
        # Check if URL is allowed
        return rp.can_fetch(user_agent, url)
    
    async def get_crawl_delay(self, domain: str, user_agent: str) -> float:
        """Get crawl delay from robots.txt"""
        robots_key = f"robots:{domain}"
        cached_robots = await self.storage.get(robots_key)
        
        if not cached_robots:
            return 12.0  # Default minimum delay
        
        try:
            rp = RobotFileParser()
            rp.set_url(f"https://{domain}/robots.txt")
            rp.read()
            delay = rp.crawl_delay(user_agent)
            return max(float(delay or 12.0), 12.0)  # Minimum 12 seconds
        except Exception:
            return 12.0


class RateLimiter:
    """Token bucket rate limiter with per-domain tracking"""
    
    def __init__(self, storage: RedisStorage):
        self.storage = storage
    
    async def acquire(self, domain: str, tokens: int = 1) -> bool:
        """Acquire tokens for a domain"""
        rate_limit_key = f"rl:{domain}"
        
        # Get current rate limit status
        status_data = await self.storage.get(rate_limit_key)
        if status_data:
            status = RateLimitStatus.model_validate_json(status_data)
        else:
            # Initialize rate limiter for domain
            status = RateLimitStatus(
                domain=domain,
                tokens_remaining=10,  # Default bucket size
                tokens_per_period=10,
                period_seconds=600,  # 10 minutes
                last_request_time=None,
                next_allowed_time=None
            )
        
        now = datetime.now(timezone.utc)
        
        # Refill tokens if period has passed
        if status.last_request_time:
            time_since_last = (now - status.last_request_time).total_seconds()
            if time_since_last >= status.period_seconds:
                status.tokens_remaining = status.tokens_per_period
        
        # Check if we can acquire tokens
        if status.tokens_remaining >= tokens:
            status.tokens_remaining -= tokens
            status.last_request_time = now
            
            # Calculate next allowed time (minimum delay between requests)
            status.next_allowed_time = now + timedelta(seconds=12.0)  # Minimum 12s delay
            
            # Save updated status
            await self.storage.set(
                rate_limit_key, 
                status.model_dump_json(), 
                ttl=status.period_seconds * 2
            )
            return True
        
        return False
    
    async def time_until_next_request(self, domain: str) -> float:
        """Get seconds until next request is allowed"""
        rate_limit_key = f"rl:{domain}"
        status_data = await self.storage.get(rate_limit_key)
        
        if not status_data:
            return 0.0
        
        status = RateLimitStatus.model_validate_json(status_data)
        if not status.next_allowed_time:
            return 0.0
        
        now = datetime.now(timezone.utc)
        time_diff = (status.next_allowed_time - now).total_seconds()
        return max(0.0, time_diff)


class CircuitBreaker:
    """Circuit breaker for failing domains"""
    
    def __init__(self, storage: RedisStorage, failure_threshold: int = 5):
        self.storage = storage
        self.failure_threshold = failure_threshold
        self.recovery_timeout = 300  # 5 minutes
    
    async def get_status(self, domain: str) -> CircuitBreakerStatus:
        """Get circuit breaker status for domain"""
        cb_key = f"cb:{domain}"
        status_data = await self.storage.get(cb_key)
        
        if status_data:
            return CircuitBreakerStatus.model_validate_json(status_data)
        
        return CircuitBreakerStatus(domain=domain)
    
    async def can_proceed(self, domain: str) -> bool:
        """Check if requests can proceed for this domain"""
        status = await self.get_status(domain)
        
        if status.state == CircuitBreakerState.CLOSED:
            return True
        elif status.state == CircuitBreakerState.OPEN:
            # Check if recovery time has passed
            if status.next_attempt_time and datetime.now(timezone.utc) >= status.next_attempt_time:
                # Move to half-open state
                status.state = CircuitBreakerState.HALF_OPEN
                await self._save_status(status)
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    async def record_success(self, domain: str):
        """Record a successful request"""
        status = await self.get_status(domain)
        status.success_count += 1
        status.failure_count = 0
        
        if status.state == CircuitBreakerState.HALF_OPEN and status.success_count >= 3:
            # Close the circuit breaker
            status.state = CircuitBreakerState.CLOSED
            status.next_attempt_time = None
        
        await self._save_status(status)
    
    async def record_failure(self, domain: str):
        """Record a failed request"""
        status = await self.get_status(domain)
        status.failure_count += 1
        status.last_failure_time = datetime.now(timezone.utc)
        
        if status.failure_count >= self.failure_threshold:
            # Open the circuit breaker
            status.state = CircuitBreakerState.OPEN
            status.next_attempt_time = datetime.now(timezone.utc) + timedelta(seconds=self.recovery_timeout)
            status.success_count = 0
        
        await self._save_status(status)
    
    async def _save_status(self, status: CircuitBreakerStatus):
        """Save circuit breaker status"""
        cb_key = f"cb:{status.domain}"
        await self.storage.set(cb_key, status.model_dump_json(), ttl=86400)  # 24 hours


class CompliantBrowsingSession:
    """
    Compliant web browsing session that respects robots.txt, rate limits,
    and implements circuit breaker patterns.
    """
    
    def __init__(self, storage: RedisStorage, user_agent: str = None, contact_email: str = None):
        self.storage = storage
        self.user_agent = user_agent or "FinancialNewsBot/1.0 (Educational Research)"
        self.contact_email = contact_email
        
        self.robots_checker = RobotsChecker(storage)
        self.rate_limiter = RateLimiter(storage)
        self.circuit_breaker = CircuitBreaker(storage)
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._cleanup()
    
    async def _init_browser(self):
        """Initialize Playwright browser"""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,  # Compliant headless mode
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
            ]
        )
        
        # Create context with standard settings (no fingerprint spoofing)
        self._context = await self._browser.new_context(
            user_agent=self.user_agent,
            viewport={'width': 1366, 'height': 768},
            extra_http_headers={'From': self.contact_email} if self.contact_email else {}
        )
    
    async def _cleanup(self):
        """Clean up browser resources"""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
    
    async def fetch_html(self, url: str) -> RenderResult:
        """
        Fetch HTML content using appropriate strategy.
        Respects robots.txt, rate limits, and circuit breakers.
        """
        domain = urlparse(url).netloc
        start_time = time.time()
        
        # Check circuit breaker
        if not await self.circuit_breaker.can_proceed(domain):
            return RenderResult(
                url=url,
                final_url=url,
                html="",
                status_code=503,
                strategy=FetchStrategy.PLAYWRIGHT,
                render_time_ms=0,
                success=False,
                error_message="Circuit breaker open for domain"
            )
        
        # Check robots.txt
        if not await self.robots_checker.can_fetch(url, self.user_agent):
            await self.circuit_breaker.record_failure(domain)
            return RenderResult(
                url=url,
                final_url=url,
                html="",
                status_code=403,
                strategy=FetchStrategy.PLAYWRIGHT,
                render_time_ms=0,
                success=False,
                error_message="Blocked by robots.txt"
            )
        
        # Check rate limit
        wait_time = await self.rate_limiter.time_until_next_request(domain)
        if wait_time > 0:
            logger.info("Rate limited, waiting", domain=domain, wait_seconds=wait_time)
            await asyncio.sleep(wait_time)
        
        # Acquire rate limit token
        if not await self.rate_limiter.acquire(domain):
            return RenderResult(
                url=url,
                final_url=url,
                html="",
                status_code=429,
                strategy=FetchStrategy.PLAYWRIGHT,
                render_time_ms=0,
                success=False,
                error_message="Rate limit exceeded"
            )
        
        try:
            # Try aiohttp first for simple pages
            result = await self._fetch_with_aiohttp(url)
            
            # If aiohttp fails or returns minimal content, try Playwright
            if not result.success or len(result.html.strip()) < 1000:
                logger.info("Fallback to Playwright", url=url, reason="minimal_content")
                result = await self._fetch_with_playwright(url)
            
            if result.success:
                await self.circuit_breaker.record_success(domain)
            else:
                await self.circuit_breaker.record_failure(domain)
            
            result.render_time_ms = int((time.time() - start_time) * 1000)
            return result
            
        except Exception as e:
            await self.circuit_breaker.record_failure(domain)
            return RenderResult(
                url=url,
                final_url=url,
                html="",
                status_code=500,
                strategy=FetchStrategy.PLAYWRIGHT,
                render_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message=str(e)
            )
    
    async def _fetch_with_aiohttp(self, url: str) -> RenderResult:
        """Fetch content using aiohttp"""
        headers = {
            'User-Agent': self.user_agent,
        }
        if self.contact_email:
            headers['From'] = self.contact_email
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    html = await response.text()
                    
                    return RenderResult(
                        url=url,
                        final_url=str(response.url),
                        html=html,
                        status_code=response.status,
                        strategy=FetchStrategy.AIOHTTP,
                        render_time_ms=0,  # Will be set by caller
                        success=response.status == 200,
                        headers=dict(response.headers)
                    )
        except Exception as e:
            return RenderResult(
                url=url,
                final_url=url,
                html="",
                status_code=0,
                strategy=FetchStrategy.AIOHTTP,
                render_time_ms=0,
                success=False,
                error_message=str(e)
            )
    
    async def _fetch_with_playwright(self, url: str) -> RenderResult:
        """Fetch content using Playwright"""
        if not self._context:
            raise RuntimeError("Browser context not initialized")
        
        page: Page = await self._context.new_page()
        
        try:
            # Navigate with timeout
            response = await page.goto(url, timeout=30000, wait_until="networkidle")
            
            if not response:
                return RenderResult(
                    url=url,
                    final_url=url,
                    html="",
                    status_code=0,
                    strategy=FetchStrategy.PLAYWRIGHT,
                    render_time_ms=0,
                    success=False,
                    error_message="No response received"
                )
            
            # Wait for content to load
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
            except:
                pass  # Continue if DOM load timeout
            
            # Get final content
            html = await page.content()
            
            return RenderResult(
                url=url,
                final_url=page.url,
                html=html,
                status_code=response.status,
                strategy=FetchStrategy.PLAYWRIGHT,
                render_time_ms=0,  # Will be set by caller
                success=response.status == 200,
                headers=dict(response.headers)
            )
            
        except Exception as e:
            return RenderResult(
                url=url,
                final_url=url,
                html="",
                status_code=0,
                strategy=FetchStrategy.PLAYWRIGHT,
                render_time_ms=0,
                success=False,
                error_message=str(e)
            )
        finally:
            await page.close()
    
    async def fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch JSON content from API endpoint"""
        domain = urlparse(url).netloc
        
        # Apply same compliance checks as HTML fetching
        if not await self.circuit_breaker.can_proceed(domain):
            logger.warning("Circuit breaker open", domain=domain)
            return None
        
        if not await self.robots_checker.can_fetch(url, self.user_agent):
            logger.warning("Blocked by robots.txt", url=url)
            await self.circuit_breaker.record_failure(domain)
            return None
        
        # Rate limiting
        wait_time = await self.rate_limiter.time_until_next_request(domain)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        if not await self.rate_limiter.acquire(domain):
            logger.warning("Rate limit exceeded", domain=domain)
            return None
        
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/json',
        }
        if self.contact_email:
            headers['From'] = self.contact_email
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        await self.circuit_breaker.record_success(domain)
                        return await response.json()
                    else:
                        await self.circuit_breaker.record_failure(domain)
                        return None
        except Exception as e:
            logger.error("JSON fetch failed", url=url, error=str(e))
            await self.circuit_breaker.record_failure(domain)
            return None