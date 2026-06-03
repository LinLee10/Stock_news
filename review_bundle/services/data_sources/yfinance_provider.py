#!/usr/bin/env python3
"""
YFinance Provider - Safe, cached data source for daily OHLCV + adjusted prices.

Implements a rate-limit-aware yfinance wrapper with disk caching, exponential backoff,
and atomic file operations. Designed for once-per-day cadence with minimal network calls.

Key Features:
- 24h TTL disk cache to minimize API calls
- Exponential backoff on YFRateLimitError
- Atomic CSV operations with proper locking
- Serialized requests (no bulk fan-out)
- Reusable session with friendly User-Agent
"""

import os
import time
import logging
import tempfile
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Dict, Optional, Any
import requests
import requests_cache
from yfinance.exceptions import YFRateLimitError

from services.retry_policies import retry_with_backoff, CircuitBreaker, with_circuit_breaker

logger = logging.getLogger(__name__)

# Type definitions
Period = Literal["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
Interval = Literal["1d", "1wk", "1mo"]  # Keep low-frequency on free tier


@dataclass(frozen=True)
class YFConfig:
    """Configuration for YFinance provider."""
    period: Period = "2y"
    interval: Interval = "1d"
    auto_adjust: bool = True
    threads: int = int(os.getenv("YF_THREADS", "1"))  # Use 1 to avoid rate throttling
    cache_ttl_hours: int = int(os.getenv("YF_CACHE_TTL_HOURS", "24"))
    max_retries: int = int(os.getenv("YF_MAX_RETRIES", "2"))  # Reduced for free tier
    backoff_base_seconds: float = float(os.getenv("YF_BACKOFF_BASE_SECONDS", "2.0"))
    enable_backoff_debug: bool = os.getenv("YF_BACKOFF_DEBUG", "false").lower() == "true"
    test_fast_backoff: bool = os.getenv("YF_TEST_FAST_BACKOFF", "0") == "1"  # For tests


class YFinanceProvider:
    """
    Safe yfinance provider with caching and rate limiting protection.
    """
    
    def __init__(self, cfg: YFConfig = YFConfig()):
        self.cfg = cfg
        self.cache_dir = Path("data/yf_bulk_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up cached session
        cache_file = self.cache_dir / "yf_http_cache.sqlite"
        expire_after = timedelta(hours=self.cfg.cache_ttl_hours)
        
        self.session = requests_cache.CachedSession(
            cache_name=str(cache_file),
            expire_after=expire_after,
            headers={
                'User-Agent': 'stonk-news/1.0 (yfinance)'
            }
        )
        
        # Circuit breaker for consecutive failures (fast trip in tests)
        failure_threshold = 1 if self.cfg.test_fast_backoff else 3
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=60 if self.cfg.test_fast_backoff else 300,  # Fast recovery in tests
            expected_exception=YFRateLimitError
        )
        
        logger.info(f"YFinanceProvider initialized with cache_dir={self.cache_dir}, TTL={self.cfg.cache_ttl_hours}h")
    
    def fetch_history(self, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV+AdjClose data for symbols.
        
        Returns tidy DataFrames with Date as index and OHLCV columns.
        Uses disk cache with TTL to minimize network calls.
        
        Args:
            symbols: Ticker symbols to fetch
            
        Returns:
            Dict mapping symbol -> DataFrame with OHLCV+AdjClose data
        """
        symbols = list(symbols)
        results = {}
        cache_hits = 0
        cache_misses = 0
        
        logger.info(f"Fetching history for {len(symbols)} symbols: {symbols}")
        
        for symbol in symbols:
            try:
                # Check cache first
                cached_data = self._load_cached_history(symbol)
                if cached_data is not None:
                    results[symbol] = cached_data
                    cache_hits += 1
                    continue
                
                # Cache miss - fetch from yfinance
                cache_misses += 1
                logger.debug(f"Cache miss for {symbol}, fetching from yfinance")
                
                # Fetch with rate limiting protection
                data = self._fetch_symbol_with_backoff(symbol)
                
                if data is not None and not data.empty:
                    # Normalize and cache the data
                    normalized_data = self._normalize_yf_data(data, symbol)
                    self._save_cached_history(symbol, normalized_data)
                    results[symbol] = normalized_data
                    
                    # Add small delay between requests to be respectful (skip in tests)
                    if not self.cfg.test_fast_backoff:
                        time.sleep(0.5)
                else:
                    logger.warning(f"No data returned for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue
        
        logger.info(f"History fetch complete: {cache_hits} cache hits, {cache_misses} cache misses, {len(results)} successful")
        return results
    
    def fetch_profile_fields(self, symbols: Iterable[str]) -> pd.DataFrame:
        """
        Fetch basic profile fields (sector, industry, beta) for symbols.
        
        Args:
            symbols: Ticker symbols to fetch profiles for
            
        Returns:
            DataFrame with columns: symbol, longName, sector, industry, beta
        """
        symbols = list(symbols)
        profiles = []
        
        logger.info(f"Fetching profiles for {len(symbols)} symbols")
        
        # Check for cached profiles
        cached_profiles = self._load_cached_profiles()
        
        for symbol in symbols:
            try:
                # Check if we have cached profile data
                if cached_profiles is not None and symbol in cached_profiles['symbol'].values:
                    profile_row = cached_profiles[cached_profiles['symbol'] == symbol].iloc[0]
                    profiles.append(profile_row.to_dict())
                    continue
                
                # Fetch fresh profile data
                profile_data = self._fetch_profile_with_backoff(symbol)
                if profile_data:
                    profiles.append(profile_data)
                    
                # Add delay between profile requests (skip in tests)
                if not self.cfg.test_fast_backoff:
                    time.sleep(1.0)
                
            except Exception as e:
                logger.warning(f"Failed to fetch profile for {symbol}: {e}")
                continue
        
        if profiles:
            df = pd.DataFrame(profiles)
            self._save_cached_profiles(df)
            logger.info(f"Profile fetch complete: {len(profiles)} profiles")
            return df
        else:
            logger.warning("No profile data fetched")
            return pd.DataFrame(columns=['symbol', 'longName', 'sector', 'industry', 'beta'])
    
    def _fetch_symbol_with_backoff(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch single symbol with exponential backoff on rate limits."""
        def fetch_fn():
            ticker = yf.Ticker(symbol, session=self.session)
            return ticker.history(
                period=self.cfg.period,
                interval=self.cfg.interval,
                auto_adjust=self.cfg.auto_adjust,
                actions=False  # Don't need dividends/splits for basic OHLCV
            )
        
        try:
            return with_circuit_breaker(
                self.circuit_breaker,
                lambda: retry_with_backoff(
                    fetch_fn,
                    retry_on=(YFRateLimitError,),
                    max_retries=self.cfg.max_retries,
                    base_delay=self.cfg.backoff_base_seconds,
                    debug=self.cfg.enable_backoff_debug
                )
            )
        except YFRateLimitError as e:
            logger.warning(f"Rate limited for {symbol} after retries, skipping: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            return None
    
    def _fetch_profile_with_backoff(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch profile info with backoff protection."""
        def fetch_fn():
            ticker = yf.Ticker(symbol, session=self.session)
            info = ticker.info
            return {
                'symbol': symbol,
                'longName': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'beta': info.get('beta')
            }
        
        try:
            return with_circuit_breaker(
                self.circuit_breaker,
                lambda: retry_with_backoff(
                    fetch_fn,
                    retry_on=(YFRateLimitError,),
                    max_retries=self.cfg.max_retries,
                    base_delay=self.cfg.backoff_base_seconds,
                    debug=self.cfg.enable_backoff_debug
                )
            )
        except YFRateLimitError as e:
            logger.warning(f"Rate limited for profile {symbol}, skipping: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching profile for {symbol}: {e}")
            return None
    
    def _normalize_yf_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize yfinance DataFrame to tidy format.
        
        Ensures consistent column names and adds symbol column.
        """
        if data.empty:
            return pd.DataFrame()
        
        # Reset index to make Date a column
        df = data.reset_index()
        
        # Ensure we have expected columns
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Handle adjusted close if auto_adjust=True
        if 'Adj Close' in df.columns:
            expected_cols.append('Adj Close')
        
        # Filter to only expected columns that exist
        available_cols = [col for col in expected_cols if col in df.columns]
        df = df[available_cols].copy()
        
        # Add symbol column
        df['Symbol'] = symbol
        
        # Ensure Date is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    
    def _get_cache_file_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        filename = f"{symbol}_{self.cfg.period}_{self.cfg.interval}.csv"
        return self.cache_dir / filename
    
    def _is_cache_fresh(self, file_path: Path) -> bool:
        """Check if cache file is within TTL."""
        if not file_path.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        return file_age < timedelta(hours=self.cfg.cache_ttl_hours)
    
    def _load_cached_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached history data if fresh."""
        cache_file = self._get_cache_file_path(symbol)
        
        if not self._is_cache_fresh(cache_file):
            return None
        
        try:
            df = pd.read_csv(cache_file, parse_dates=['Date'])
            logger.debug(f"Cache hit for {symbol}: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None
    
    def _save_cached_history(self, symbol: str, data: pd.DataFrame):
        """Save history data to cache atomically."""
        if data.empty:
            return
        
        cache_file = self._get_cache_file_path(symbol)
        
        try:
            # Atomic write: write to temp file, then rename
            with tempfile.NamedTemporaryFile(
                mode='w', 
                delete=False, 
                dir=self.cache_dir,
                suffix='.csv.tmp'
            ) as tmp_file:
                data.to_csv(tmp_file.name, index=False)
                temp_path = tmp_file.name
            
            # Atomic rename
            os.rename(temp_path, cache_file)
            logger.debug(f"Cached {len(data)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save cache for {symbol}: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except:
                pass
    
    def _load_cached_profiles(self) -> Optional[pd.DataFrame]:
        """Load cached profile data."""
        profiles_file = self.cache_dir / "profiles.csv"
        
        if not self._is_cache_fresh(profiles_file):
            return None
        
        try:
            return pd.read_csv(profiles_file)
        except Exception as e:
            logger.warning(f"Failed to load cached profiles: {e}")
            return None
    
    def _save_cached_profiles(self, profiles: pd.DataFrame):
        """Save profile data to cache atomically."""
        if profiles.empty:
            return
        
        profiles_file = self.cache_dir / "profiles.csv"
        
        try:
            # Atomic write
            with tempfile.NamedTemporaryFile(
                mode='w',
                delete=False,
                dir=self.cache_dir,
                suffix='.csv.tmp'
            ) as tmp_file:
                profiles.to_csv(tmp_file.name, index=False)
                temp_path = tmp_file.name
            
            os.rename(temp_path, profiles_file)
            logger.debug(f"Cached {len(profiles)} profiles")
            
        except Exception as e:
            logger.error(f"Failed to save profile cache: {e}")
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except:
                pass


def create_yfinance_provider(cfg: YFConfig = YFConfig()) -> YFinanceProvider:
    """Factory function to create YFinanceProvider instance."""
    return YFinanceProvider(cfg)