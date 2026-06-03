"""
Multisource Price Data Provider with Fallback Strategy

Production-ready price data fetching with:
- Alpha Vantage (primary) → Yahoo (fallback) → Paid sources (premium)
- Rate limiting and exponential backoff
- Once-daily refresh with CSV caching
- Comprehensive error handling and source provenance tracking
"""

import asyncio
import logging
import time
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from enum import Enum

# Core dependencies
import requests
import yfinance as yf

# Configuration imports
from config.config import ALPHA_VANTAGE_KEY, FINNHUB_TOKEN, POLYGON_KEY, EODHD_KEY, PRICE_DATA_CONFIG, AV_BATCH_SIZE
from config.feature_flags import is_paid_sources_enabled, is_alpha_vantage_batching_enabled

logger = logging.getLogger(__name__)

class DataSource(Enum):
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO = "yahoo"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    EODHD = "eodhd"

@dataclass
class FetchResult:
    """Result of price data fetch operation"""
    success: bool
    data: Optional[pd.DataFrame] = None
    source: Optional[DataSource] = None
    error: Optional[str] = None
    fetch_time: datetime = field(default_factory=datetime.now)
    cache_hit: bool = False
    api_calls_used: int = 0

@dataclass
class RateLimitTracker:
    """Track API rate limits per source"""
    calls_made: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    last_reset: datetime = field(default_factory=datetime.now)
    
    def reset_if_needed(self, window_seconds: int) -> bool:
        """Reset counters if rate limit window has passed"""
        if (datetime.now() - self.window_start).total_seconds() >= window_seconds:
            self.calls_made = 0
            self.window_start = datetime.now()
            self.last_reset = datetime.now()
            return True
        return False
    
    def can_make_request(self, limit: int, window_seconds: int) -> bool:
        """Check if we can make another API request"""
        self.reset_if_needed(window_seconds)
        return self.calls_made < limit

class BasePriceStrategy(ABC):
    """Abstract base class for price data fetching strategies"""
    
    def __init__(self, source: DataSource):
        self.source = source
        self.config = PRICE_DATA_CONFIG[source.value]
        self.rate_tracker = RateLimitTracker()
        
    @abstractmethod
    async def fetch_one(self, symbol: str, lookback_days: int) -> FetchResult:
        """Fetch price data for a single symbol"""
        pass
    
    async def fetch_batch(self, symbols: List[str], lookback_days: int) -> Dict[str, FetchResult]:
        """Fetch price data for multiple symbols"""
        results = {}
        
        # Default implementation: sequential fetching
        for symbol in symbols:
            if self.rate_tracker.can_make_request(
                self.config["rate_limit"], 
                self.config["rate_window"]
            ):
                result = await self.fetch_one(symbol, lookback_days)
                results[symbol] = result
                if result.success:
                    self.rate_tracker.calls_made += result.api_calls_used
                
                # Add delay between requests
                await asyncio.sleep(1.0)
            else:
                results[symbol] = FetchResult(
                    success=False,
                    error=f"Rate limit exceeded for {self.source.value}"
                )
        
        return results
    
    def _exponential_backoff(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay"""
        return min(base_delay * (2 ** attempt), 60.0)  # Max 60 seconds

class AlphaVantageStrategy(BasePriceStrategy):
    """Alpha Vantage price data strategy (primary source)"""
    
    def __init__(self):
        super().__init__(DataSource.ALPHA_VANTAGE)
        self.api_key = ALPHA_VANTAGE_KEY
        
        # BEGIN F02 - Batching and quota integration
        self.batch_size = AV_BATCH_SIZE  # Up to 100 symbols per batch
        # END F02
        
    async def fetch_one(self, symbol: str, lookback_days: int) -> FetchResult:
        """Fetch daily price data from Alpha Vantage"""
        
        if not self.api_key:
            return FetchResult(
                success=False,
                error="Alpha Vantage API key not configured"
            )
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=self.config["timeout"]
                )
                
                if response.status_code == 429:
                    # Rate limited
                    if attempt < max_retries - 1:
                        delay = self._exponential_backoff(attempt)
                        logger.warning(f"Alpha Vantage rate limited, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        return FetchResult(
                            success=False,
                            error="Rate limited after retries",
                            api_calls_used=1
                        )
                
                response.raise_for_status()
                data = response.json()
                
                # Check for API error messages
                if "Error Message" in data:
                    return FetchResult(
                        success=False,
                        error=data["Error Message"],
                        api_calls_used=1
                    )
                
                if "Note" in data:
                    # API call frequency limit
                    return FetchResult(
                        success=False,
                        error="API call frequency limit reached",
                        api_calls_used=1
                    )
                
                # Parse time series data
                time_series_key = "Time Series (Daily)"
                if time_series_key not in data:
                    return FetchResult(
                        success=False,
                        error=f"No time series data found for {symbol}",
                        api_calls_used=1
                    )
                
                # Convert to DataFrame
                ts_data = data[time_series_key]
                df = pd.DataFrame.from_dict(ts_data, orient='index')
                
                # Standardize column names
                df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend', 'split']
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df = df.sort_index(ascending=False)  # Most recent first
                
                # Limit to lookback days
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                df = df[df.index >= cutoff_date]
                
                return FetchResult(
                    success=True,
                    data=df,
                    source=self.source,
                    api_calls_used=1
                )
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    logger.warning(f"Alpha Vantage request failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    return FetchResult(
                        success=False,
                        error=f"Request failed after retries: {str(e)}",
                        api_calls_used=0
                    )
            except Exception as e:
                return FetchResult(
                    success=False,
                    error=f"Unexpected error: {str(e)}",
                    api_calls_used=0
                )
    
    # BEGIN F02 - Enhanced batch fetching with Alpha Vantage batching
    async def fetch_batch(self, symbols: List[str], lookback_days: int) -> Dict[str, FetchResult]:
        """
        Fetch price data for multiple symbols with batching optimization
        Uses Alpha Vantage batch quotes API when batching is enabled
        """
        
        if not is_alpha_vantage_batching_enabled():
            # Use standard sequential fetching
            return await super().fetch_batch(symbols, lookback_days)
        
        logger.info(f"Alpha Vantage batch fetching enabled for {len(symbols)} symbols")
        
        results = {}
        
        # Group symbols into optimal batches
        batch_size = min(self.batch_size, 100)  # Alpha Vantage limit
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        logger.info(f"Processing {len(batches)} batches of up to {batch_size} symbols each")
        
        for batch_num, batch_symbols in enumerate(batches, 1):
            # Check rate limits before each batch
            if not self.rate_tracker.can_make_request(
                self.config["rate_limit"], 
                self.config["rate_window"]
            ):
                logger.warning(f"Rate limit reached, stopping at batch {batch_num}")
                # Return failures for remaining symbols
                for symbol in batch_symbols:
                    results[symbol] = FetchResult(
                        success=False,
                        error="Rate limit exceeded for Alpha Vantage batching"
                    )
                break
            
            # Try batch quote API first (most efficient)
            batch_result = await self._fetch_batch_quotes(batch_symbols)
            
            if batch_result['success']:
                # Batch API succeeded - process all symbols from single call
                results.update(batch_result['results'])
                self.rate_tracker.calls_made += 1
                calls_saved = len(batch_symbols) - 1  # Saved individual calls
                logger.info(f"✅ Batch {batch_num}: {len(batch_symbols)} symbols in 1 call (saved {calls_saved} calls)")
                
            else:
                # Batch API failed - fall back to individual calls
                logger.warning(f"Batch API failed for batch {batch_num}: {batch_result['error']}")
                logger.info(f"Falling back to individual calls for {len(batch_symbols)} symbols")
                
                for symbol in batch_symbols:
                    if self.rate_tracker.can_make_request(
                        self.config["rate_limit"],
                        self.config["rate_window"]
                    ):
                        individual_result = await self.fetch_one(symbol, lookback_days)
                        results[symbol] = individual_result
                        if individual_result.success:
                            self.rate_tracker.calls_made += individual_result.api_calls_used
                        
                        # Add delay between individual requests
                        await asyncio.sleep(1.0)
                    else:
                        results[symbol] = FetchResult(
                            success=False,
                            error="Rate limit exceeded during individual fallback"
                        )
            
            # Add delay between batches
            if batch_num < len(batches):
                await asyncio.sleep(2.0)
        
        successful_fetches = sum(1 for r in results.values() if r.success)
        logger.info(f"Batch operation complete: {successful_fetches}/{len(symbols)} symbols fetched")
        
        return results
    
    async def _fetch_batch_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch quotes for multiple symbols using Alpha Vantage batch API
        
        Args:
            symbols: List of symbols (up to 100)
            
        Returns:
            Dict with success flag, results, and error info
        """
        
        if not self.api_key:
            return {
                'success': False,
                'error': 'Alpha Vantage API key not configured',
                'results': {}
            }
        
        # Alpha Vantage batch quotes endpoint
        url = "https://www.alphavantage.co/query"
        symbol_list = ",".join(symbols[:100])  # Limit to 100 symbols
        
        params = {
            "function": "BATCH_STOCK_QUOTES",
            "symbols": symbol_list,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.config["timeout"]
            )
            
            if response.status_code == 429:
                return {
                    'success': False,
                    'error': 'Rate limited',
                    'results': {}
                }
            
            response.raise_for_status()
            data = response.json()
            
            # Check for API error messages
            if "Error Message" in data:
                return {
                    'success': False,
                    'error': data["Error Message"],
                    'results': {}
                }
            
            if "Note" in data:
                return {
                    'success': False,
                    'error': "API call frequency limit reached",
                    'results': {}
                }
            
            # Parse batch quotes response
            quotes_key = "Stock Quotes"
            if quotes_key not in data:
                return {
                    'success': False,
                    'error': f"No quotes data found in response",
                    'results': {}
                }
            
            # Convert quotes to FetchResult format
            results = {}
            quotes = data[quotes_key]
            
            for quote in quotes:
                symbol = quote.get("1. symbol", "")
                if not symbol:
                    continue
                
                try:
                    # Extract quote data
                    current_price = float(quote.get("2. price", 0))
                    volume = int(quote.get("3. volume", 0))
                    timestamp = quote.get("4. timestamp", "")
                    
                    # Create minimal DataFrame with current quote
                    # Note: Batch quotes don't provide historical data
                    # This is a limitation - we get current price only
                    df = pd.DataFrame({
                        'open': [current_price],
                        'high': [current_price],
                        'low': [current_price],
                        'close': [current_price],
                        'adjusted_close': [current_price],
                        'volume': [volume],
                        'dividend': [0.0],
                        'split': [1.0]
                    }, index=[pd.Timestamp.now()])
                    
                    results[symbol] = FetchResult(
                        success=True,
                        data=df,
                        source=self.source,
                        api_calls_used=0  # Shared call cost
                    )
                    
                except (ValueError, KeyError) as e:
                    results[symbol] = FetchResult(
                        success=False,
                        error=f"Failed to parse quote for {symbol}: {str(e)}",
                        api_calls_used=0
                    )
            
            return {
                'success': True,
                'error': None,
                'results': results
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Request failed: {str(e)}",
                'results': {}
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'results': {}
            }
    # END F02

class YahooStrategy(BasePriceStrategy):
    """Yahoo Finance price data strategy (fallback source)"""
    
    def __init__(self):
        super().__init__(DataSource.YAHOO)
        
    async def fetch_one(self, symbol: str, lookback_days: int) -> FetchResult:
        """Fetch daily price data from Yahoo Finance"""
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                prepost=False
            )
            
            if df.empty:
                return FetchResult(
                    success=False,
                    error=f"No data found for {symbol} on Yahoo Finance"
                )
            
            # Standardize column names to match Alpha Vantage
            df.columns = df.columns.str.lower()
            if 'adj close' in df.columns:
                df['adjusted_close'] = df['adj close']
                df.drop('adj close', axis=1, inplace=True)
            
            # Add missing columns that Alpha Vantage provides
            df['dividend'] = 0.0  # Not directly available in yfinance history
            df['split'] = 1.0    # Not directly available in yfinance history
            
            # Sort by date (most recent first)
            df = df.sort_index(ascending=False)
            
            return FetchResult(
                success=True,
                data=df,
                source=self.source,
                api_calls_used=1
            )
            
        except Exception as e:
            return FetchResult(
                success=False,
                error=f"Yahoo Finance error: {str(e)}",
                api_calls_used=0
            )

class FinnhubStrategy(BasePriceStrategy):
    """Finnhub price data strategy (premium source)"""
    
    def __init__(self):
        super().__init__(DataSource.FINNHUB)
        self.api_key = FINNHUB_TOKEN
        
    async def fetch_one(self, symbol: str, lookback_days: int) -> FetchResult:
        """Fetch daily price data from Finnhub"""
        
        if not self.api_key:
            return FetchResult(
                success=False,
                error="Finnhub API key not configured"
            )
        
        # Calculate Unix timestamps
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp())
        
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": "D",  # Daily
            "from": start_time,
            "to": end_time,
            "token": self.api_key
        }
        
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.config["timeout"]
            )
            
            if response.status_code == 429:
                return FetchResult(
                    success=False,
                    error="Finnhub rate limit exceeded",
                    api_calls_used=1
                )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("s") != "ok":
                return FetchResult(
                    success=False,
                    error=f"Finnhub API error for {symbol}: {data.get('s', 'unknown')}",
                    api_calls_used=1
                )
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            
            # Convert timestamps to datetime index
            df.index = pd.to_datetime(data['t'], unit='s')
            
            # Add missing columns to match Alpha Vantage format
            df['adjusted_close'] = df['close']  # Finnhub doesn't provide adjusted close
            df['dividend'] = 0.0
            df['split'] = 1.0
            
            # Sort by date (most recent first)
            df = df.sort_index(ascending=False)
            
            return FetchResult(
                success=True,
                data=df,
                source=self.source,
                api_calls_used=1
            )
            
        except Exception as e:
            return FetchResult(
                success=False,
                error=f"Finnhub error: {str(e)}",
                api_calls_used=0
            )

class PolygonStrategy(BasePriceStrategy):
    """Polygon.io price data strategy (premium source)"""
    
    def __init__(self):
        super().__init__(DataSource.POLYGON)
        self.api_key = POLYGON_KEY
        
    async def fetch_one(self, symbol: str, lookback_days: int) -> FetchResult:
        """Fetch daily price data from Polygon"""
        
        if not self.api_key:
            return FetchResult(
                success=False,
                error="Polygon API key not configured"
            )
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "apikey": self.api_key,
            "adjusted": "true",
            "sort": "desc"
        }
        
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.config["timeout"]
            )
            
            if response.status_code == 429:
                return FetchResult(
                    success=False,
                    error="Polygon rate limit exceeded",
                    api_calls_used=1
                )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "OK":
                return FetchResult(
                    success=False,
                    error=f"Polygon API error for {symbol}: {data.get('status', 'unknown')}",
                    api_calls_used=1
                )
            
            if not data.get("results"):
                return FetchResult(
                    success=False,
                    error=f"No data found for {symbol} on Polygon",
                    api_calls_used=1
                )
            
            # Convert to DataFrame
            results = data["results"]
            df = pd.DataFrame([{
                'open': r['o'],
                'high': r['h'],
                'low': r['l'],
                'close': r['c'],
                'volume': r['v'],
                'timestamp': r['t']
            } for r in results])
            
            # Convert timestamp to datetime index
            df.index = pd.to_datetime(df['timestamp'], unit='ms')
            df.drop('timestamp', axis=1, inplace=True)
            
            # Add missing columns
            df['adjusted_close'] = df['close']  # Polygon provides adjusted data by default
            df['dividend'] = 0.0
            df['split'] = 1.0
            
            # Sort by date (most recent first)
            df = df.sort_index(ascending=False)
            
            return FetchResult(
                success=True,
                data=df,
                source=self.source,
                api_calls_used=1
            )
            
        except Exception as e:
            return FetchResult(
                success=False,
                error=f"Polygon error: {str(e)}",
                api_calls_used=0
            )

class EodhdStrategy(BasePriceStrategy):
    """EODHD price data strategy (premium source)"""
    
    def __init__(self):
        super().__init__(DataSource.EODHD)
        self.api_key = EODHD_KEY
        
    async def fetch_one(self, symbol: str, lookback_days: int) -> FetchResult:
        """Fetch daily price data from EODHD"""
        
        if not self.api_key:
            return FetchResult(
                success=False,
                error="EODHD API key not configured"
            )
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        url = f"https://eodhd.com/api/eod/{symbol}.US"
        params = {
            "api_token": self.api_key,
            "from": start_date,
            "to": end_date,
            "period": "d",
            "fmt": "json"
        }
        
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.config["timeout"]
            )
            
            if response.status_code == 429:
                return FetchResult(
                    success=False,
                    error="EODHD rate limit exceeded",
                    api_calls_used=1
                )
            
            response.raise_for_status()
            data = response.json()
            
            if not data or not isinstance(data, list):
                return FetchResult(
                    success=False,
                    error=f"No data found for {symbol} on EODHD",
                    api_calls_used=1
                )
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Set datetime index
            df.index = pd.to_datetime(df['date'])
            df.drop('date', axis=1, inplace=True)
            
            # Standardize column names
            df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
            
            # Add missing columns
            df['dividend'] = 0.0
            df['split'] = 1.0
            
            # Sort by date (most recent first)
            df = df.sort_index(ascending=False)
            
            return FetchResult(
                success=True,
                data=df,
                source=self.source,
                api_calls_used=1
            )
            
        except Exception as e:
            return FetchResult(
                success=False,
                error=f"EODHD error: {str(e)}",
                api_calls_used=0
            )

class PriceProvider:
    """
    Unified price data provider with fallback strategy
    
    Primary: Alpha Vantage → Fallback: Yahoo Finance → Premium: Finnhub/Polygon/EODHD
    """
    
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir) / "av_bulk_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize strategies in fallback order
        self.strategies = [
            AlphaVantageStrategy(),
            YahooStrategy()
        ]
        
        # Add premium sources if enabled and configured
        if is_paid_sources_enabled():
            premium_strategies = [
                FinnhubStrategy(),
                PolygonStrategy(), 
                EodhdStrategy()
            ]
            # Insert premium sources after Alpha Vantage but before Yahoo
            self.strategies = [self.strategies[0]] + premium_strategies + [self.strategies[1]]
        
        # BEGIN F02 - Smart Cache Manager Integration
        self.smart_cache = None
        if is_alpha_vantage_batching_enabled():
            from ..smart_cache_manager import SmartCacheManager
            self.smart_cache = SmartCacheManager(str(self.cache_dir))
            logger.info("F02: SmartCacheManager integrated with PriceProvider")
        # END F02
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'source_success': {source.value: 0 for source in DataSource},
            'source_failures': {source.value: 0 for source in DataSource},
            'api_calls_used': {source.value: 0 for source in DataSource}
        }
        
        logger.info(f"PriceProvider initialized with {len(self.strategies)} strategies")
    
    def _get_cache_path(self, symbol: str, lookback_days: int) -> Path:
        """Get cache file path for symbol"""
        return self.cache_dir / f"{symbol}_{lookback_days}d.csv"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid (once-daily refresh)"""
        if not cache_path.exists():
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age.total_seconds() < (max_age_hours * 3600)
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load price data from cache file"""
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path, source: DataSource) -> bool:
        """Save price data to cache with source provenance"""
        try:
            # Add metadata as comments in CSV
            with open(cache_path, 'w') as f:
                f.write(f"# Source: {source.value}\n")
                f.write(f"# Cached: {datetime.now().isoformat()}\n")
                f.write(f"# Records: {len(data)}\n")
                data.to_csv(f, index=True)
            
            logger.info(f"Cached {len(data)} records from {source.value} to {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache {cache_path}: {e}")
            return False
    
    def _calculate_input_hash(self, symbols: List[str], lookback_days: int) -> str:
        """Calculate deterministic hash of inputs for logging"""
        input_data = {
            'symbols': sorted(symbols),
            'lookback_days': lookback_days,
            'timestamp': datetime.now().strftime('%Y-%m-%d')  # Daily idempotency
        }
        input_str = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()[:8]
    
    async def get_history(self, symbols: List[str], lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for multiple symbols with fallback strategy
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days of historical data
            
        Returns:
            Dict mapping symbols to price DataFrames
        """
        
        input_hash = self._calculate_input_hash(symbols, lookback_days)
        logger.info(f"Fetching price data for {len(symbols)} symbols, {lookback_days} days (hash: {input_hash})")
        
        # BEGIN F02 - Quota-aware batch planning
        if self.smart_cache and is_alpha_vantage_batching_enabled():
            return await self._get_history_with_batch_planning(symbols, lookback_days, input_hash)
        # END F02
        
        # Original implementation for backward compatibility
        return await self._get_history_fallback_strategy(symbols, lookback_days, input_hash)
    
    async def _get_history_with_batch_planning(self, symbols: List[str], lookback_days: int, input_hash: str) -> Dict[str, pd.DataFrame]:
        """F02: Get history with quota-aware batch planning"""
        logger.info(f"F02: Using quota-aware batch planning for {len(symbols)} symbols")
        
        # Plan optimal batch requests
        batch_plan = self.smart_cache.plan_batch_requests(symbols, lookback_days)
        
        logger.info(f"F02: Batch plan - {batch_plan['symbols']['cached']} cached, {batch_plan['symbols']['missing']} missing, {batch_plan['batches']['calls_needed']} calls needed")
        
        results = {}
        
        # Add cached symbols to results  
        for symbol in batch_plan['symbols']['cached_symbols']:
            cache_path = self._get_cache_path(symbol, lookback_days)
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                results[symbol] = cached_data
                self.stats['cache_hits'] += 1
        
        # Process missing symbols with batch optimization
        missing_symbols = batch_plan['symbols']['missing_symbols']
        if missing_symbols:
            # Execute batch plan
            execution_result = self.smart_cache.execute_batch_plan(batch_plan)
            
            if execution_result['fallback_used']:
                logger.warning("F02: Quota insufficient - falling back to non-Alpha Vantage strategies")
                # Use fallback strategies (skip Alpha Vantage)
                fallback_strategies = [s for s in self.strategies if s.source != DataSource.ALPHA_VANTAGE]
                await self._fetch_with_strategies(missing_symbols, lookback_days, fallback_strategies, results)
                
            elif batch_plan['quota']['can_proceed']:
                # Use Alpha Vantage with batch optimization
                alpha_vantage_strategy = next((s for s in self.strategies if s.source == DataSource.ALPHA_VANTAGE), None)
                
                if alpha_vantage_strategy and hasattr(alpha_vantage_strategy, 'fetch_batch'):
                    logger.info(f"F02: Using Alpha Vantage batch fetching for {len(missing_symbols)} symbols")
                    
                    # Use batch fetching
                    batch_results = await alpha_vantage_strategy.fetch_batch(missing_symbols, lookback_days)
                    
                    # Process batch results
                    for symbol, fetch_result in batch_results.items():
                        self.stats['api_calls_used'][alpha_vantage_strategy.source.value] += fetch_result.api_calls_used
                        
                        if fetch_result.success and fetch_result.data is not None:
                            cache_path = self._get_cache_path(symbol, lookback_days)
                            if self._save_to_cache(fetch_result.data, cache_path, alpha_vantage_strategy.source):
                                results[symbol] = fetch_result.data
                                self.stats['source_success'][alpha_vantage_strategy.source.value] += 1
                                logger.info(f"✅ {symbol} fetched via F02 batch from {alpha_vantage_strategy.source.value}")
                        else:
                            self.stats['source_failures'][alpha_vantage_strategy.source.value] += 1
                            logger.warning(f"❌ {alpha_vantage_strategy.source.value} batch failed for {symbol}: {fetch_result.error}")
                    
                    # Record successful quota consumption
                    if execution_result['calls_made'] > 0:
                        self.smart_cache.quota_ledger.consume_calls(
                            execution_result['calls_made'],
                            execution_result['calls_saved']
                        )
                    
                    # Handle any symbols that failed batch fetching with fallback
                    failed_symbols = [s for s in missing_symbols if s not in results]
                    if failed_symbols:
                        logger.info(f"F02: {len(failed_symbols)} symbols failed batch fetch, using fallback strategies")
                        fallback_strategies = [s for s in self.strategies if s.source != DataSource.ALPHA_VANTAGE]
                        await self._fetch_with_strategies(failed_symbols, lookback_days, fallback_strategies, results)
                else:
                    # No batch support - fallback to individual calls
                    logger.info("F02: No batch support found, falling back to individual calls")
                    await self._fetch_with_strategies(missing_symbols, lookback_days, self.strategies, results)
            else:
                # Quota insufficient
                logger.error(f"F02: Cannot proceed - {batch_plan['quota']['reason']}")
                self.smart_cache.quota_ledger.trigger_fallback(batch_plan['quota']['reason'])
                
                # Use non-Alpha Vantage strategies
                fallback_strategies = [s for s in self.strategies if s.source != DataSource.ALPHA_VANTAGE]
                await self._fetch_with_strategies(missing_symbols, lookback_days, fallback_strategies, results)
        
        # Log F02 performance summary
        self._log_f02_performance_summary(symbols, results, batch_plan)
        
        return results
    
    async def _get_history_fallback_strategy(self, symbols: List[str], lookback_days: int, input_hash: str) -> Dict[str, pd.DataFrame]:
        """Original implementation with fallback strategy (backward compatibility)"""
        results = {}
        symbols_to_fetch = []
        
        # Check cache first
        for symbol in symbols:
            cache_path = self._get_cache_path(symbol, lookback_days)
            
            if self._is_cache_valid(cache_path):
                cached_data = self._load_from_cache(cache_path)
                if cached_data is not None:
                    results[symbol] = cached_data
                    self.stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for {symbol}")
                    continue
            
            symbols_to_fetch.append(symbol)
        
        # Fetch missing data using fallback strategy
        if symbols_to_fetch:
            await self._fetch_with_strategies(symbols_to_fetch, lookback_days, self.strategies, results)
        
        # Log performance summary
        self._log_performance_summary(symbols, results, symbols_to_fetch)
        
        return results
    
    async def _fetch_with_strategies(self, symbols_to_fetch: List[str], lookback_days: int, strategies: List, results: Dict[str, pd.DataFrame]):
        """Common logic for fetching symbols using a list of strategies"""
        logger.info(f"Fetching {len(symbols_to_fetch)} symbols from APIs: {symbols_to_fetch}")
        
        for symbol in symbols_to_fetch:
            self.stats['total_requests'] += 1
            symbol_result = None
            
            # Try each strategy in order
            for strategy in strategies:
                try:
                    logger.debug(f"Trying {strategy.source.value} for {symbol}")
                    fetch_result = await strategy.fetch_one(symbol, lookback_days)
                    
                    # Track API usage
                    self.stats['api_calls_used'][strategy.source.value] += fetch_result.api_calls_used
                    
                    if fetch_result.success and fetch_result.data is not None:
                        # Success - save to cache and use result
                        cache_path = self._get_cache_path(symbol, lookback_days)
                        if self._save_to_cache(fetch_result.data, cache_path, strategy.source):
                            symbol_result = fetch_result.data
                            self.stats['source_success'][strategy.source.value] += 1
                            logger.info(f"✅ {symbol} fetched from {strategy.source.value}")
                            break
                    else:
                        # Strategy failed - log and try next
                        self.stats['source_failures'][strategy.source.value] += 1
                        logger.warning(f"❌ {strategy.source.value} failed for {symbol}: {fetch_result.error}")
                        
                except Exception as e:
                    logger.error(f"Strategy {strategy.source.value} crashed for {symbol}: {e}")
                    self.stats['source_failures'][strategy.source.value] += 1
                    continue
            
            if symbol_result is not None:
                results[symbol] = symbol_result
            else:
                logger.error(f"🚨 All strategies failed for {symbol}")
    
    def _log_f02_performance_summary(self, symbols: List[str], results: Dict[str, pd.DataFrame], batch_plan: Dict[str, Any]):
        """Log F02-specific performance metrics"""
        successful_symbols = len(results)
        failed_symbols = len(symbols) - successful_symbols
        
        logger.info(f"F02: Fetch complete: {successful_symbols}/{len(symbols)} symbols successful")
        if failed_symbols > 0:
            logger.warning(f"F02: Failed to fetch {failed_symbols} symbols")
        
        # Log batch optimization metrics
        if batch_plan['batches']['calls_saved'] > 0:
            logger.info(f"F02: Batch optimization saved {batch_plan['batches']['calls_saved']} API calls")
        
        # Log cache efficiency 
        cache_hits = batch_plan['symbols']['cached']
        total_symbols = len(symbols)
        if total_symbols > 0:
            cache_hit_rate = (cache_hits / total_symbols) * 100
            logger.info(f"F02: Cache hit rate: {cache_hit_rate:.1f}%")
    
    def _log_performance_summary(self, symbols: List[str], results: Dict[str, pd.DataFrame], symbols_to_fetch: List[str]):
        """Log standard performance metrics"""
        successful_symbols = len(results)
        failed_symbols = len(symbols) - successful_symbols
        
        logger.info(f"Fetch complete: {successful_symbols}/{len(symbols)} symbols successful")
        if failed_symbols > 0:
            logger.warning(f"Failed to fetch {failed_symbols} symbols")
        
        # Log cache efficiency
        if self.stats['total_requests'] > 0 and len(symbols_to_fetch) > 0:
            cache_hit_rate = (self.stats['cache_hits'] / (self.stats['cache_hits'] + len(symbols_to_fetch))) * 100
            logger.info(f"Cache hit rate: {cache_hit_rate:.1f}%")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance and usage statistics"""
        stats = {
            'summary': {
                'total_requests': self.stats['total_requests'],
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': (self.stats['cache_hits'] / max(self.stats['total_requests'], 1)) * 100
            },
            'source_performance': {
                'success_counts': self.stats['source_success'].copy(),
                'failure_counts': self.stats['source_failures'].copy(),
                'api_calls_used': self.stats['api_calls_used'].copy()
            },
            'cache_info': {
                'cache_dir': str(self.cache_dir),
                'cached_files': len(list(self.cache_dir.glob("*.csv")))
            }
        }
        
        # BEGIN F02 - Include SmartCacheManager metrics
        if self.smart_cache and is_alpha_vantage_batching_enabled():
            f02_metrics = self.smart_cache.get_performance_metrics()
            stats['f02_quota_management'] = f02_metrics
        # END F02
        
        return stats
    
    def clear_cache(self, older_than_hours: int = 24) -> int:
        """Clear old cache files"""
        cleared_count = 0
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        for cache_file in self.cache_dir.glob("*.csv"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old cache files")
        
        return cleared_count

# Convenience function for easy import
async def fetch_price_data(symbols: List[str], lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """Convenience function to fetch price data"""
    provider = PriceProvider()
    return await provider.get_history(symbols, lookback_days)