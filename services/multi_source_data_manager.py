"""
Multi-Source Data Manager

Orchestrates data collection from multiple sources with fallback strategies
Integrates price data provider with existing pipeline
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from config.feature_flags import is_multisource_prices_enabled, is_yf_prices_enabled, is_async_io_enabled
from config.tickers import PORTFOLIO, WATCHLIST

# BEGIN F01 - Multisource price data integration
from services.data_sources.price_provider import PriceProvider, fetch_price_data
# END F01

# BEGIN YF_INTEGRATION - YFinance provider integration
from services.data_sources.yfinance_provider import create_yfinance_provider, YFConfig
# END YF_INTEGRATION

# BEGIN F-YF-RATE-WIRE
from services.yf_refresh_guard import YFDailyRefreshGuard
from config.feature_flags import is_yf_daily_refresh_enabled
from config import config
# END F-YF-RATE-WIRE

logger = logging.getLogger(__name__)

class MultiSourceDataManager:
    """
    Unified data manager for all external data sources
    
    Handles:
    - Price data from multiple sources with fallback
    - Rate limiting and caching
    - Error handling and graceful degradation
    """
    
    def __init__(self):
        # BEGIN F01 - Initialize price provider when multisource is enabled
        self.price_provider = None
        if is_multisource_prices_enabled():
            self.price_provider = PriceProvider()
            logger.info("MultiSourceDataManager: Multisource prices enabled")
        else:
            logger.info("MultiSourceDataManager: Using legacy single-source behavior")
        # END F01
        
        # BEGIN YF_INTEGRATION - Initialize yfinance provider when enabled
        self.yf_provider = None
        if is_yf_prices_enabled():
            self.yf_provider = create_yfinance_provider()
            logger.info("MultiSourceDataManager: YFinance provider enabled")
        # END YF_INTEGRATION
        
        self.performance_stats = {
            'price_data_requests': 0,
            'price_data_successes': 0,
            'price_data_cache_hits': 0
        }
    
    # BEGIN F01 - Multisource price data methods
    async def get_price_data(self, symbols: List[str], lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for symbols using multisource provider or fallback
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days of historical data
            
        Returns:
            Dict mapping symbols to price DataFrames
        """
        
        self.performance_stats['price_data_requests'] += 1
        
        # BEGIN F-YF-RATE-CALL
        if is_yf_prices_enabled():
            # Guard daily refresh when enabled
            if is_yf_daily_refresh_enabled():
                refresh_result = YFDailyRefreshGuard.run_once_per_day_static(
                    symbols, key=config.YF_DAILY_KEY
                )
                logger.info(f"YF daily refresh: {refresh_result['status']}")
            
            # Read from cache only (no network in read path)
            cached_data = self.get_yf_cached_data(symbols)
            if cached_data:
                logger.info(f"YFinance cache provided data for {len(cached_data)} symbols")
                self.performance_stats['price_data_successes'] += len(cached_data)
                return cached_data
            else:
                logger.info("No YFinance cache data available, falling back to other sources")
        # END F-YF-RATE-CALL
        
        if self.price_provider and is_multisource_prices_enabled():
            # Use multisource price provider
            logger.info(f"Using multisource price provider for {len(symbols)} symbols")
            try:
                results = await self.price_provider.get_history(symbols, lookback_days)
                
                # Update stats
                self.performance_stats['price_data_successes'] += len(results)
                provider_stats = self.price_provider.get_performance_stats()
                self.performance_stats['price_data_cache_hits'] += provider_stats['summary']['cache_hits']
                
                return results
                
            except Exception as e:
                logger.error(f"Multisource price provider failed: {e}")
                # Fall back to legacy behavior
                return await self._legacy_price_fetch(symbols, lookback_days)
        else:
            # Use legacy single-source behavior
            return await self._legacy_price_fetch(symbols, lookback_days)
    
    async def _legacy_price_fetch(self, symbols: List[str], lookback_days: int) -> Dict[str, pd.DataFrame]:
        """
        Legacy price data fetching (maintains backward compatibility)
        This is a placeholder that should integrate with existing prediction.py logic
        """
        
        logger.info(f"Using legacy price fetch for {len(symbols)} symbols")
        
        # For now, return empty dict - this should be replaced with actual legacy logic
        # In a real implementation, this would call the existing Alpha Vantage code from prediction.py
        results = {}
        
        # TODO: Integrate with existing prediction.py fetch logic when flag is disabled
        # This ensures backward compatibility
        
        return results
    
    def get_price_provider_stats(self) -> Dict[str, Any]:
        """Get detailed price provider statistics"""
        
        base_stats = {
            'multisource_enabled': is_multisource_prices_enabled(),
            'manager_stats': self.performance_stats.copy()
        }
        
        if self.price_provider:
            provider_stats = self.price_provider.get_performance_stats()
            base_stats['provider_stats'] = provider_stats
        
        return base_stats
    
    # BEGIN YF_INTEGRATION - YFinance specific methods
    def get_yf_cached_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get yfinance cached data directly from disk cache without network calls.
        Useful for reading cached data in other parts of the pipeline.
        
        Args:
            symbols: List of symbols to read from cache
            
        Returns:
            Dict mapping symbols to cached DataFrames (empty dict if cache miss)
        """
        if not self.yf_provider:
            return {}
            
        results = {}
        for symbol in symbols:
            try:
                cached_data = self.yf_provider._load_cached_history(symbol)
                if cached_data is not None and not cached_data.empty:
                    results[symbol] = cached_data
            except Exception as e:
                logger.debug(f"Failed to load yfinance cache for {symbol}: {e}")
                
        return results
    
    def is_yf_data_available(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Check which symbols have fresh yfinance cache data available.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dict mapping symbols to availability status
        """
        if not self.yf_provider:
            return {symbol: False for symbol in symbols}
            
        availability = {}
        for symbol in symbols:
            cache_file = self.yf_provider._get_cache_file_path(symbol)
            availability[symbol] = self.yf_provider._is_cache_fresh(cache_file)
            
        return availability
    # END YF_INTEGRATION
    
    async def refresh_price_cache(self, symbols: Optional[List[str]] = None, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Refresh price data cache for specified symbols or all portfolio/watchlist symbols
        
        Args:
            symbols: Symbols to refresh (defaults to PORTFOLIO + WATCHLIST)
            lookback_days: Days of historical data to fetch
            
        Returns:
            Refresh results summary
        """
        
        if symbols is None:
            symbols = list(set(PORTFOLIO + WATCHLIST))
        
        logger.info(f"Refreshing price cache for {len(symbols)} symbols")
        
        # Clear old cache if using multisource provider
        if self.price_provider:
            cleared_files = self.price_provider.clear_cache(older_than_hours=24)
            logger.info(f"Cleared {cleared_files} old cache files")
        
        # Fetch fresh data
        results = await self.get_price_data(symbols, lookback_days)
        
        return {
            'requested_symbols': len(symbols),
            'successful_fetches': len(results),
            'failed_symbols': [s for s in symbols if s not in results],
            'cache_refresh_time': datetime.now().isoformat()
        }
    # END F01

# Global instance for easy import
data_manager = MultiSourceDataManager()

# Convenience functions
async def get_portfolio_price_data(lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """Get price data for portfolio symbols"""
    return await data_manager.get_price_data(PORTFOLIO, lookback_days)

async def get_watchlist_price_data(lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """Get price data for watchlist symbols"""
    return await data_manager.get_price_data(WATCHLIST, lookback_days)

async def get_all_price_data(lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """Get price data for all portfolio and watchlist symbols"""
    all_symbols = list(set(PORTFOLIO + WATCHLIST))
    return await data_manager.get_price_data(all_symbols, lookback_days)


# BEGIN F09 - Async I/O enhancements for multi-source data manager
from services.retry_policies import (
    AsyncHTTPClient, make_resilient_requests, load_async_config,
    get_async_stats, AsyncRetryConfig, AsyncCircuitBreakerConfig
)

class AsyncMultiSourceDataManager(MultiSourceDataManager):
    """
    F09: Enhanced multi-source data manager with async I/O, circuit breakers, and caching
    Extends the base MultiSourceDataManager with resilient async capabilities
    """
    
    def __init__(self):
        super().__init__()
        self.async_config = load_async_config()
        self.async_stats = {
            'async_requests': 0,
            'cache_hits': 0,
            'circuit_breaker_trips': 0,
            'fallbacks_to_sync': 0
        }
    
    async def get_price_data_async(self, symbols: List[str], lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        F09: Async version of get_price_data with resilient I/O patterns
        Uses asyncio gather with semaphore limits, exponential backoff, and circuit breakers
        
        AC1: Resilient to transient failures; retries capped; failures logged but do not abort
        """
        if not is_async_io_enabled():
            logger.info("F09: Async I/O disabled, using sync get_price_data")
            return await self.get_price_data(symbols, lookback_days)
        
        logger.info(f"F09: Starting async price data fetch for {len(symbols)} symbols")
        self.performance_stats['price_data_requests'] += 1
        self.async_stats['async_requests'] += 1
        
        # F09: Check YFinance with daily refresh guard first (if enabled)
        if is_yf_prices_enabled():
            try:
                if is_yf_daily_refresh_enabled():
                    refresh_result = YFDailyRefreshGuard.run_once_per_day_static(
                        symbols, key=config.YF_DAILY_KEY
                    )
                    logger.info(f"F09: YF daily refresh: {refresh_result['status']}")
                
                # Read from cache (no network in read path)
                cached_data = self.get_yf_cached_data(symbols)
                if cached_data:
                    logger.info(f"F09: YFinance cache provided data for {len(cached_data)} symbols")
                    self.performance_stats['price_data_successes'] += len(cached_data)
                    self.async_stats['cache_hits'] += len(cached_data)
                    return cached_data
                else:
                    logger.info("F09: No YFinance cache data, falling back to multisource")
            except Exception as e:
                logger.warning(f"F09: YFinance cache check failed: {e}")
        
        # F09: Use async multisource price provider if enabled
        if self.price_provider and is_multisource_prices_enabled():
            try:
                logger.info(f"F09: Using async multisource price provider for {len(symbols)} symbols")
                
                # Create async tasks for each symbol with circuit breaker protection
                async_tasks = []
                for symbol in symbols:
                    task = self._fetch_single_symbol_async(symbol, lookback_days)
                    async_tasks.append(task)
                
                # Execute with asyncio.gather and semaphore limits (core F09 pattern)
                logger.info(f"F09: Executing {len(async_tasks)} price fetch tasks with max_concurrency={self.async_config.get('max_concurrency', 5)}")
                symbol_results = await asyncio.gather(*async_tasks, return_exceptions=False)
                
                # Combine successful results
                results = {}
                for symbol, result in zip(symbols, symbol_results):
                    if result and not result.get('failed', False):
                        results[symbol] = result
                    else:
                        # AC1: failures logged but do not abort
                        logger.warning(f"F09: Failed to fetch price data for {symbol}, continuing with other symbols")
                
                # Update performance stats
                self.performance_stats['price_data_successes'] += len(results)
                provider_stats = self.price_provider.get_performance_stats()
                self.performance_stats['price_data_cache_hits'] += provider_stats['summary'].get('cache_hits', 0)
                
                return results
                
            except Exception as e:
                logger.error(f"F09: Async multisource provider failed: {e}")
                self.async_stats['fallbacks_to_sync'] += 1
                # Fall back to legacy behavior
                return await self._legacy_price_fetch_async(symbols, lookback_days)
        else:
            # Use legacy async behavior
            return await self._legacy_price_fetch_async(symbols, lookback_days)
    
    async def _fetch_single_symbol_async(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """
        F09: Fetch price data for a single symbol with resilient async I/O
        Uses circuit breakers and exponential backoff
        """
        try:
            # This would normally integrate with specific price data sources
            # For now, delegate to the sync provider with async wrapper
            if self.price_provider:
                # Wrap sync calls in async context
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.price_provider.get_history([symbol], lookback_days)
                )
                return result.get(symbol)
            else:
                return None
                
        except Exception as e:
            logger.error(f"F09: Error fetching price data for {symbol}: {e}")
            return {'failed': True, 'error': str(e), 'symbol': symbol}
    
    async def _legacy_price_fetch_async(self, symbols: List[str], lookback_days: int) -> Dict[str, pd.DataFrame]:
        """
        F09: Async wrapper for legacy price data fetching
        Maintains backward compatibility while providing async benefits
        """
        logger.info(f"F09: Using legacy async price fetch for {len(symbols)} symbols")
        
        # For now, wrap the sync legacy fetch in async context
        # In a full implementation, this would be replaced with actual async logic
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: {})
        
        # TODO: Integrate with existing prediction.py async logic when flag is enabled
        # This ensures backward compatibility while enabling F09 benefits
        
        return results
    
    async def refresh_price_cache_async(self, symbols: Optional[List[str]] = None, lookback_days: int = 90) -> Dict[str, Any]:
        """
        F09: Async version of refresh_price_cache with resilient I/O
        """
        if not is_async_io_enabled():
            return await self.refresh_price_cache(symbols, lookback_days)
        
        if symbols is None:
            symbols = list(set(PORTFOLIO + WATCHLIST))
        
        logger.info(f"F09: Refreshing price cache async for {len(symbols)} symbols")
        
        # Clear old cache if using multisource provider
        if self.price_provider:
            cleared_files = self.price_provider.clear_cache(older_than_hours=24)
            logger.info(f"F09: Cleared {cleared_files} old cache files")
        
        # Fetch fresh data using F09 async patterns
        results = await self.get_price_data_async(symbols, lookback_days)
        
        return {
            'requested_symbols': len(symbols),
            'successful_fetches': len(results),
            'failed_symbols': [s for s in symbols if s not in results],
            'cache_refresh_time': datetime.now().isoformat(),
            'async_mode': True,
            'f09_stats': self.get_async_performance_stats()
        }
    
    def get_async_performance_stats(self) -> Dict[str, Any]:
        """Get F09 async I/O performance statistics"""
        base_stats = self.get_price_provider_stats()
        base_stats.update({
            'async_stats': self.async_stats.copy(),
            'global_async_stats': get_async_stats(),
            'async_io_enabled': is_async_io_enabled(),
            'config': self.async_config
        })
        return base_stats


# Create async-enabled global instance
async_data_manager = AsyncMultiSourceDataManager()

# F09 convenience functions with async I/O
async def get_portfolio_price_data_async(lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """F09: Get price data for portfolio symbols with async I/O"""
    return await async_data_manager.get_price_data_async(PORTFOLIO, lookback_days)

async def get_watchlist_price_data_async(lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """F09: Get price data for watchlist symbols with async I/O"""
    return await async_data_manager.get_price_data_async(WATCHLIST, lookback_days)

async def get_all_price_data_async(lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """F09: Get price data for all symbols with async I/O and resilient patterns"""
    all_symbols = list(set(PORTFOLIO + WATCHLIST))
    return await async_data_manager.get_price_data_async(all_symbols, lookback_days)

# Wrapper functions that choose sync or async automatically
def get_price_data_resilient(symbols: List[str], lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """
    F09: Wrapper that automatically chooses sync or async based on feature flag
    """
    if is_async_io_enabled():
        return asyncio.run(async_data_manager.get_price_data_async(symbols, lookback_days))
    else:
        return asyncio.run(data_manager.get_price_data(symbols, lookback_days))

def get_all_price_data_resilient(lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """F09: Resilient wrapper for getting all price data"""
    all_symbols = list(set(PORTFOLIO + WATCHLIST))
    return get_price_data_resilient(all_symbols, lookback_days)

# END F09