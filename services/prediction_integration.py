"""
Prediction System Integration Layer

Integrates the new smart API management system with the existing prediction.py module.
Provides backward compatibility while adding intelligent data fetching capabilities.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# Add services to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from services.multi_source_data_manager import (
    MultiSourceDataManager, 
    MarketDataRequest, 
    DataResponse,
    DataSource,
    DataQuality
)
from services.smart_resource_allocator import (
    SmartResourceAllocator,
    AllocationRequest,
    TickerType,
    AllocationStrategy
)
from services.alpha_vantage_manager import RequestPriority
from services.intelligent_cache_system import IntelligentCacheSystem
from services.data_quality_validator import DataQualityValidator, validate_ticker_data

logger = logging.getLogger(__name__)

class SmartPredictionDataManager:
    """Integration layer that replaces the basic Alpha Vantage calls in prediction.py"""
    
    def __init__(self):
        self.data_manager: Optional[MultiSourceDataManager] = None
        self.resource_allocator: Optional[SmartResourceAllocator] = None
        self.cache_system: Optional[IntelligentCacheSystem] = None
        self.quality_validator: Optional[DataQualityValidator] = None
        self.initialized = False
        
        # Portfolio and watchlist configuration (from original prediction.py analysis)
        self.portfolio_tickers = ['RTX', 'PFE', 'MRVL', 'ADI', 'LLY', 'RIVN', 'TSLA', 'PLTR']
        self.watchlist_tickers = ['NVDA', 'GOOGL', 'AMD', 'MSFT']
        
        # Quality thresholds
        self.min_quality_score = 60.0
        self.prefer_premium_for_portfolio = True

    async def initialize(self) -> bool:
        """Initialize all components of the smart data management system"""
        try:
            logger.info("Initializing SmartPredictionDataManager...")
            
            # Initialize components
            self.data_manager = MultiSourceDataManager()
            self.resource_allocator = SmartResourceAllocator()
            self.cache_system = IntelligentCacheSystem()
            self.quality_validator = DataQualityValidator()
            
            # Initialize all components
            init_tasks = [
                self.data_manager.initialize(),
                self.cache_system.initialize()
            ]
            
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Check initialization results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to initialize component {i}: {result}")
                    return False
                elif not result:
                    logger.error(f"Component {i} initialization returned False")
                    return False
            
            # Set allocation strategy based on configuration
            self.resource_allocator.set_allocation_strategy(AllocationStrategy.BALANCED)
            
            self.initialized = True
            logger.info("SmartPredictionDataManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SmartPredictionDataManager: {e}")
            return False

    async def fetch_ticker_data_smart(self, ticker: str, period: str = "90d", 
                                    priority: RequestPriority = RequestPriority.RESEARCH,
                                    require_quality_validation: bool = True) -> Optional[pd.DataFrame]:
        """
        Smart replacement for the original fetch_price_history_bulk function.
        Returns DataFrame compatible with existing prediction.py code.
        """
        if not self.initialized:
            logger.error("SmartPredictionDataManager not initialized")
            return None
        
        try:
            # Determine ticker type for resource allocation
            ticker_type = self._get_ticker_type(ticker)
            
            # Request allocation
            allocation_request = AllocationRequest(
                ticker=ticker,
                data_type='daily',
                ticker_type=ticker_type,
                priority=priority,
                required_quality=DataQuality.PREMIUM if ticker_type == TickerType.PORTFOLIO else DataQuality.STANDARD
            )
            
            allocation_decision = await self.resource_allocator.request_allocation(allocation_request)
            
            if not allocation_decision.approved:
                logger.warning(f"Allocation denied for {ticker}: {allocation_decision.reason}")
                
                # Try to get cached data as fallback
                cached_data = await self._get_cached_data_fallback(ticker, period)
                if cached_data is not None:
                    logger.info(f"Using cached fallback data for {ticker}")
                    return cached_data
                
                return None
            
            # Make data request
            market_request = MarketDataRequest(
                ticker=ticker,
                data_type='daily',
                period=period,
                priority=priority
            )
            
            start_time = datetime.now()
            response = await self.data_manager.get_market_data(market_request)
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Report allocation result
            await self.resource_allocator.report_allocation_result(
                allocation_request, 
                response.success, 
                int(response_time_ms),
                response.quality
            )
            
            if not response.success:
                logger.error(f"Failed to fetch data for {ticker}: {response.error_message}")
                return None
            
            # Validate data quality if required
            if require_quality_validation:
                validation_result = await validate_ticker_data(
                    ticker, 'daily', response.data, response.source
                )
                
                if validation_result.quality_score < self.min_quality_score:
                    logger.warning(f"Data quality too low for {ticker}: {validation_result.quality_score:.1f}%")
                    
                    # Try fallback source if quality is poor
                    fallback_response = await self._try_fallback_source(market_request, response.source)
                    if fallback_response and fallback_response.success:
                        response = fallback_response
                        logger.info(f"Using fallback source {response.source.value} for {ticker}")
            
            # Convert to pandas DataFrame (compatible with existing prediction.py)
            df = self._convert_to_prediction_format(response.data, ticker)
            
            # Cache the processed data
            await self._cache_processed_data(ticker, df, response.source, response.quality)
            
            logger.info(f"Successfully fetched {len(df)} data points for {ticker} from {response.source.value}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching smart data for {ticker}: {e}")
            return None

    async def fetch_bulk_data_smart(self, tickers: List[str], period: str = "90d") -> Dict[str, pd.DataFrame]:
        """
        Smart replacement for fetch_price_history_bulk with concurrent processing
        """
        if not self.initialized:
            logger.error("SmartPredictionDataManager not initialized")
            return {}
        
        logger.info(f"Fetching bulk data for {len(tickers)} tickers: {tickers}")
        
        # Create tasks for concurrent processing
        tasks = []
        for ticker in tickers:
            # Determine priority based on ticker type
            ticker_type = self._get_ticker_type(ticker)
            priority = RequestPriority.HIGH if ticker_type == TickerType.PORTFOLIO else RequestPriority.RESEARCH
            
            task = self.fetch_ticker_data_smart(ticker, period, priority)
            tasks.append((ticker, task))
        
        # Execute requests concurrently with controlled parallelism
        results = {}
        batch_size = 5  # Process 5 tickers at a time to avoid overwhelming APIs
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_tasks = [task for _, task in batch]
            batch_tickers = [ticker for ticker, _ in batch]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for ticker, result in zip(batch_tickers, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching {ticker}: {result}")
                        results[ticker] = pd.DataFrame(columns=["Date", "Stock_Close"])
                    elif result is not None:
                        results[ticker] = result
                    else:
                        logger.warning(f"No data returned for {ticker}")
                        results[ticker] = pd.DataFrame(columns=["Date", "Stock_Close"])
                
                # Add small delay between batches to be respectful to APIs
                if i + batch_size < len(tasks):
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                for ticker in batch_tickers:
                    if ticker not in results:
                        results[ticker] = pd.DataFrame(columns=["Date", "Stock_Close"])
        
        # Log summary
        successful_tickers = [t for t, df in results.items() if len(df) > 0]
        logger.info(f"Successfully fetched data for {len(successful_tickers)}/{len(tickers)} tickers")
        
        return results

    def _get_ticker_type(self, ticker: str) -> TickerType:
        """Determine ticker type for resource allocation"""
        if ticker in self.portfolio_tickers:
            return TickerType.PORTFOLIO
        elif ticker in self.watchlist_tickers:
            return TickerType.WATCHLIST
        else:
            return TickerType.RESEARCH

    def _convert_to_prediction_format(self, raw_data: Dict[str, Any], ticker: str) -> pd.DataFrame:
        """Convert API response to format expected by prediction.py"""
        try:
            time_series = raw_data.get('Time Series (Daily)', {})
            
            if not time_series:
                logger.warning(f"No time series data found for {ticker}")
                return pd.DataFrame(columns=["Date", "Stock_Close"])
            
            # Convert to list of records
            records = []
            for date_str, daily_data in time_series.items():
                try:
                    date = pd.to_datetime(date_str)
                    close_price = float(daily_data.get('4. close', 0))
                    
                    if close_price > 0:  # Only include valid prices
                        records.append({
                            'Date': date,
                            'Stock_Close': close_price
                        })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid data for {ticker} on {date_str}: {e}")
                    continue
            
            if not records:
                logger.warning(f"No valid price records found for {ticker}")
                return pd.DataFrame(columns=["Date", "Stock_Close"])
            
            # Create DataFrame and sort by date
            df = pd.DataFrame(records)
            df = df.sort_values('Date').reset_index(drop=True)
            
            logger.debug(f"Converted {len(df)} records for {ticker} (date range: {df['Date'].min()} to {df['Date'].max()})")
            return df
            
        except Exception as e:
            logger.error(f"Error converting data format for {ticker}: {e}")
            return pd.DataFrame(columns=["Date", "Stock_Close"])

    async def _get_cached_data_fallback(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Get cached data as fallback when allocation is denied"""
        try:
            if self.cache_system:
                cache_key = f"{ticker}|daily|{period}"
                cached_result = await self.cache_system.get(cache_key)
                
                if cached_result:
                    data, entry = cached_result
                    df = self._convert_to_prediction_format(data, ticker)
                    if len(df) > 0:
                        return df
            
            # Try loading from disk cache (original prediction.py cache)
            cache_dir = "data/av_bulk_cache"
            cache_path = f"{cache_dir}/{ticker}_{period}.csv"
            
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
                if len(df) > 1:  # Has data beyond header
                    df["Date"] = pd.to_datetime(df["Date"])
                    if "Stock_Close" not in df.columns and len(df.columns) > 1:
                        df = df.rename(columns={df.columns[1]: "Stock_Close"})
                    
                    df = df[["Date", "Stock_Close"]].sort_values("Date").reset_index(drop=True)
                    return df
            
        except Exception as e:
            logger.warning(f"Error getting cached fallback for {ticker}: {e}")
        
        return None

    async def _try_fallback_source(self, request: MarketDataRequest, 
                                 failed_source: DataSource) -> Optional[DataResponse]:
        """Try alternative data source when primary source fails quality check"""
        try:
            # Create a new request with lower quality requirements
            fallback_request = MarketDataRequest(
                ticker=request.ticker,
                data_type=request.data_type,
                period=request.period,
                priority=RequestPriority.RESEARCH,  # Lower priority for fallback
                required_fields=request.required_fields
            )
            
            response = await self.data_manager.get_market_data(fallback_request)
            
            # Only return if it's from a different source
            if response.success and response.source != failed_source:
                return response
                
        except Exception as e:
            logger.warning(f"Fallback source attempt failed: {e}")
        
        return None

    async def _cache_processed_data(self, ticker: str, df: pd.DataFrame, 
                                  source: DataSource, quality: DataQuality):
        """Cache the processed DataFrame for future use"""
        try:
            if self.cache_system and len(df) > 0:
                # Convert back to API format for caching
                time_series = {}
                for _, row in df.iterrows():
                    date_str = row['Date'].strftime('%Y-%m-%d')
                    time_series[date_str] = {
                        '4. close': str(row['Stock_Close'])
                    }
                
                cache_data = {
                    'Meta Data': {
                        '2. Symbol': ticker,
                        '3. Last Refreshed': df['Date'].max().strftime('%Y-%m-%d')
                    },
                    'Time Series (Daily)': time_series
                }
                
                cache_key = f"{ticker}|daily|90d"  # Standard cache key
                await self.cache_system.put(cache_key, cache_data, source, quality)
                
        except Exception as e:
            logger.warning(f"Error caching processed data for {ticker}: {e}")

    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all system components"""
        stats = {
            'initialized': self.initialized,
            'portfolio_tickers': self.portfolio_tickers,
            'watchlist_tickers': self.watchlist_tickers
        }
        
        if not self.initialized:
            return stats
        
        try:
            # Get allocation statistics
            if self.resource_allocator:
                stats['allocation'] = self.resource_allocator.get_allocation_statistics()
            
            # Get cache statistics
            if self.cache_system:
                stats['cache'] = self.cache_system.get_statistics()
            
            # Get data manager statistics
            if self.data_manager:
                stats['data_sources'] = await self.data_manager.get_usage_statistics()
            
        except Exception as e:
            logger.warning(f"Error getting system statistics: {e}")
            stats['error'] = str(e)
        
        return stats

    async def cleanup(self):
        """Clean up all system resources"""
        if self.data_manager:
            await self.data_manager.cleanup()
        
        if self.cache_system:
            await self.cache_system.cleanup()

# Global instance for backward compatibility
_smart_manager: Optional[SmartPredictionDataManager] = None

async def get_smart_manager() -> SmartPredictionDataManager:
    """Get or create the global smart data manager instance"""
    global _smart_manager
    
    if _smart_manager is None:
        _smart_manager = SmartPredictionDataManager()
        await _smart_manager.initialize()
    
    return _smart_manager

# Backward compatibility functions that can replace the original prediction.py functions
async def fetch_price_history_bulk_smart(tickers: List[str], period: str = "90d") -> Dict[str, pd.DataFrame]:
    """
    Smart replacement for the original fetch_price_history_bulk function.
    Can be used as a drop-in replacement in prediction.py
    """
    manager = await get_smart_manager()
    return await manager.fetch_bulk_data_smart(tickers, period)

async def fetch_ticker_data_smart(ticker: str, period: str = "90d") -> Optional[pd.DataFrame]:
    """
    Smart replacement for individual ticker data fetching
    """
    manager = await get_smart_manager()
    return await manager.fetch_ticker_data_smart(ticker, period)

def create_compatibility_wrapper():
    """
    Creates a synchronous wrapper for integration with existing prediction.py code
    """
    def sync_fetch_price_history_bulk(tickers: List[str], period: str = "90d") -> Dict[str, pd.DataFrame]:
        """Synchronous wrapper for the smart data fetching"""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create task and run it
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(fetch_price_history_bulk_smart(tickers, period))
                    )
                    return future.result(timeout=60)  # 60 second timeout
            else:
                # Loop not running, can use asyncio.run
                return asyncio.run(fetch_price_history_bulk_smart(tickers, period))
        except Exception as e:
            logger.error(f"Error in sync wrapper: {e}")
            # Fallback to empty DataFrames
            return {ticker: pd.DataFrame(columns=["Date", "Stock_Close"]) for ticker in tickers}
    
    return sync_fetch_price_history_bulk

# Export the compatibility wrapper
sync_fetch_price_history_bulk = create_compatibility_wrapper()