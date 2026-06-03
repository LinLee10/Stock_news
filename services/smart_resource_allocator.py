"""
Smart Resource Allocation System for API Management

Intelligently allocates API calls based on:
- Portfolio vs Watchlist priority
- Historical usage patterns  
- Market conditions (trading hours, volatility)
- Data freshness requirements
- Source reliability scores
"""

import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import sqlite3
from pathlib import Path
from multi_source_data_manager import DataSource, DataQuality
from alpha_vantage_manager import RequestPriority

logger = logging.getLogger(__name__)

class AllocationStrategy(Enum):
    CONSERVATIVE = auto()  # Minimize API usage
    BALANCED = auto()      # Balance cost and quality
    AGGRESSIVE = auto()    # Prioritize data quality
    EMERGENCY = auto()     # Emergency mode with strict limits

class TickerType(Enum):
    PORTFOLIO = auto()     # Holdings - highest priority
    WATCHLIST = auto()     # Monitoring - medium priority
    RESEARCH = auto()      # Analysis - lowest priority

class MarketCondition(Enum):
    TRADING_HOURS = auto()
    AFTER_HOURS = auto()
    PRE_MARKET = auto()
    WEEKEND = auto()
    HOLIDAY = auto()

@dataclass
class TickerProfile:
    ticker: str
    type: TickerType
    weight: float = 1.0  # Portfolio weight or watchlist priority
    last_updated: Optional[datetime] = None
    update_frequency: timedelta = field(default_factory=lambda: timedelta(hours=4))
    volatility_score: float = 1.0  # Higher = more frequent updates needed
    news_sensitivity: float = 1.0  # Higher = more sensitive to news
    enabled: bool = True

@dataclass
class AllocationBudget:
    source: DataSource
    daily_total: int
    portfolio_allocation: int
    watchlist_allocation: int
    research_allocation: int
    emergency_reserve: int
    current_usage: Dict[TickerType, int] = field(default_factory=lambda: {
        TickerType.PORTFOLIO: 0,
        TickerType.WATCHLIST: 0, 
        TickerType.RESEARCH: 0
    })

@dataclass
class AllocationRequest:
    ticker: str
    data_type: str
    ticker_type: TickerType
    priority: RequestPriority
    required_quality: DataQuality
    max_age: timedelta = field(default_factory=lambda: timedelta(hours=6))
    retry_count: int = 0

@dataclass
class AllocationDecision:
    approved: bool
    recommended_source: Optional[DataSource] = None
    recommended_priority: Optional[RequestPriority] = None
    estimated_cost: int = 0
    reason: str = ""
    defer_until: Optional[datetime] = None

class SmartResourceAllocator:
    """Intelligent allocation of API resources across tickers and sources"""
    
    def __init__(self, config_path: str = "config/config.py"):
        self.config_path = config_path
        self.strategy = AllocationStrategy.BALANCED
        self.ticker_profiles: Dict[str, TickerProfile] = {}
        self.allocation_budgets: Dict[DataSource, AllocationBudget] = {}
        self.db_path = "data/allocation_tracking.db"
        self.market_hours = {
            'open': time(9, 30),    # 9:30 AM ET
            'close': time(16, 0),   # 4:00 PM ET
            'pre_open': time(4, 0), # 4:00 AM ET
            'after_close': time(20, 0) # 8:00 PM ET
        }
        self._initialize_database()
        self._load_ticker_profiles()
        self._initialize_budgets()

    def _initialize_database(self):
        """Initialize SQLite database for allocation tracking"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Allocation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS allocation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                ticker_type TEXT NOT NULL,
                data_type TEXT NOT NULL,
                source TEXT NOT NULL,
                priority TEXT NOT NULL,
                cost INTEGER NOT NULL,
                success BOOLEAN NOT NULL,
                response_time_ms INTEGER,
                data_quality TEXT,
                strategy TEXT NOT NULL
            )
        """)
        
        # Usage statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_usage_stats (
                date TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                ticker_type TEXT NOT NULL,
                total_requests INTEGER NOT NULL,
                successful_requests INTEGER NOT NULL,
                total_cost INTEGER NOT NULL,
                average_response_time REAL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_allocation_ticker ON allocation_history(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_allocation_timestamp ON allocation_history(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_date ON daily_usage_stats(date)")
        
        conn.commit()
        conn.close()

    def _load_ticker_profiles(self):
        """Load ticker profiles from configuration"""
        try:
            # Load portfolio tickers
            portfolio_tickers = ['RTX', 'PFE', 'MRVL', 'ADI', 'LLY', 'RIVN', 'TSLA', 'PLTR']
            watchlist_tickers = ['NVDA', 'GOOGL', 'AMD', 'MSFT']
            
            # Create profiles for portfolio tickers (high priority)
            for ticker in portfolio_tickers:
                self.ticker_profiles[ticker] = TickerProfile(
                    ticker=ticker,
                    type=TickerType.PORTFOLIO,
                    weight=1.0,
                    update_frequency=timedelta(hours=2),  # More frequent updates
                    volatility_score=self._estimate_volatility(ticker),
                    news_sensitivity=1.2
                )
            
            # Create profiles for watchlist tickers (medium priority)
            for ticker in watchlist_tickers:
                self.ticker_profiles[ticker] = TickerProfile(
                    ticker=ticker,
                    type=TickerType.WATCHLIST,
                    weight=0.8,
                    update_frequency=timedelta(hours=6),  # Less frequent updates
                    volatility_score=self._estimate_volatility(ticker),
                    news_sensitivity=1.0
                )
                
        except Exception as e:
            logger.warning(f"Error loading ticker profiles: {e}")

    def _estimate_volatility(self, ticker: str) -> float:
        """Estimate volatility score for a ticker (placeholder implementation)"""
        # In a real implementation, this would use historical volatility data
        high_volatility = ['TSLA', 'RIVN', 'PLTR', 'AMD', 'NVDA']
        medium_volatility = ['GOOGL', 'MSFT', 'MRVL']
        
        if ticker in high_volatility:
            return 1.5
        elif ticker in medium_volatility:
            return 1.2
        else:
            return 1.0

    def _initialize_budgets(self):
        """Initialize allocation budgets for each data source"""
        # Alpha Vantage - Premium but limited
        self.allocation_budgets[DataSource.ALPHA_VANTAGE] = AllocationBudget(
            source=DataSource.ALPHA_VANTAGE,
            daily_total=25,
            portfolio_allocation=18,    # 72% for portfolio
            watchlist_allocation=5,     # 20% for watchlist  
            research_allocation=1,      # 4% for research
            emergency_reserve=1         # 4% emergency
        )
        
        # yfinance - Unlimited but lower quality
        self.allocation_budgets[DataSource.YFINANCE] = AllocationBudget(
            source=DataSource.YFINANCE,
            daily_total=10000,  # Effectively unlimited
            portfolio_allocation=5000,
            watchlist_allocation=3000,
            research_allocation=2000,
            emergency_reserve=0
        )
        
        # EODHD - High limits, good quality
        self.allocation_budgets[DataSource.EODHD] = AllocationBudget(
            source=DataSource.EODHD,
            daily_total=1000,   # Conservative estimate
            portfolio_allocation=600,
            watchlist_allocation=300,
            research_allocation=100,
            emergency_reserve=0
        )
        
        # FMP - Very limited lifetime calls
        self.allocation_budgets[DataSource.FMP] = AllocationBudget(
            source=DataSource.FMP,
            daily_total=10,     # Very conservative
            portfolio_allocation=8,
            watchlist_allocation=2,
            research_allocation=0,
            emergency_reserve=0
        )

    async def request_allocation(self, request: AllocationRequest) -> AllocationDecision:
        """Request resource allocation for a data fetch"""
        try:
            # Get ticker profile
            profile = self.ticker_profiles.get(request.ticker)
            if not profile or not profile.enabled:
                return AllocationDecision(
                    approved=False,
                    reason=f"Ticker {request.ticker} not in active profiles"
                )
            
            # Check if data is fresh enough
            if self._is_data_fresh_enough(profile, request.max_age):
                return AllocationDecision(
                    approved=False,
                    reason=f"Existing data for {request.ticker} is still fresh"
                )
            
            # Get current market condition
            market_condition = self._get_market_condition()
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(profile, request, market_condition)
            
            # Find best source allocation
            source_decision = self._find_best_source_allocation(request, priority_score)
            
            if source_decision.approved:
                # Reserve the allocation
                await self._reserve_allocation(source_decision, request)
                
            return source_decision
            
        except Exception as e:
            logger.error(f"Error in allocation request: {e}")
            return AllocationDecision(
                approved=False,
                reason=f"Allocation error: {e}"
            )

    def _is_data_fresh_enough(self, profile: TickerProfile, max_age: timedelta) -> bool:
        """Check if existing data is fresh enough"""
        if not profile.last_updated:
            return False
            
        age = datetime.now() - profile.last_updated
        
        # Adjust freshness requirements based on market conditions and volatility
        market_condition = self._get_market_condition()
        
        if market_condition == MarketCondition.TRADING_HOURS:
            # During trading hours, reduce max age for high volatility stocks
            adjusted_max_age = max_age / profile.volatility_score
        elif market_condition in [MarketCondition.WEEKEND, MarketCondition.HOLIDAY]:
            # During non-trading periods, allow older data
            adjusted_max_age = max_age * 2
        else:
            adjusted_max_age = max_age
            
        return age < adjusted_max_age

    def _get_market_condition(self) -> MarketCondition:
        """Determine current market condition"""
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        # Weekend check
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return MarketCondition.WEEKEND
            
        # Trading hours (assuming ET timezone)
        if self.market_hours['open'] <= current_time <= self.market_hours['close']:
            return MarketCondition.TRADING_HOURS
        elif self.market_hours['pre_open'] <= current_time < self.market_hours['open']:
            return MarketCondition.PRE_MARKET
        elif self.market_hours['close'] < current_time <= self.market_hours['after_close']:
            return MarketCondition.AFTER_HOURS
        else:
            return MarketCondition.AFTER_HOURS

    def _calculate_priority_score(self, profile: TickerProfile, request: AllocationRequest, market_condition: MarketCondition) -> float:
        """Calculate priority score for allocation decision"""
        base_score = 0.0
        
        # Ticker type priority
        if profile.type == TickerType.PORTFOLIO:
            base_score += 100.0
        elif profile.type == TickerType.WATCHLIST:
            base_score += 50.0
        else:
            base_score += 10.0
        
        # Portfolio weight
        base_score *= profile.weight
        
        # Volatility adjustment
        base_score *= profile.volatility_score
        
        # Market condition adjustment
        if market_condition == MarketCondition.TRADING_HOURS:
            base_score *= 1.5
        elif market_condition == MarketCondition.PRE_MARKET:
            base_score *= 1.2
        elif market_condition == MarketCondition.WEEKEND:
            base_score *= 0.3
            
        # Request priority adjustment
        if request.priority == RequestPriority.CRITICAL:
            base_score *= 2.0
        elif request.priority == RequestPriority.HIGH:
            base_score *= 1.5
        elif request.priority == RequestPriority.RESEARCH:
            base_score *= 0.8
        
        # Age-based urgency
        if profile.last_updated:
            age = datetime.now() - profile.last_updated
            if age > profile.update_frequency:
                urgency_multiplier = min(2.0, 1.0 + (age.total_seconds() / profile.update_frequency.total_seconds()))
                base_score *= urgency_multiplier
        
        # Retry penalty
        if request.retry_count > 0:
            base_score *= (0.8 ** request.retry_count)
            
        return base_score

    def _find_best_source_allocation(self, request: AllocationRequest, priority_score: float) -> AllocationDecision:
        """Find the best source allocation for the request"""
        
        # Sort sources by preference based on strategy and quality requirements
        source_preferences = self._get_source_preferences(request.required_quality)
        
        for source in source_preferences:
            budget = self.allocation_budgets[source]
            
            # Check if source has available allocation
            if self._has_available_allocation(budget, request.ticker_type):
                
                # Calculate estimated cost
                estimated_cost = self._estimate_request_cost(source, request.data_type)
                
                # Check priority threshold for expensive sources
                if source == DataSource.ALPHA_VANTAGE and priority_score < self._get_priority_threshold():
                    continue
                
                return AllocationDecision(
                    approved=True,
                    recommended_source=source,
                    recommended_priority=self._map_to_source_priority(request.priority),
                    estimated_cost=estimated_cost,
                    reason=f"Allocated to {source.value} (priority score: {priority_score:.1f})"
                )
        
        # No allocation available - determine deferral
        defer_until = self._calculate_defer_time(priority_score)
        
        return AllocationDecision(
            approved=False,
            reason="No allocation available in any source",
            defer_until=defer_until
        )

    def _get_source_preferences(self, required_quality: DataQuality) -> List[DataSource]:
        """Get source preferences based on required quality and strategy"""
        
        if self.strategy == AllocationStrategy.CONSERVATIVE:
            # Prefer free/unlimited sources
            return [DataSource.YFINANCE, DataSource.EODHD, DataSource.ALPHA_VANTAGE, DataSource.FMP]
            
        elif self.strategy == AllocationStrategy.AGGRESSIVE:
            # Prefer high-quality sources
            if required_quality == DataQuality.PREMIUM:
                return [DataSource.ALPHA_VANTAGE, DataSource.EODHD, DataSource.YFINANCE, DataSource.FMP]
            else:
                return [DataSource.EODHD, DataSource.ALPHA_VANTAGE, DataSource.YFINANCE, DataSource.FMP]
                
        else:  # BALANCED
            # Balance quality and cost
            return [DataSource.YFINANCE, DataSource.ALPHA_VANTAGE, DataSource.EODHD, DataSource.FMP]

    def _has_available_allocation(self, budget: AllocationBudget, ticker_type: TickerType) -> bool:
        """Check if source has available allocation for ticker type"""
        
        if ticker_type == TickerType.PORTFOLIO:
            return budget.current_usage[ticker_type] < budget.portfolio_allocation
        elif ticker_type == TickerType.WATCHLIST:
            return budget.current_usage[ticker_type] < budget.watchlist_allocation
        else:  # RESEARCH
            return budget.current_usage[ticker_type] < budget.research_allocation

    def _estimate_request_cost(self, source: DataSource, data_type: str) -> int:
        """Estimate the cost of a request to a source"""
        # Most sources charge 1 call per request
        base_cost = 1
        
        # Some data types might cost more
        if data_type in ['intraday', 'realtime']:
            base_cost = 2
        elif data_type in ['fundamental', 'news']:
            base_cost = 1
            
        return base_cost

    def _get_priority_threshold(self) -> float:
        """Get minimum priority score required for expensive sources"""
        if self.strategy == AllocationStrategy.CONSERVATIVE:
            return 150.0  # High threshold
        elif self.strategy == AllocationStrategy.AGGRESSIVE:
            return 50.0   # Low threshold
        else:
            return 100.0  # Medium threshold

    def _map_to_source_priority(self, priority: RequestPriority) -> RequestPriority:
        """Map request priority to source-specific priority"""
        # For now, pass through unchanged
        # Could implement source-specific priority mapping
        return priority

    def _calculate_defer_time(self, priority_score: float) -> datetime:
        """Calculate when to retry the request"""
        # Higher priority = shorter deferral
        if priority_score > 150:
            defer_minutes = 5
        elif priority_score > 100:
            defer_minutes = 15
        elif priority_score > 50:
            defer_minutes = 30
        else:
            defer_minutes = 60
            
        return datetime.now() + timedelta(minutes=defer_minutes)

    async def _reserve_allocation(self, decision: AllocationDecision, request: AllocationRequest):
        """Reserve the allocated resource"""
        if decision.recommended_source:
            budget = self.allocation_budgets[decision.recommended_source]
            ticker_type = self.ticker_profiles[request.ticker].type
            
            # Update current usage
            budget.current_usage[ticker_type] += decision.estimated_cost
            
            # Log the allocation
            await self._log_allocation(decision, request)

    async def _log_allocation(self, decision: AllocationDecision, request: AllocationRequest):
        """Log allocation decision to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            profile = self.ticker_profiles[request.ticker]
            
            cursor.execute("""
                INSERT INTO allocation_history 
                (timestamp, ticker, ticker_type, data_type, source, priority, cost, success, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(datetime.now().timestamp()),
                request.ticker,
                profile.type.name,
                request.data_type,
                decision.recommended_source.value if decision.recommended_source else 'NONE',
                request.priority.name,
                decision.estimated_cost,
                decision.approved,
                self.strategy.name
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to log allocation: {e}")

    async def report_allocation_result(self, request: AllocationRequest, success: bool, 
                                     response_time_ms: int, data_quality: Optional[DataQuality]):
        """Report the result of an allocation to update statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update the most recent allocation record
            cursor.execute("""
                UPDATE allocation_history 
                SET success = ?, response_time_ms = ?, data_quality = ?
                WHERE ticker = ? AND timestamp = (
                    SELECT MAX(timestamp) FROM allocation_history WHERE ticker = ?
                )
            """, (
                success,
                response_time_ms,
                data_quality.value if data_quality else None,
                request.ticker,
                request.ticker
            ))
            
            # Update ticker profile last_updated time if successful
            if success and request.ticker in self.ticker_profiles:
                self.ticker_profiles[request.ticker].last_updated = datetime.now()
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to report allocation result: {e}")

    def set_allocation_strategy(self, strategy: AllocationStrategy):
        """Update allocation strategy"""
        self.strategy = strategy
        logger.info(f"Allocation strategy set to {strategy.name}")

    def get_allocation_statistics(self) -> Dict[str, Any]:
        """Get current allocation statistics"""
        stats = {
            'strategy': self.strategy.name,
            'market_condition': self._get_market_condition().name,
            'budgets': {}
        }
        
        for source, budget in self.allocation_budgets.items():
            total_used = sum(budget.current_usage.values())
            stats['budgets'][source.value] = {
                'daily_total': budget.daily_total,
                'total_used': total_used,
                'total_remaining': budget.daily_total - total_used,
                'portfolio_used': budget.current_usage[TickerType.PORTFOLIO],
                'portfolio_remaining': budget.portfolio_allocation - budget.current_usage[TickerType.PORTFOLIO],
                'watchlist_used': budget.current_usage[TickerType.WATCHLIST],
                'watchlist_remaining': budget.watchlist_allocation - budget.current_usage[TickerType.WATCHLIST],
                'research_used': budget.current_usage[TickerType.RESEARCH],
                'research_remaining': budget.research_allocation - budget.current_usage[TickerType.RESEARCH]
            }
        
        return stats

    async def reset_daily_allocations(self):
        """Reset daily allocation counters (call at midnight)"""
        for budget in self.allocation_budgets.values():
            budget.current_usage = {
                TickerType.PORTFOLIO: 0,
                TickerType.WATCHLIST: 0,
                TickerType.RESEARCH: 0
            }
        
        logger.info("Daily allocation counters reset")

    def add_ticker_profile(self, ticker: str, ticker_type: TickerType, weight: float = 1.0):
        """Add a new ticker profile"""
        self.ticker_profiles[ticker] = TickerProfile(
            ticker=ticker,
            type=ticker_type,
            weight=weight,
            update_frequency=timedelta(hours=4) if ticker_type == TickerType.PORTFOLIO else timedelta(hours=8),
            volatility_score=self._estimate_volatility(ticker)
        )
        
        logger.info(f"Added ticker profile: {ticker} ({ticker_type.name})")

    def remove_ticker_profile(self, ticker: str):
        """Remove a ticker profile"""
        if ticker in self.ticker_profiles:
            del self.ticker_profiles[ticker]
            logger.info(f"Removed ticker profile: {ticker}")

# Convenience functions
async def request_ticker_data_allocation(ticker: str, data_type: str = 'daily', 
                                       priority: RequestPriority = RequestPriority.RESEARCH) -> AllocationDecision:
    """Request allocation for ticker data"""
    allocator = SmartResourceAllocator()
    
    # Determine ticker type
    portfolio_tickers = ['RTX', 'PFE', 'MRVL', 'ADI', 'LLY', 'RIVN', 'TSLA', 'PLTR']
    watchlist_tickers = ['NVDA', 'GOOGL', 'AMD', 'MSFT']
    
    if ticker in portfolio_tickers:
        ticker_type = TickerType.PORTFOLIO
    elif ticker in watchlist_tickers:
        ticker_type = TickerType.WATCHLIST
    else:
        ticker_type = TickerType.RESEARCH
    
    request = AllocationRequest(
        ticker=ticker,
        data_type=data_type,
        ticker_type=ticker_type,
        priority=priority,
        required_quality=DataQuality.STANDARD
    )
    
    return await allocator.request_allocation(request)