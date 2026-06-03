"""
Smart Cache Manager with Alpha Vantage Quota Ledger

Production-ready quota management and intelligent caching:
- Persistent quota tracking with daily resets
- Predictive overrun prevention
- Atomic ledger updates for consistency
- Performance optimization through intelligent caching
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import threading
import fcntl
import os

logger = logging.getLogger(__name__)

@dataclass
class QuotaState:
    """Alpha Vantage quota state tracking"""
    calls_made: int = 0
    calls_remaining: int = 25
    daily_limit: int = 25
    reset_date: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    last_reset: datetime = field(default_factory=datetime.now)
    batch_calls_saved: int = 0  # Calls saved through batching
    fallback_triggered: int = 0  # Times fallback was used due to quota
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'calls_made': self.calls_made,
            'calls_remaining': self.calls_remaining,
            'daily_limit': self.daily_limit,
            'reset_date': self.reset_date,
            'last_reset': self.last_reset.isoformat(),
            'batch_calls_saved': self.batch_calls_saved,
            'fallback_triggered': self.fallback_triggered
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuotaState':
        return cls(
            calls_made=data.get('calls_made', 0),
            calls_remaining=data.get('calls_remaining', 25),
            daily_limit=data.get('daily_limit', 25),
            reset_date=data.get('reset_date', datetime.now().strftime('%Y-%m-%d')),
            last_reset=datetime.fromisoformat(data.get('last_reset', datetime.now().isoformat())),
            batch_calls_saved=data.get('batch_calls_saved', 0),
            fallback_triggered=data.get('fallback_triggered', 0)
        )

class QuotaLedger:
    """
    Thread-safe, persistent quota ledger for Alpha Vantage API
    
    Features:
    - Atomic file operations with file locking
    - Automatic daily reset detection
    - Predictive quota overrun prevention
    - Comprehensive usage statistics
    """
    
    def __init__(self, ledger_path: str = "data/av_quota_state.json", daily_limit: int = 25):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_limit = daily_limit
        self._lock = threading.Lock()
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Check if we need a daily reset
        self._check_daily_reset()
        
        logger.info(f"QuotaLedger initialized: {self.state.calls_remaining}/{self.daily_limit} calls remaining")
    
    def _load_state(self) -> QuotaState:
        """Load quota state from persistent storage"""
        if not self.ledger_path.exists():
            logger.info("No existing quota ledger found, creating new one")
            return QuotaState(daily_limit=self.daily_limit, calls_remaining=self.daily_limit)
        
        try:
            with open(self.ledger_path, 'r') as f:
                # Use file locking for concurrent access
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                return QuotaState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load quota ledger: {e}, creating new state")
            return QuotaState(daily_limit=self.daily_limit, calls_remaining=self.daily_limit)
    
    def _save_state(self) -> bool:
        """Atomically save quota state to persistent storage"""
        try:
            # Use temporary file for atomic writes
            temp_path = self.ledger_path.with_suffix('.tmp')
            
            with open(temp_path, 'w') as f:
                # Use file locking for concurrent access
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(self.state.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic rename
            temp_path.replace(self.ledger_path)
            
            logger.debug(f"Quota state saved: {self.state.calls_made}/{self.daily_limit} calls used")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save quota ledger: {e}")
            return False
    
    def _check_daily_reset(self) -> bool:
        """Check if daily reset is needed and perform it"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if self.state.reset_date != today:
            logger.info(f"Daily quota reset: {self.state.reset_date} → {today}")
            
            # Reset counters
            old_calls_made = self.state.calls_made
            self.state.calls_made = 0
            self.state.calls_remaining = self.daily_limit
            self.state.reset_date = today
            self.state.last_reset = datetime.now()
            
            # Save the reset
            self._save_state()
            
            logger.info(f"Quota reset complete: {old_calls_made} calls used yesterday, {self.daily_limit} available today")
            return True
        
        return False
    
    def can_make_calls(self, num_calls: int) -> Tuple[bool, str]:
        """
        Check if we can make the requested number of API calls
        
        Args:
            num_calls: Number of calls needed
            
        Returns:
            (can_proceed, reason)
        """
        with self._lock:
            self._check_daily_reset()
            
            if self.state.calls_remaining >= num_calls:
                return True, f"OK: {self.state.calls_remaining} calls available"
            else:
                reason = f"Quota insufficient: need {num_calls}, have {self.state.calls_remaining}"
                return False, reason
    
    def reserve_calls(self, num_calls: int) -> bool:
        """
        Reserve API calls (pre-allocation before making requests)
        
        Args:
            num_calls: Number of calls to reserve
            
        Returns:
            True if reservation successful, False otherwise
        """
        with self._lock:
            can_proceed, reason = self.can_make_calls(num_calls)
            
            if can_proceed:
                self.state.calls_remaining -= num_calls
                logger.debug(f"Reserved {num_calls} calls: {self.state.calls_remaining} remaining")
                return self._save_state()
            else:
                logger.warning(f"Call reservation failed: {reason}")
                return False
    
    def consume_calls(self, num_calls: int, calls_saved: int = 0) -> bool:
        """
        Mark API calls as consumed (after successful requests)
        
        Args:
            num_calls: Number of calls actually made
            calls_saved: Number of calls saved through batching optimization
            
        Returns:
            True if update successful
        """
        with self._lock:
            self.state.calls_made += num_calls
            
            if calls_saved > 0:
                self.state.batch_calls_saved += calls_saved
                logger.info(f"Batch optimization saved {calls_saved} API calls")
            
            logger.info(f"Consumed {num_calls} calls: {self.state.calls_made}/{self.daily_limit} used")
            return self._save_state()
    
    def trigger_fallback(self, reason: str = "") -> bool:
        """Record that fallback was triggered due to quota exhaustion"""
        with self._lock:
            self.state.fallback_triggered += 1
            logger.warning(f"Fallback triggered (#{self.state.fallback_triggered}): {reason}")
            return self._save_state()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive quota status"""
        with self._lock:
            self._check_daily_reset()
            
            return {
                'quota_status': {
                    'calls_made': self.state.calls_made,
                    'calls_remaining': self.state.calls_remaining,
                    'daily_limit': self.daily_limit,
                    'utilization_pct': (self.state.calls_made / self.daily_limit) * 100,
                },
                'efficiency_metrics': {
                    'batch_calls_saved': self.state.batch_calls_saved,
                    'fallback_triggered': self.state.fallback_triggered,
                    'efficiency_gain_pct': (self.state.batch_calls_saved / max(self.state.calls_made + self.state.batch_calls_saved, 1)) * 100
                },
                'timing': {
                    'reset_date': self.state.reset_date,
                    'last_reset': self.state.last_reset.isoformat(),
                    'hours_until_reset': self._hours_until_reset()
                }
            }
    
    def _hours_until_reset(self) -> float:
        """Calculate hours until next quota reset"""
        now = datetime.now()
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (tomorrow - now).total_seconds() / 3600
    
    def reset_for_testing(self) -> None:
        """Reset quota state for testing purposes"""
        with self._lock:
            self.state = QuotaState(daily_limit=self.daily_limit, calls_remaining=self.daily_limit)
            self._save_state()
            logger.warning("Quota state reset for testing")

# BEGIN F02 - Smart Cache Manager with Budget Ledger
class SmartCacheManager:
    """
    Intelligent cache manager with quota awareness and optimization
    
    Features:
    - Quota-aware cache invalidation
    - Predictive cache warming
    - Batch operation optimization
    - Performance metrics tracking
    """
    
    def __init__(self, cache_dir: str = "data/av_bulk_cache", daily_limit: int = 25):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize quota ledger
        self.quota_ledger = QuotaLedger(
            ledger_path=self.cache_dir.parent / "av_quota_state.json",
            daily_limit=daily_limit
        )
        
        # Performance tracking
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'quota_prevented_calls': 0,
            'batch_optimizations': 0,
            'total_requests': 0
        }
        
        logger.info(f"SmartCacheManager initialized with quota ledger")
    
    def plan_batch_requests(self, symbols: List[str], lookback_days: int) -> Dict[str, Any]:
        """
        Plan optimal batch requests considering quota constraints
        
        Args:
            symbols: List of symbols to fetch
            lookback_days: Days of historical data needed
            
        Returns:
            Batch plan with quota impact analysis
        """
        # Check what's available in cache
        cached_symbols = []
        missing_symbols = []
        
        for symbol in symbols:
            cache_path = self._get_cache_path(symbol, lookback_days)
            if self._is_cache_valid(cache_path):
                cached_symbols.append(symbol)
                self.metrics['cache_hits'] += 1
            else:
                missing_symbols.append(symbol)
                self.metrics['cache_misses'] += 1
        
        # Calculate optimal batch strategy
        batch_size = 100  # Alpha Vantage batch quotes limit
        
        if len(missing_symbols) <= batch_size:
            # Single batch call can handle all missing symbols
            batches = [missing_symbols] if missing_symbols else []
            calls_needed = len(batches)
            calls_saved = max(0, len(missing_symbols) - calls_needed)
        else:
            # Multiple batches needed
            batches = [missing_symbols[i:i + batch_size] for i in range(0, len(missing_symbols), batch_size)]
            calls_needed = len(batches)
            calls_saved = len(missing_symbols) - calls_needed
        
        # Check quota availability
        can_proceed, quota_reason = self.quota_ledger.can_make_calls(calls_needed)
        
        plan = {
            'symbols': {
                'total': len(symbols),
                'cached': len(cached_symbols),
                'missing': len(missing_symbols),
                'cached_symbols': cached_symbols,
                'missing_symbols': missing_symbols
            },
            'batches': {
                'count': len(batches),
                'batches': batches,
                'calls_needed': calls_needed,
                'calls_saved': calls_saved
            },
            'quota': {
                'can_proceed': can_proceed,
                'reason': quota_reason,
                'calls_remaining': self.quota_ledger.state.calls_remaining,
                'utilization_pct': (self.quota_ledger.state.calls_made / self.quota_ledger.daily_limit) * 100
            },
            'recommendations': self._generate_recommendations(can_proceed, calls_needed, len(missing_symbols))
        }
        
        logger.info(f"Batch plan: {len(cached_symbols)} cached, {len(missing_symbols)} missing, {calls_needed} calls needed")
        return plan
    
    def _generate_recommendations(self, can_proceed: bool, calls_needed: int, missing_count: int) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not can_proceed:
            recommendations.append("FALLBACK_REQUIRED: Quota insufficient, use alternative data source")
        
        if calls_needed < missing_count:
            savings = missing_count - calls_needed
            recommendations.append(f"BATCH_OPTIMIZATION: Saving {savings} API calls through batching")
        
        if self.quota_ledger.state.calls_remaining < 5:
            recommendations.append("QUOTA_WARNING: Less than 5 calls remaining, consider caching strategy")
        
        if missing_count > 50:
            recommendations.append("LARGE_REQUEST: Consider splitting into multiple runs for better cache efficiency")
        
        return recommendations
    
    def execute_batch_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the planned batch requests with quota tracking
        
        Args:
            plan: Batch plan from plan_batch_requests()
            
        Returns:
            Execution results with performance metrics
        """
        results = {
            'success': False,
            'symbols_fetched': [],
            'symbols_failed': [],
            'calls_made': 0,
            'calls_saved': plan['batches']['calls_saved'],
            'cache_hits': len(plan['symbols']['cached_symbols']),
            'fallback_used': False,
            'execution_time': 0
        }
        
        start_time = time.time()
        
        # If quota insufficient, trigger fallback immediately
        if not plan['quota']['can_proceed']:
            self.quota_ledger.trigger_fallback("Quota insufficient for planned batch requests")
            self.metrics['quota_prevented_calls'] += plan['batches']['calls_needed']
            results['fallback_used'] = True
            results['execution_time'] = time.time() - start_time
            return results
        
        # Reserve quota for planned calls
        if plan['batches']['calls_needed'] > 0:
            if not self.quota_ledger.reserve_calls(plan['batches']['calls_needed']):
                logger.error("Failed to reserve quota calls")
                results['fallback_used'] = True
                results['execution_time'] = time.time() - start_time
                return results
        
        # Track batch optimizations
        if plan['batches']['calls_saved'] > 0:
            self.metrics['batch_optimizations'] += 1
        
        # Update metrics
        self.metrics['total_requests'] += 1
        
        # Record successful execution (actual API calls would happen in price provider)
        results['success'] = True
        results['calls_made'] = plan['batches']['calls_needed']
        results['execution_time'] = time.time() - start_time
        
        logger.info(f"Batch plan executed: {results['calls_made']} calls made, {results['calls_saved']} calls saved")
        
        return results
    
    def _get_cache_path(self, symbol: str, lookback_days: int) -> Path:
        """Get cache file path for symbol"""
        return self.cache_dir / f"{symbol}_{lookback_days}d.csv"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False
        
        file_age = time.time() - cache_path.stat().st_mtime
        return file_age < (max_age_hours * 3600)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        quota_status = self.quota_ledger.get_status()
        
        return {
            'cache_metrics': {
                'hits': self.metrics['cache_hits'],
                'misses': self.metrics['cache_misses'],
                'hit_rate_pct': (self.metrics['cache_hits'] / max(self.metrics['cache_hits'] + self.metrics['cache_misses'], 1)) * 100
            },
            'quota_metrics': quota_status,
            'optimization_metrics': {
                'quota_prevented_calls': self.metrics['quota_prevented_calls'],
                'batch_optimizations': self.metrics['batch_optimizations'],
                'total_requests': self.metrics['total_requests']
            }
        }
    
    def clear_old_cache(self, max_age_hours: int = 48) -> int:
        """Clear cache files older than specified age"""
        cleared_count = 0
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for cache_file in self.cache_dir.glob("*.csv"):
            if cache_file.stat().st_mtime < cutoff_time:
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old cache files")
        
        return cleared_count
# END F02

# Global instance for easy import
smart_cache = SmartCacheManager()