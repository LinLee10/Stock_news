# Scheduling & Resets - Daily Pipeline Orchestration

## Daily Pipeline Window

### Optimal Execution Time: **6:00 PM ET** 
- **Rationale**: After US market close (4 PM ET), before Asian markets open
- **Duration Budget**: 15 minutes maximum (cloud platform limits)
- **Reset Advantage**: 6 hours before midnight UTC resets for most providers

### Execution Sequence
```
18:00 ET (22:00 UTC) - Pipeline Start
├── 18:00-18:03 - YFinance bulk OHLCV (portfolio + watchlist)
├── 18:03-18:05 - Price fallback (Alpha Vantage ≤25/day OR Twelve Data)  
├── 18:05-18:08 - US intraday (Alpaca ~390 bars)
├── 18:08-18:10 - Non-US intraday (Twelve Data, quota permitting)
├── 18:10-18:12 - News ingestion (Finnhub prioritized, NewsAPI/GNews breadth)
├── 18:12-18:14 - Indicators (Twelve Data remote OR local computation)  
└── 18:14-18:15 - Macro refresh (FRED/BEA/BLS, long TTL)
```

## Quota Packing Mathematics

### "8/min, 800/day" Example - Twelve Data
**Daily Budget Allocation:**
```python
TWELVE_DATA_DAILY_BUDGET = 800
ALLOCATION = {
    "intraday_bars": 400,      # 50% - Primary function  
    "indicators_remote": 300,  # 37.5% - RSI, MACD, EMA
    "price_fallback": 80,      # 10% - When yfinance fails
    "buffer_reserve": 20       # 2.5% - Error margin
}

# Pre-flight validation
def validate_daily_tasks():
    total_planned = sum(ALLOCATION.values())
    assert total_planned <= TWELVE_DATA_DAILY_BUDGET
```

**Minute Rate Distribution:**
```python
# 8/min limit across 15-minute window = 120 total calls max
MINUTE_PACING = {
    "minutes_1_3": 24,    # YF fallback period (8/min × 3)
    "minutes_4_8": 40,    # Intraday period (8/min × 5) 
    "minutes_9_12": 32,   # Indicator period (8/min × 4)
    "minutes_13_15": 24,  # Buffer/cleanup (8/min × 3)
}
```

### Provider Quota Matrix
| Provider | Per-Minute | Daily | Monthly | Reset Time | Priority |
|----------|------------|-------|---------|------------|----------|
| **YFinance** | No limit | No limit | No limit | N/A | 1st (Bulk) |  
| **Alpaca** | 200 | No limit | No limit | N/A | 2nd (US intraday) |
| **Finnhub** | 60 | ~1800 | 30k | Month end | 3rd (News) |
| **Twelve Data** | 8 | 800 | No limit | UTC midnight | 4th (Indicators) |
| **NewsAPI** | 1000 | 100 | No limit | UTC midnight | 5th (News breadth) |
| **Alpha Vantage** | 5 | 25 | No limit | 🟡 Unknown | Last (Fallback) |

## Reset Timing Verification

### ✅ Confirmed UTC Midnight Resets
- **Twelve Data**: Standard practice for daily quotas
- **NewsAPI**: Confirmed via `X-API-Key-Requests-Remaining` header  
- **Polygon, Tiingo, Marketstack**: Industry standard

### 🟡 Reset Probe Plan - Alpha Vantage
**Current Status**: `probes/probe_state.json` shows 72h probe scheduled Sept 1-3

**Probe Schedule**:
```json
{
  "day_1": {
    "pre_midnight": "2025-09-01T23:58:00Z",  
    "post_midnight": "2025-09-02T00:01:00Z",
    "budget": 2  
  },
  "day_2": {
    "pre_midnight": "2025-09-02T23:58:00Z",
    "post_midnight": "2025-09-03T00:01:00Z", 
    "budget": 2
  },
  "day_3": {
    "pre_midnight": "2025-09-03T23:58:00Z",
    "post_midnight": "2025-09-04T00:01:00Z",
    "budget": 2
  }
}
```

**Hypothesis Testing**:
- **H1**: UTC midnight reset (quota=25 after 00:01Z call)
- **H2**: Rolling 24h window (quota remains <25)  
- **Evidence Required**: 3 consecutive days for confidence

## Task Capping Strategy

### Pre-Flight Quota Calculation
```python
class DailyTaskCapper:
    def __init__(self):
        self.provider_limits = {
            "twelve_data": {"daily": 800, "minute": 8},
            "newsapi": {"daily": 100, "minute": 1000},  
            "alpha_vantage": {"daily": 25, "minute": 5},
            "finnhub": {"daily": 1800, "minute": 60}
        }
    
    def calculate_max_tasks(self, provider: str, universe_size: int) -> int:
        """Calculate maximum tasks without exceeding quotas"""
        limits = self.provider_limits[provider]
        
        # Factor in current usage  
        current_usage = self.get_current_usage(provider)
        remaining_daily = limits["daily"] - current_usage["daily"]
        
        # Apply safety buffer (10%)
        safe_daily = int(remaining_daily * 0.9)
        
        # Consider minute rate over 15-minute window
        minute_capacity = limits["minute"] * 15  # 15-minute window
        
        return min(safe_daily, minute_capacity, universe_size)
```

### Dynamic Priority Adjustment
```python
class TaskPrioritizer:
    PRIORITY_MATRIX = {
        "portfolio_prices": 1,      # Must have
        "watchlist_prices": 2,      # Important  
        "news_sentiment": 3,        # Nice to have
        "technical_indicators": 4,  # Optional
        "macro_data": 5            # Background
    }
    
    def allocate_quota(self, available_quota: int, tasks_by_priority: dict) -> dict:
        """Allocate quota starting with highest priority"""
        allocation = {}
        remaining = available_quota
        
        for priority in sorted(self.PRIORITY_MATRIX.values()):
            task_type = self._get_task_type_by_priority(priority)
            if task_type in tasks_by_priority and remaining > 0:
                requested = len(tasks_by_priority[task_type])
                allocated = min(requested, remaining)
                allocation[task_type] = allocated
                remaining -= allocated
                
        return allocation
```

## Cloud Scheduling Configuration

### GitHub Actions (Recommended)
```yaml
name: Daily Data Pipeline
on:
  schedule:
    - cron: "0 22 * * 1-5"  # 6 PM ET, weekdays only
  workflow_dispatch:        # Manual trigger

jobs:
  data_pipeline:
    runs-on: ubuntu-latest
    timeout-minutes: 20     # 5min buffer
    env:
      PIPELINE_MODE: "production"
      MAX_RUNTIME_MINUTES: 15
```

### Cloud Run Alternative  
```python
# Cloud Scheduler trigger
SCHEDULE_CONFIG = {
    "schedule": "0 22 * * 1-5",      # 6 PM ET weekdays
    "time_zone": "America/New_York",  
    "timeout": "15m",
    "retry_config": {
        "retry_count": 1,
        "min_backoff_duration": "300s"
    }
}
```

## Monitoring & Alerts

### Real-Time Monitoring
```python  
class PipelineMonitor:
    def __init__(self):
        self.alert_thresholds = {
            "runtime_minutes": 15,
            "quota_utilization": 0.95,  # 95% of daily limit
            "failure_rate": 0.1         # 10% task failure rate
        }
    
    def check_alerts(self, pipeline_stats: dict):
        if pipeline_stats["runtime"] > self.alert_thresholds["runtime_minutes"]:
            self.send_alert("TIMEOUT_RISK", pipeline_stats)
            
        for provider, usage in pipeline_stats["quota_usage"].items():
            if usage["utilization"] > self.alert_thresholds["quota_utilization"]:
                self.send_alert("QUOTA_EXHAUSTION", provider, usage)
```

### Success Criteria
- **Completion Time**: <15 minutes, 95% of runs
- **Quota Compliance**: Never exceed daily limits  
- **Data Coverage**: ≥90% of portfolio/watchlist symbols
- **Error Rate**: <5% task failure rate
- **Reset Timing**: Confirmed for all ambiguous providers

## Emergency Procedures

### Quota Exhaustion Response
1. **Immediate**: Switch to fallback providers if available
2. **Short-term**: Reduce universe size by priority
3. **Next-day**: Investigate cause and adjust allocations

### Pipeline Timeout Response  
1. **Graceful shutdown**: Complete in-flight requests
2. **State persistence**: Save partial results  
3. **Resume capability**: Continue from checkpoint next day

**Implementation Timeline**: Pipeline runner ready by Week 2, full monitoring by Week 3.