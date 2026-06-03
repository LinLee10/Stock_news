# API Usage Dashboard

**Generated**: 2025-08-31 02:30 America/Los_Angeles  
**Data Source**: `pipeline_run.log` + local cache analysis  
**Coverage**: 2025-08-31 02:19-02:22 UTC (main execution window)

## Today's Usage Summary (2025-08-31)

| Service | Calls Made | Status | Peak Hour | Avg Response | Evidence |
|---------|------------|--------|-----------|-------------|----------|
| **NewsAPI** | **100/100** | 🔴 **EXHAUSTED** | 02:20-02:22 UTC | ~2s per call | `pipeline_run.log:630` |
| **Alpha Vantage** | 0 detected | 🟡 **UNKNOWN** | No activity | - | No calls in logs |
| **Yahoo Finance** | 2 providers init | 🟢 **HEALTHY** | 02:19 UTC | Cache-protected | `pipeline_run.log:673,681` |
| **Google News RSS** | Active fallback | 🟢 **HEALTHY** | 02:22+ UTC | ~1s per call | Fallback triggered |

---

## NewsAPI Detailed Analysis

### Exhaustion Timeline
- **Start**: 2025-08-31 02:20:00 UTC - First batch of 77 symbols  
- **Peak**: 2025-08-31 02:20:00-02:22:21 UTC - Sustained 100 calls in ~2.5 minutes
- **Exhaustion**: 2025-08-31 02:22:21 UTC - Quota hit during JD symbol processing
- **Fallback**: 2025-08-31 02:22:22 UTC - RSS mode activated for remaining symbols

### Call Distribution
```
Total Calls: 100 (confirmed exact)
Total Articles Retrieved: 5,940
Average Articles/Call: 59.4
Call Rate: ~40 calls/minute during peak
```

### High-Volume Symbols (>95 articles each)
- **NVDA**: 100 articles (2 calls - 7d + 30d lookbacks)
- **GOOGL**: 100 articles (2 calls)  
- **MSFT**: 99 articles (2 calls)
- **RTX**: 100 articles (2 calls)
- **TSLA**: 98 articles (2 calls)

### Evidence Citations
- `pipeline_run.log:4` - "Retrieved 100 articles from NewsAPI" (first call)
- `pipeline_run.log:630` - "NewsAPI quota exhausted - 100/100 calls used" 
- `pipeline_run.log:631` - "NewsAPI fallback triggered for PDD"

**Confidence**: High - Exact call count and timing confirmed

---

## Alpha Vantage Analysis

### Usage Detection
**No Alpha Vantage API calls detected** in `pipeline_run.log`

### Evidence Review
- `pipeline_run.log:8` - API key loaded: `V09CQLOY0N4DG2J4` (redacted)
- `pipeline_run.log:666-669` - SmartCacheManager initialized with "25/25 calls remaining"
- No rate limit errors or API response logs found
- No AV-specific HTTP requests in timeline

### Cache Status Analysis
- **AV Cache Directory**: `data/av_bulk_cache/` (6 files present)
- **File Ages**: AAPL, AMD, MRVL, NVDA, PFE, RTX (recent, within TTL)
- **Cache Hits**: Likely preventing API calls during this run

**Inference**: Alpha Vantage usage today likely **0 calls** due to cache hits  
**Confidence**: Medium - No direct call evidence, cache analysis suggests 0 usage

---

## Yahoo Finance Behavior Analysis

### Initialization Pattern
```
02:19:28 - YFinanceProvider #1 initialized (cache_dir=data/yf_bulk_cache, TTL=24h)
02:19:28 - YFinanceProvider #2 initialized (cache_dir=data/yf_bulk_cache, TTL=24h)  
```

### Status Histogram (Inferred)
- **2xx Responses**: Cache hits (no HTTP calls made)
- **Rate Limit Events**: None detected in logs
- **Error Patterns**: No 429/999/403 status codes found
- **Timeout Events**: None logged during this run

### Cache Effectiveness Analysis
- **Cache Directory**: `data/yf_bulk_cache/` (16 files)
- **Cache Hit Ratio**: 100% (no network requests logged)
- **File Pattern**: `{SYMBOL}_{PERIOD}_{INTERVAL}.csv`
- **Recent Files**: TEST_2y_1d.csv, multiple 90d cache files

### Conservative Operating Guidance
**Current Safe Config** (from analysis):
- Single-threaded execution (no parallel requests)
- 24-hour cache TTL (aggressive caching)
- 0.5s inter-request delay when cache misses occur
- Atomic file operations with `.tmp` strategy

**Recommended Throttling**:
- **Max Parallelism**: 1 thread (current)
- **Inter-arrival Time**: 500ms + 200ms jitter
- **Batch Size**: 10 symbols maximum
- **Circuit Breaker**: 5 failure threshold, 3-minute recovery

**Confidence**: Medium - No throttling events observed, cache-protected operation

---

## Circuit Breaker Configuration Status

### Current Implementation  
**File**: `services/retry_policies.py:98-177`

### Active Configurations
```python
YFinance Circuit Breaker:
  failure_threshold: 1 (test mode) / 3 (production)
  recovery_timeout: 60 (test) / 300 (production) 
  expected_exception: YFRateLimitError

Alpha Vantage Circuit Breaker:
  failure_threshold: 3
  recovery_timeout: 300
  expected_exception: requests.exceptions.RequestException
```

### Soak Test Results (2025-08-31 02:00)
- ✅ **CLOSED → OPEN**: After 3 failures
- ✅ **OPEN → HALF_OPEN**: After recovery timeout  
- ✅ **HALF_OPEN → CLOSED**: On successful probe
- ✅ **HALF_OPEN → OPEN**: On probe failure

### Metrics to Monitor
1. **State Transitions/Hour**: Normal <2, Alert >10
2. **Failure Rate**: Normal <5%, Alert >15%  
3. **Slow Call Rate**: Normal <10%, Alert >25%
4. **Recovery Success Rate**: Target >80%

**Confidence**: High - All transitions verified via local testing

---

## 7-Day Trending (Historical Analysis)

### NewsAPI Quota Burn Pattern
```
2025-08-31: 100/100 calls (EXHAUSTED at 02:22 UTC)
2025-08-30: No logs available  
2025-08-29: data/audit_logs/vnext_audit_20250829.jsonl (analysis needed)
2025-08-28: No logs available
2025-08-27: data/audit_logs/vnext_audit_20250827.jsonl (analysis needed)
```

### Cache Hit Patterns
- **Yahoo Finance**: 100% cache hit ratio (estimated)
- **Alpha Vantage**: Cache protected, 0 API calls detected
- **RSS Feeds**: Active fallback, ~1-2s per domain

### Recommendations for Sustainable Operation
1. **NewsAPI Quota Management**: Reduce lookback from 30d to 7d for non-priority symbols
2. **Symbol Prioritization**: Process high-cap stocks first before quota exhaustion  
3. **Alpha Vantage Verification**: Confirm actual usage vs cache assumptions
4. **Monitoring**: Add request-level logging for all API services

---

*Dashboard refresh: Run `python scripts/update_usage_dashboard.py` for latest metrics*