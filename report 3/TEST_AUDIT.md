# Test Audit - Coverage Analysis

## Current Test Inventory

### Unit Tests ✅ Present
- **Rate Limiting Comprehensive** - `tests/unit/test_rate_limiting_comprehensive.py`
  - Token bucket mechanics
  - Circuit breaker state transitions  
  - Exponential backoff with jitter

- **Symbol Intake** - `tests/unit/test_symbol_intake.py`
  - Validation regex patterns
  - Deduplication logic
  - Idempotent upserts

- **Feature Flags** - `tests/unit/test_feature_flags_comprehensive.py` 
  - Boolean environment parsing
  - Runtime override capability
  - Default value handling

- **API Disclaimer** - `tests/unit/test_api_disclaimer.py`
  - Template rendering
  - Email formatting

- **Redaction** - `tests/unit/test_redact.py`
  - PII detection and masking
  - API key sanitization

### Integration Tests ✅ Present  
- **System Integration** - `tests/test_system_integration_working.py`
- **Core Integration Fast** - `tests/test_core_integration_fast.py`
- **Correlation Analysis** - `tests/integration/test_correlation.py`
- **E2E Pipeline Regression** - `tests/integration/test_e2e_pipeline_regression.py`

## Critical Test Gaps 🔴 

### Alpha Vantage Infinite Loop
**Priority: URGENT**
```python
# MISSING: tests/unit/test_alpha_vantage_loop.py
def test_backoff_request_not_immediately_requeued():
    """Test that requests with backoff don't spin in tight loop"""
    
def test_daily_quota_exhausted_stops_enqueuing():
    """Test that daily exhaustion sets closed-for-day flag"""
    
def test_429_response_triggers_bounded_wait():
    """Test that 429/Note responses don't cause CPU spin"""
```

### Provider Rate Limit Edge Cases  
**Priority: HIGH**
```python  
# MISSING: tests/unit/test_provider_rate_limits.py
def test_minute_quota_exhaustion_blocks_correctly():
    """Verify minute buckets prevent over-calling"""
    
def test_daily_quota_exhaustion_graceful_fallback():
    """Verify daily exhaustion triggers fallback chain"""
    
def test_429_burst_handling():
    """Test handling of rapid 429 responses"""
    
def test_network_timeout_vs_rate_limit_classification():
    """Ensure proper error classification"""
```

### Fallback Chain Correctness
**Priority: HIGH**  
```python
# MISSING: tests/integration/test_fallback_chains.py
def test_yfinance_to_alpha_vantage_fallback():
    """Test price data fallback when yfinance fails"""
    
def test_newsapi_to_gnews_fallback():
    """Test news fallback when NewsAPI quota exhausted"""
    
def test_alpaca_to_twelve_data_intraday_fallback():
    """Test intraday fallback for non-US tickers"""
```

### Deduplication Effectiveness
**Priority: MEDIUM**
```python
# MISSING: tests/unit/test_deduplication.py
def test_content_hash_collision_handling():
    """Test handling of hash collisions"""
    
def test_url_canonicalization():
    """Test URL normalization for dedup"""
    
def test_cross_provider_dedup():
    """Test dedup across different news sources"""
```

### Quota Packing Mathematics  
**Priority: MEDIUM**
```python
# MISSING: tests/unit/test_quota_packing.py
def test_daily_task_cap_calculation():
    """Verify task counts don't exceed quotas"""
    
def test_provider_priority_scheduling():
    """Test high-priority providers get quota first"""
    
def test_quota_exhaustion_prevents_new_tasks():
    """Test pre-flight quota checks"""
```

## Mock/Fake Infrastructure Gaps

### API Response Fakes
**Priority: HIGH**
- **Finnhub Mock Responses** - Need realistic company news, social sentiment JSON
- **Twelve Data Mock Responses** - Need OHLCV + indicator response structures  
- **429 Response Simulators** - Need rate limit error generators for all providers

### Network Condition Simulation
**Priority: MEDIUM**
- **Timeout Simulation** - Need configurable network delays
- **Partial Response Handling** - Need truncated JSON scenarios
- **Connection Drops** - Need mid-request failure testing

## Performance Test Coverage  

### Load Testing ❌ Missing
```python
# MISSING: tests/performance/test_concurrent_requests.py
def test_100_concurrent_symbol_requests():
    """Verify system handles burst loads"""
    
def test_rate_limiter_under_pressure():  
    """Test token buckets under concurrent access"""
```

### Memory Leak Detection ❌ Missing
```python
# MISSING: tests/performance/test_memory_usage.py  
def test_long_running_worker_memory_stability():
    """Verify no memory leaks in Alpha Vantage worker"""
    
def test_cache_size_bounds():
    """Verify caches don't grow unbounded"""
```

## Test Prioritization Matrix

| Test Category | Priority | Effort | Risk if Missing |
|---------------|----------|--------|-----------------|
| Alpha Vantage Loop Fix | 🔴 Critical | Low | System hang/high CPU |
| Rate Limit Edge Cases | 🔴 High | Medium | Quota violations |
| Fallback Chain Logic | 🔴 High | Medium | Service degradation |  
| Provider Integration | 🟡 Medium | High | Limited functionality |
| Performance/Load | 🟡 Medium | High | Scalability issues |
| Dedup Effectiveness | 🟠 Low | Medium | Duplicate data |

## Recommended Test Implementation Order

### Week 1 - Critical Fixes
1. **Alpha Vantage Loop Tests** - Prevent infinite loop regression
2. **Rate Limit Boundary Tests** - Ensure quota compliance
3. **429 Response Handling** - Verify graceful degradation

### Week 2 - Integration Coverage  
1. **Fallback Chain Tests** - Multi-provider scenarios
2. **New Provider Integration Tests** - Finnhub, Twelve Data basic flows
3. **Quota Packing Validation** - Daily task limits

### Week 3 - Edge Cases
1. **Network Condition Tests** - Timeouts, drops, partial responses  
2. **Deduplication Tests** - Hash collisions, URL canonicalization
3. **Long-running Stability** - Memory leaks, worker health

## Test Infrastructure Needs

### Fixtures Required
- **Provider Response Fixtures** - JSON samples for each API
- **Rate Limit Scenarios** - 429 responses with proper headers  
- **Network Condition Mocks** - Timeout/failure simulators

### Test Database
- **Isolated Test DB** - SQLite in-memory for unit tests
- **Redis Test Instance** - Separate keyspace for integration tests
- **Fixture Data** - Known-good OHLCV/news datasets

**Total Test Gap**: ~35 missing test files, estimated 2-3 weeks to achieve 90% critical path coverage.