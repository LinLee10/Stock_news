# Observability Strategy - Monitoring & Logging

## Current Logging Infrastructure

### ✅ Existing Implementation
**Location**: `services/audit_logger.py:36-100`

```python
@dataclass
class AuditLogEntry:
    timestamp: str
    log_id: str
    feature: str          # symbol_intake, news_clustering, recommendations
    operation: str        # intake_job, clustering_job, generate_recs
    symbol: Optional[str]
    step: str            # validate, fetch_data, analyze, persist
    duration_ms: Optional[int]
    count_in: Optional[int]   # Input records/items
    count_out: Optional[int]  # Output records/items
    source_links: List[str]   # URLs or file paths accessed
    status: str          # started, completed, failed, warning
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
```

**Daily JSONL Output**: `data/audit_logs/vnext_audit_YYYYMMDD.jsonl`

## Enhanced JSONL Event Schema

### API Call Events
```json
{
  "timestamp": "2025-09-03T18:05:23.456Z",
  "event_type": "api_call",
  "log_id": "req_20250903_180523_001", 
  "provider": "finnhub",
  "endpoint": "/company-news",
  "symbol": "AAPL",
  "method": "GET",
  "status_code": 200,
  "duration_ms": 342,
  "quota_used": 1,
  "quota_remaining": 1799,
  "rate_limit_window": "minute",
  "response_size_bytes": 15420,
  "success": true,
  "metadata": {
    "request_priority": "portfolio",
    "cache_hit": false,
    "retry_count": 0
  }
}
```

### Rate Limiting Events
```json
{
  "timestamp": "2025-09-03T18:12:45.123Z",
  "event_type": "rate_limited",
  "log_id": "rate_20250903_181245_001",
  "provider": "twelve_data", 
  "limit_type": "minute",  // "minute", "daily", "monthly"
  "limit_value": 8,
  "current_usage": 8,
  "wait_time_seconds": 23,
  "backoff_strategy": "token_bucket",
  "affected_symbol": "NVDA",
  "queue_depth": 15,
  "metadata": {
    "bucket_refill_rate": "8_per_minute",
    "next_refill_at": "2025-09-03T18:13:00.000Z"
  }
}
```

### Fallback Events  
```json
{
  "timestamp": "2025-09-03T18:15:12.789Z",
  "event_type": "fallback",
  "log_id": "fallback_20250903_181512_001",
  "from_provider": "newsapi",
  "to_provider": "gnews", 
  "reason": "quota_exhausted",
  "symbol": "TSLA",
  "operation": "news_fetch",
  "fallback_success": true,
  "primary_error": "Daily quota exhausted (100/100)",
  "fallback_duration_ms": 876,
  "metadata": {
    "primary_attempts": 1,
    "fallback_quota_remaining": 87
  }
}
```

### Deduplication Events
```json
{
  "timestamp": "2025-09-03T18:20:33.456Z", 
  "event_type": "dedup",
  "log_id": "dedup_20250903_182033_001",
  "data_type": "news_article", // "news_article", "price_data"
  "dedup_method": "content_hash", // "content_hash", "url_canonical", "similarity"
  "duplicates_found": 3,
  "unique_retained": 1,
  "merge_strategy": "provider_precedence",
  "providers_merged": ["newsapi", "gnews", "finnhub"],
  "content_hash": "sha256:a1b2c3...",
  "canonical_url": "https://example.com/article",
  "metadata": {
    "similarity_threshold": 0.85,
    "processing_time_ms": 45
  }
}
```

### Error Events
```json
{
  "timestamp": "2025-09-03T18:25:10.234Z",
  "event_type": "error", 
  "log_id": "error_20250903_182510_001",
  "provider": "alpha_vantage",
  "error_type": "network_timeout", // "rate_limit", "auth_error", "parse_error", "network_timeout"
  "error_message": "Request timeout after 15 seconds",
  "symbol": "AMD",
  "operation": "daily_price_fetch", 
  "retry_count": 2,
  "will_retry": false,
  "backoff_until": null,
  "metadata": {
    "http_status_code": null,
    "timeout_seconds": 15,
    "circuit_breaker_state": "closed"
  }
}
```

## Per-Provider Counter Tracking

### Real-Time Counters (Redis)
```python
class ProviderCounters:
    """Real-time usage tracking per provider"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def increment_usage(self, provider: str, window: str = "minute"):
        """Increment usage counter"""
        now = datetime.now(timezone.utc)
        
        if window == "minute":
            key = f"usage:{provider}:minute:{now.strftime('%Y%m%d_%H%M')}"
            ttl = 120  # 2 minutes TTL
        elif window == "daily": 
            key = f"usage:{provider}:daily:{now.strftime('%Y%m%d')}"
            ttl = 86400 * 2  # 2 days TTL
        elif window == "monthly":
            key = f"usage:{provider}:monthly:{now.strftime('%Y%m')}" 
            ttl = 86400 * 35  # 35 days TTL
            
        await self.redis.incr(key)
        await self.redis.expire(key, ttl)
    
    async def get_usage(self, provider: str, window: str = "minute") -> int:
        """Get current usage count"""
        now = datetime.now(timezone.utc)
        
        if window == "minute":
            key = f"usage:{provider}:minute:{now.strftime('%Y%m%d_%H%M')}"
        elif window == "daily":
            key = f"usage:{provider}:daily:{now.strftime('%Y%m%d')}"
        elif window == "monthly":
            key = f"usage:{provider}:monthly:{now.strftime('%Y%m')}"
            
        count = await self.redis.get(key)
        return int(count) if count else 0
```

### Daily Usage Rollups
```python
class UsageRollups:
    """Generate daily summary statistics"""
    
    async def generate_daily_summary(self, date: str) -> dict:
        """Generate comprehensive daily usage summary"""
        log_file = f"data/audit_logs/vnext_audit_{date.replace('-', '')}.jsonl"
        
        summary = {
            "date": date,
            "providers": {},
            "operations": {},
            "errors": {},
            "performance": {}
        }
        
        # Parse JSONL logs
        with open(log_file, 'r') as f:
            for line in f:
                event = json.loads(line.strip())
                
                # Provider usage
                if event["event_type"] == "api_call":
                    provider = event["provider"]
                    if provider not in summary["providers"]:
                        summary["providers"][provider] = {
                            "total_calls": 0,
                            "successful_calls": 0,
                            "failed_calls": 0,
                            "avg_duration_ms": 0,
                            "total_quota_used": 0,
                            "rate_limited_count": 0
                        }
                    
                    summary["providers"][provider]["total_calls"] += 1
                    if event["success"]:
                        summary["providers"][provider]["successful_calls"] += 1
                    else:
                        summary["providers"][provider]["failed_calls"] += 1
                        
                    summary["providers"][provider]["total_quota_used"] += event.get("quota_used", 0)
                
                # Rate limiting
                elif event["event_type"] == "rate_limited":
                    provider = event["provider"]
                    if provider in summary["providers"]:
                        summary["providers"][provider]["rate_limited_count"] += 1
                
                # Errors
                elif event["event_type"] == "error":
                    error_type = event["error_type"]
                    if error_type not in summary["errors"]:
                        summary["errors"][error_type] = 0
                    summary["errors"][error_type] += 1
        
        return summary
```

## Alert Thresholds & Monitoring

### Quota Utilization Alerts
```python
class QuotaAlerts:
    """Monitor quota usage and trigger alerts"""
    
    ALERT_THRESHOLDS = {
        "twelve_data": {
            "daily_quota": 800,
            "warning_threshold": 0.8,   # 640 calls
            "critical_threshold": 0.95  # 760 calls  
        },
        "newsapi": {
            "daily_quota": 100,
            "warning_threshold": 0.85,  # 85 calls
            "critical_threshold": 0.95  # 95 calls
        },
        "finnhub": {
            "daily_quota": 1800,
            "warning_threshold": 0.8,   # 1440 calls
            "critical_threshold": 0.9   # 1620 calls
        }
    }
    
    async def check_quota_alerts(self, provider: str, current_usage: int):
        """Check if quota usage triggers alerts"""
        if provider not in self.ALERT_THRESHOLDS:
            return
            
        config = self.ALERT_THRESHOLDS[provider]
        quota = config["daily_quota"]
        utilization = current_usage / quota
        
        if utilization >= config["critical_threshold"]:
            await self.send_alert(
                level="CRITICAL", 
                message=f"{provider} quota critically low: {current_usage}/{quota} ({utilization:.1%})",
                provider=provider,
                usage=current_usage,
                quota=quota
            )
        elif utilization >= config["warning_threshold"]:
            await self.send_alert(
                level="WARNING",
                message=f"{provider} quota warning: {current_usage}/{quota} ({utilization:.1%})",
                provider=provider,
                usage=current_usage, 
                quota=quota
            )
```

### Performance Monitoring
```python
class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.response_times = {}
        self.error_rates = {}
    
    async def track_response_time(self, provider: str, duration_ms: int):
        """Track response time for performance analysis"""
        if provider not in self.response_times:
            self.response_times[provider] = []
            
        self.response_times[provider].append(duration_ms)
        
        # Keep only last 100 measurements
        if len(self.response_times[provider]) > 100:
            self.response_times[provider] = self.response_times[provider][-100:]
    
    async def get_performance_stats(self, provider: str) -> dict:
        """Get performance statistics"""
        if provider not in self.response_times:
            return {}
            
        times = self.response_times[provider]
        return {
            "avg_response_time_ms": sum(times) / len(times),
            "min_response_time_ms": min(times),
            "max_response_time_ms": max(times),
            "p95_response_time_ms": sorted(times)[int(len(times) * 0.95)],
            "sample_count": len(times)
        }
```

## Log Retention Policy

### Retention Schedule
- **JSONL Logs**: 30 days rolling retention
- **Redis Counters**: Auto-expire based on window (2min for minute counters, 2 days for daily)
- **Summary Reports**: 90 days retention  
- **Error Logs**: 60 days retention

### Auto-Cleanup Implementation
```python
class LogRetention:
    """Manage log file retention and cleanup"""
    
    def __init__(self, logs_dir: Path = Path("data/audit_logs")):
        self.logs_dir = logs_dir
    
    async def cleanup_old_logs(self):
        """Remove logs older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for log_file in self.logs_dir.glob("vnext_audit_*.jsonl"):
            try:
                # Extract date from filename  
                date_str = log_file.stem.split('_')[-1]  # vnext_audit_20250903.jsonl
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")
                    
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse log file date: {log_file}, error: {e}")
```

## Monitoring Dashboard Schema

### Real-Time Metrics
```json
{
  "timestamp": "2025-09-03T18:30:00Z",
  "system_health": "green", // "green", "yellow", "red"  
  "active_providers": 8,
  "total_requests_last_hour": 450,
  "avg_response_time_ms": 285,
  "error_rate_percent": 2.1,
  "providers": {
    "yfinance": {
      "status": "healthy",
      "requests_today": 0,
      "quota_utilization": "0%",
      "avg_response_time_ms": 120,
      "last_success": "2025-09-03T18:25:00Z"
    },
    "finnhub": {
      "status": "healthy", 
      "requests_today": 145,
      "quota_utilization": "8%",
      "rate_limit_hits": 0,
      "avg_response_time_ms": 340,
      "last_success": "2025-09-03T18:29:45Z"
    },
    "newsapi": {
      "status": "quota_exhausted",
      "requests_today": 100,
      "quota_utilization": "100%", 
      "last_success": "2025-09-03T14:22:15Z",
      "reset_time": "2025-09-04T00:00:00Z"
    }
  }
}
```

## Implementation Timeline

### Week 1: Enhanced Logging
- Extend `AuditLogger` with new event types
- Add Redis counter tracking
- Implement basic alert thresholds

### Week 2: Performance Monitoring  
- Add response time tracking
- Create performance statistics aggregation
- Implement dashboard data generation

### Week 3: Production Monitoring
- Deploy monitoring infrastructure
- Set up real-time alerting
- Create operational runbooks

**Log Retention**: Minimum 3 days, recommended 30 days with auto-cleanup for operational efficiency.