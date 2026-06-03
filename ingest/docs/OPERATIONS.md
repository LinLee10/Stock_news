# Operations Guide

## Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
python -m playwright install chromium

# Start Redis (required)
redis-server

# Optional: Start PostgreSQL
# Configure connection in config.yaml
```

### Basic Usage
```bash
# Enqueue URLs for processing
python -m ingest.main enqueue --urls https://example.com/article1 https://example.com/article2

# Start workers
python -m ingest.main worker --workers 3

# Check status
python -m ingest.main status

# Run demo
python -m ingest.main demo
```

## System Architecture

### Components
1. **Job Queue** (Redis): Manages ingestion jobs
2. **Workers**: Process jobs asynchronously  
3. **Browser Session**: Handles compliant web browsing
4. **Content Extractor**: Parses articles from HTML
5. **Quality Filter**: Scores and filters content
6. **Storage**: Redis (cache) + PostgreSQL (optional persistence)

### Data Flow
```
URLs → Job Queue → Workers → Browser → Extractor → Quality Filter → Storage
```

## Configuration

### Main Config File: `ingest/config.yaml`

#### Redis Configuration
```yaml
redis:
  url: "redis://localhost:6379/0"
```

#### Browser Settings
```yaml
browser:
  user_agent: "YourBot/1.0 (Purpose; +contact-url)"
  contact_email: "your-email@domain.com"
  headless: true
  timeout_seconds: 30
```

#### Rate Limiting (Critical for Compliance)
```yaml
rate_limiting:
  default_delay_seconds: 12.0  # Minimum delay between requests
  default_tokens_per_period: 10
  period_seconds: 600
  max_concurrent_per_domain: 1
  global_max_concurrent: 5
```

#### Domain Policies
```yaml
domains:
  example.com:
    allowed: true
    crawl_delay_seconds: 15.0
    max_concurrency: 1
    credibility_score: 0.8
    preferred_strategy: "playwright"  # or "aiohttp"
```

## Monitoring

### Key Metrics
- **Jobs processed/hour**: Throughput indicator
- **Success rate**: % of successful extractions
- **Average processing time**: Performance metric
- **Queue depths**: Backlog monitoring
- **Circuit breaker states**: Health indicator

### Status Commands
```bash
# Real-time status
python -m ingest.main status

# Queue statistics in logs
# Jobs processed: 150
# Jobs successful: 142  
# Jobs failed: 8
# Queue Status:
#   pending: 25
#   inflight: 3
#   deadletter: 2
```

### Log Analysis
Structured JSON logs include:
- `job_id`: Unique identifier
- `url`: Target URL
- `domain`: Domain being accessed
- `success`: Boolean outcome
- `processing_time_ms`: Performance data
- `error_message`: Failure details
- `strategy_used`: Fetch method (playwright/aiohttp)

## Performance Tuning

### Worker Scaling
```bash
# Scale up for higher throughput (respect rate limits!)
python -m ingest.main worker --workers 5

# Scale down for resource conservation
python -m ingest.main worker --workers 1
```

### Memory Management
- Each worker uses ~100-200MB (Playwright browser)
- Redis memory usage scales with queue size
- Monitor memory usage: `docker stats` or `htop`

### Browser Optimization
```yaml
# In config.yaml - reduce resource usage
browser:
  headless: true  # Always true for production
  timeout_seconds: 20  # Reduce from 30 for faster failures
```

## Error Handling

### Common Issues

#### 1. Redis Connection Failed
```
ERROR: Redis connection failed
```
**Solution**: Ensure Redis is running: `redis-server`

#### 2. Circuit Breaker Open
```
ERROR: Circuit breaker open for domain
```
**Solution**: Wait for recovery period (5 minutes default) or restart workers

#### 3. Rate Limit Exceeded
```
WARNING: Rate limit exceeded, domain: example.com
```
**Solution**: Normal behavior - system will wait and retry

#### 4. Robots.txt Blocked
```
INFO: Blocked by robots.txt, url: https://example.com/path
```  
**Solution**: Remove URL from queue or check robots.txt compliance

### Recovery Procedures

#### Restart Workers
```bash
# Stop current workers (Ctrl+C)
# Check queue status
python -m ingest.main status

# Restart with appropriate worker count
python -m ingest.main worker --workers 3
```

#### Clear Dead Letter Queue
```bash
# Manual Redis commands to inspect/clear
redis-cli
> LLEN jobs:deadletter
> LRANGE jobs:deadletter 0 -1  # View failed jobs
> DEL jobs:deadletter          # Clear if needed
```

#### Reset Circuit Breakers
```bash
# Clear circuit breaker state for specific domain
redis-cli
> DEL cb:problematic-domain.com
```

## Maintenance

### Daily Tasks
1. Check queue depths: `python -m ingest.main status`
2. Review error logs for patterns
3. Monitor disk space (logs, Redis persistence)

### Weekly Tasks  
1. Review domain policies for any ToS changes
2. Update robots.txt cache: Redis keys `robots:*`
3. Analyze success rates per domain
4. Check for dead letter queue buildup

### Monthly Tasks
1. Review and update domain credibility scores
2. Analyze duplicate detection accuracy
3. Update financial keyword lists
4. Performance benchmarking

## Deployment

### Docker Deployment
```dockerfile
# Dockerfile example
FROM python:3.11

RUN apt-get update && apt-get install -y \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m playwright install chromium

COPY ingest/ ./ingest/
CMD ["python", "-m", "ingest.main", "worker", "--workers", "3"]
```

### Production Checklist
- [ ] Redis persistence enabled
- [ ] Log rotation configured
- [ ] Resource limits set (memory, CPU)
- [ ] Monitoring alerts configured
- [ ] Backup strategy for data
- [ ] Rate limits tested and validated

## Troubleshooting

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python -m ingest.main worker --workers 1
```

### Performance Debugging
```bash
# Monitor Redis operations
redis-cli monitor

# Check browser resources
# Look for memory leaks, unclosed pages
```

### Network Issues
```bash
# Test connectivity to target domains
curl -I https://target-domain.com

# Check DNS resolution
nslookup target-domain.com

# Verify robots.txt access  
curl https://target-domain.com/robots.txt
```

## Security

### Access Control
- Redis should not be exposed publicly
- Use Redis AUTH if on shared infrastructure
- Rotate contact email credentials regularly

### Data Protection
- Article content may contain sensitive information
- Implement data retention policies
- Consider encryption for stored content

### Compliance Monitoring
- Regular audits of robots.txt compliance
- Monitor for unexpected behavior patterns
- Log analysis for policy violations

## Support

### Common Commands Reference
```bash
# Queue management
python -m ingest.main enqueue --urls URL1 URL2
python -m ingest.main status

# Worker management  
python -m ingest.main worker --workers N
# Ctrl+C to graceful shutdown

# Configuration validation
python -c "import yaml; yaml.safe_load(open('ingest/config.yaml'))"

# Redis inspection
redis-cli
> KEYS jobs:*
> LLEN jobs:pending
```

### Logs Location
- Application logs: stdout/stderr (capture with your logging system)
- Redis logs: typically `/var/log/redis/`
- Browser logs: embedded in application logs

For additional support, see the compliance documentation and test examples.