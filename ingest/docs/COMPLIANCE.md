# Compliance Documentation

## Overview

This system is designed to operate in full compliance with web standards, terms of service, and ethical scraping practices. **No evasion or circumvention techniques are used.**

## Compliance Principles

### 1. Robots.txt Respect
- All robots.txt files are fetched and parsed before any requests
- Disallowed paths are strictly respected
- Crawl-delay directives are honored with a minimum 12-second floor
- User-agent identification is honest and descriptive

### 2. Rate Limiting
- Minimum 12 seconds between requests to any domain
- Per-domain concurrency limits (default: 1 concurrent request)
- Global concurrency caps to prevent overwhelming any infrastructure
- Token bucket algorithm with configurable refill rates

### 3. Circuit Breaker Protection
- Automatic domain blocking after repeated failures
- Exponential backoff with jitter for retry attempts
- Graceful degradation when services are unavailable
- Recovery mechanisms with conservative retry schedules

### 4. Paywall Respect
- **No paywall circumvention attempts**
- Paywall detection flags content appropriately
- Paywalled content is not stored or processed
- Alternative access methods (APIs, licensed access) are preferred

### 5. Terms of Service Compliance
- No fingerprint spoofing or behavioral mimicking
- Standard browser identification without stealth techniques
- Respectful request patterns that don't overwhelm servers
- Contact information provided in User-Agent and From headers

## Technical Implementation

### Robots.txt Handling
```python
# Example robots.txt compliance check
async def can_fetch(self, url: str, user_agent: str) -> bool:
    robots_url = f"https://{domain}/robots.txt"
    # Fetch, parse, and check permissions
    return rp.can_fetch(user_agent, url)
```

### Rate Limiting Implementation
```python
# Token bucket with minimum delays
async def acquire(self, domain: str) -> bool:
    # Check token availability
    # Apply minimum 12-second delay
    # Update next allowed time
```

### User-Agent Policy
```
User-Agent: FinancialNewsBot/1.0 (Educational Research; +https://example.com/contact)
From: bot@example.com
```

## Domain Policies

### Tier 1: Premium Financial Sources
- **Reuters, Bloomberg, WSJ**: 15-25 second delays, single concurrency
- **Preferred access**: Official APIs where available
- **Fallback**: Respectful scraping with extended delays

### Tier 2: Business Publications
- **CNBC, MarketWatch**: 12-15 second delays
- **Strategy**: Mix of API and compliant scraping

### Tier 3: General Financial Content
- **Yahoo Finance, Seeking Alpha**: 12 second minimum delays
- **Higher concurrency**: Up to 2 concurrent where appropriate

### Restricted Domains
- Domains with explicit bot restrictions are marked as `allowed: false`
- No access attempts are made to restricted domains
- Alternative sources are used instead

## Error Handling

### Respectful Failure Modes
1. **403/429 Responses**: Immediate backoff, circuit breaker activation
2. **Rate Limiting**: Extended delays, no retry flooding
3. **Server Errors**: Exponential backoff with jitter
4. **Timeouts**: Graceful degradation, alternative sources

### Circuit Breaker States
- **Closed**: Normal operation
- **Open**: Domain blocked after 5+ failures
- **Half-Open**: Limited retry attempts after cooldown

## Monitoring and Alerts

### Compliance Metrics
- Robots.txt violation attempts (should be 0)
- Rate limit adherence (>99% compliance)
- Circuit breaker activations
- Error rate per domain

### Alert Conditions
- High error rates indicating potential overload
- Repeated policy violations
- Unusual traffic patterns

## Audit Trail

All requests are logged with:
- Timestamp and duration
- Compliance checks performed
- Rate limiting status
- Success/failure reasons
- Circuit breaker state changes

## Legal Considerations

### Fair Use
- Content is processed for financial analysis and research
- No republication or commercial redistribution
- Transformation into structured data for analysis

### Contact and Takedowns
- Contact information provided in all requests
- Responsive to takedown requests
- Proactive monitoring of terms of service changes

## Configuration Examples

### Conservative Configuration
```yaml
rate_limiting:
  default_delay_seconds: 20.0  # Extra conservative
  max_concurrent_per_domain: 1
  global_max_concurrent: 3
```

### Research Institution Configuration
```yaml
browser:
  user_agent: "AcademicResearchBot/1.0 (University Study; +https://university.edu/contact)"
  contact_email: "research@university.edu"
```

## Compliance Checklist

- [ ] Robots.txt respected for all domains
- [ ] Minimum delays enforced (≥12 seconds)
- [ ] No stealth or evasion techniques
- [ ] Honest user-agent identification  
- [ ] Paywall detection without circumvention
- [ ] Circuit breaker protection active
- [ ] Error logging and monitoring in place
- [ ] Contact information provided
- [ ] Terms of service reviewed for major domains

## Updates and Maintenance

This compliance framework is reviewed quarterly to ensure:
- New regulations are incorporated
- Domain policies remain current
- Rate limits are appropriately conservative
- Technical implementations follow best practices

**Last Updated**: 2024-12-XX
**Next Review**: 2025-03-XX