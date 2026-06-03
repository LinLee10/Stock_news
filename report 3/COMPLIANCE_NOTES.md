# Compliance Notes - ToS Analysis & Risk Posture

## Provider ToS Analysis Summary

### 🔴 Explicitly Prohibited Multi-Key Usage

#### NewsAPI - Terms of Service Section 2.1
> **"You may not... share your API key with others"**
> **"Each API key is licensed to a single person or organization"**

**Our Posture**: Single key enforcement, strict quota management
- Hard-coded single key, no rotation logic
- 100/day limit with 5-call safety buffer (95 calls max)
- Graceful degradation when quota exhausted

#### Twelve Data - Pricing FAQ
> **"One API key per user account"**
> **"Commercial use requires business plan"**

**Our Posture**: Conservative single-key usage
- 8/min, 800/day limits strictly enforced  
- Pre-flight task capping to prevent quota violations
- Free tier personal research use only

#### Alpha Vantage - Terms of Service
> **"API keys are for personal use only"**
> **"No redistribution or resale of data"**

**Our Posture**: Fallback role only, minimal usage
- Limited to ≤25/day emergency fallback scenarios
- Only triggered when primary sources (yfinance) fail
- Daily exhaustion guard prevents quota cycling

### ✅ Explicitly Allowed Multi-Key Usage

#### Finnhub - Developer Documentation
> **"You can create multiple applications with separate API keys"**
> **"Each application has its own quota allocation"**

**Our Strategy**: Research vs Production Separation
```python
FINNHUB_RESEARCH_KEY = "pk_research_..."  # Development/backtesting  
FINNHUB_PROD_KEY = "pk_production_..."    # Live trading signals
```
- **Justification**: Separate applications for different use cases
- **Partitioning**: Research calls vs production trading signals
- **Documentation**: Clear business purpose for each key

#### FRED - Federal Reserve Terms
> **"Multiple registrations allowed for different research projects"**
> **"Academic and commercial use permitted"**

**Our Strategy**: Dataset Category Separation  
```python
FRED_MACRO_KEY = "key_macro_..."     # GDP, inflation, employment
FRED_MONETARY_KEY = "key_fed_..."    # Interest rates, money supply
```
- **Justification**: Different research categories warrant separate registrations
- **Partitioning**: Economic analysis vs monetary policy research
- **Compliance**: Government data explicitly allows multiple academic uses

#### Alpaca - Account Structure
> **"Multiple paper trading accounts allowed"**
> **"Separate data-only vs trading access"**

**Our Strategy**: Account Type Separation
```python
ALPACA_PAPER_KEY = "PK_PAPER_..."    # Paper trading data
ALPACA_DATA_KEY = "PK_DATA_..."      # Market data only  
```
- **Justification**: Different account types serve different purposes
- **Usage**: Trading simulation vs pure market data access
- **ToS Compliance**: Explicitly supported account structure

### 🟡 Ambiguous Cases Requiring Clarification

#### Tiingo - "Personal Use" Boundary
**Terms**: *"For personal use and small-scale applications"*
**Ambiguity**: What constitutes "small-scale" vs commercial use?

**Vendor Question Template**:
> "For our financial research application analyzing ~100 stocks daily, we need to partition API usage between historical price data and news sentiment feeds. Does your ToS permit multiple API keys for the same research organization to achieve this data partitioning, or would this violate personal use restrictions?"

**Interim Posture**: Conservative single key until clarification

#### GNews - Application Scope  
**Terms**: *"One API key per application"*
**Ambiguity**: Can one organization have multiple applications?

**Vendor Question**:
> "Our organization develops separate applications for news research vs automated alerts. Would separate API keys for these distinct applications comply with your 'one key per application' policy?"

**Interim Posture**: Single key, treat as restrictive until confirmed

## Risk Assessment Matrix

| Provider | Multi-Key Risk | Quota Risk | Financial Risk | Detection Risk |
|----------|----------------|------------|----------------|----------------|
| **NewsAPI** | High | Medium | Low ($0) | Medium |
| **Twelve Data** | High | High | Low ($0) | High |  
| **Alpha Vantage** | Medium | Low | Low ($0) | Low |
| **Finnhub** | Low | Medium | Low ($0) | Low |
| **FRED/BEA/BLS** | Low | Low | Low ($0) | Low |
| **Alpaca** | Low | Low | Low ($0) | Low |

### Risk Definitions:
- **Multi-Key Risk**: Likelihood of ToS violation for multiple keys
- **Quota Risk**: Risk of accidental quota violation  
- **Financial Risk**: Potential monetary liability
- **Detection Risk**: Likelihood provider detects/cares about violations

## Compliance Implementation

### Technical Safeguards
```python
class ComplianceEnforcer:
    """Ensure API usage complies with provider ToS"""
    
    SINGLE_KEY_PROVIDERS = [
        "newsapi", "twelve_data", "alpha_vantage", 
        "polygon", "eodhd", "marketstack"
    ]
    
    MULTI_KEY_ALLOWED = [
        "finnhub", "fred", "bea", "bls", "alpaca", "nasdaq_datalink"
    ]
    
    def __init__(self):
        self.usage_ledger = UsageLedger()
        self.audit_logger = ComplianceAuditLogger()
    
    async def validate_key_usage(self, provider: str, key_id: str, context: RequestContext):
        """Validate API key usage for compliance"""
        
        if provider in self.SINGLE_KEY_PROVIDERS:
            # Ensure only one key is ever used
            known_key = await self.usage_ledger.get_known_key(provider)
            if known_key and known_key != key_id:
                raise ComplianceError(f"Multiple keys detected for single-key provider {provider}")
        
        elif provider in self.MULTI_KEY_ALLOWED:
            # Validate business justification exists
            purpose = context.get_purpose()
            if not purpose:
                raise ComplianceError(f"Multi-key provider {provider} requires business purpose")
            
            # Log usage with justification
            await self.audit_logger.log_key_usage(
                provider=provider,
                key_id=self.hash_key(key_id),  # Never log actual key
                purpose=purpose,
                timestamp=datetime.now(timezone.utc)
            )
        
        else:
            # Unknown provider - default to restrictive
            logger.warning(f"Unknown provider compliance posture: {provider}")
```

### Usage Monitoring
```python
class ComplianceMonitor:
    """Monitor for potential ToS violations"""
    
    async def detect_suspicious_patterns(self, provider: str) -> List[str]:
        """Detect usage patterns that may indicate violations"""
        violations = []
        
        # Check for quota cycling (using multiple keys to exceed limits)
        daily_usage = await self.get_daily_usage(provider)
        if provider in SINGLE_KEY_PROVIDERS and daily_usage > self.get_known_limit(provider):
            violations.append(f"Usage exceeds known single-key limit: {daily_usage}")
        
        # Check for rapid key switching
        key_switches = await self.count_key_switches(provider, hours=24)
        if key_switches > 1:
            violations.append(f"Multiple key switches in 24h: {key_switches}")
        
        # Check for simultaneous key usage
        concurrent_keys = await self.count_concurrent_keys(provider, minutes=60)
        if concurrent_keys > 1:
            violations.append(f"Concurrent key usage detected: {concurrent_keys}")
        
        return violations
```

## Vendor Communication Log

### Clarification Requests Sent
| Provider | Date Sent | Status | Response Summary |
|----------|-----------|--------|------------------|
| Tiingo | TBD | Pending | Multi-key for data partitioning? |
| GNews | TBD | Pending | Multiple applications same org? |
| Marketstack | TBD | Pending | Research vs production keys? |
| BLS | TBD | Pending | Academic vs commercial registration? |

### Template Email
```
Subject: API Usage Clarification - Multiple Keys for Data Partitioning

Dear [Provider] Support Team,

We are developing a financial research application that analyzes market data 
and news sentiment. To properly organize our API usage and stay within terms 
of service, we need clarification on your multi-key policy.

Our use case:
- Organization: Academic/personal research project  
- Scale: ~100 stock symbols analyzed daily
- Purpose: Separate historical data analysis from real-time alerts
- Technical need: Partition API quota between different data types

Question: Does your Terms of Service permit multiple API keys for the same 
organization when used for distinct technical purposes (data partitioning), 
or does this violate single-user restrictions?

We want to ensure full compliance with your terms and appreciate your 
guidance on structuring our API access appropriately.

Best regards,
[Research Team]
```

## Compliance Audit Trail

### Daily Compliance Report
```json
{
  "date": "2025-09-03",
  "compliance_status": "compliant",
  "violations_detected": 0,
  "provider_usage": {
    "newsapi": {
      "keys_used": 1,
      "max_allowed": 1,
      "usage": "87/100",
      "compliance_status": "compliant"
    },
    "finnhub": {
      "keys_used": 2, 
      "max_allowed": "unlimited",
      "purposes": ["research", "production"],
      "compliance_status": "compliant"
    }
  },
  "pending_clarifications": [
    {"provider": "tiingo", "days_pending": 5},
    {"provider": "gnews", "days_pending": 3}
  ],
  "risk_assessment": "low"
}
```

## Emergency Response Procedures

### ToS Violation Response
1. **Immediate**: Suspend all API usage for affected provider
2. **Within 1 hour**: Audit all recent usage for violations  
3. **Within 24 hours**: Contact provider to report and remediate
4. **Document**: All steps taken and provider responses

### Quota Violation Response  
1. **Immediate**: Activate fallback providers if available
2. **Log**: Exact quota violation and contributing factors
3. **Prevent**: Implement additional safeguards to prevent recurrence

**Review Schedule**: Monthly ToS review, quarterly compliance audit, annual policy updates based on provider changes.