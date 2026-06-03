# Multi-Key Strategy - Compliant API Management

## Compliance Framework

### Core Principles
1. **No Evasive Rotation** - Keys represent distinct logical workloads, not quota inflation
2. **Transparent Usage Partitioning** - Each key serves a clearly defined purpose  
3. **Provider ToS Compliance** - Respect explicit multi-key restrictions
4. **Audit Trail** - Full usage tracking per key for compliance verification

## Provider-Specific Strategies

### ✅ Multi-Key Allowed (Explicit Permission)

#### Finnhub - Research & Production Separation
```python
FINNHUB_RESEARCH_KEY = "pk_research_..."  # Development/backtesting
FINNHUB_PROD_KEY = "pk_production_..."    # Live trading signals
```
- **Partitioning Logic**: Research calls vs production signals  
- **Usage Limits**: 60/min per key, separate counters
- **Implementation**: Key selection based on `context.environment`

#### FRED - Dataset Categories  
```python  
FRED_MACRO_KEY = "key_macro_..."     # GDP, inflation, employment
FRED_MONETARY_KEY = "key_fed_..."    # Interest rates, money supply  
```
- **Partitioning Logic**: Data category-based routing
- **Usage Limits**: 120/min, 100k/day per key
- **Implementation**: Key selection based on series ID prefix

#### Alpaca - Account Isolation
```python
ALPACA_PAPER_KEY = "PK_PAPER_..."    # Paper trading data
ALPACA_DATA_KEY = "PK_DATA_..."      # Market data only
```  
- **Partitioning Logic**: Trading vs data-only access
- **Usage Limits**: 200/min per key
- **Implementation**: Separate API client instances

### ❌ Single Key Required (ToS Restrictions)

#### NewsAPI - Prohibited Multi-Key Usage
```python
NEWSAPI_KEY = "single_key_only"  # Per ToS Section 2.1
```
- **Enforcement**: Hard-coded single key, no rotation logic
- **Usage Tracking**: Strict 100/day limit with failsafe at 95 calls
- **Quota Management**: Pre-flight checks, graceful degradation

#### Twelve Data - One Key Per User
```python  
TWELVE_DATA_KEY = "single_user_key"  # Per pricing FAQ
```
- **Enforcement**: Single key instance, 8/min rate limiting
- **Task Capping**: Pre-calculate daily tasks ≤800 before scheduling
- **Priority System**: Portfolio > Watchlist > Research for task allocation

#### Alpha Vantage - Personal Use Only
```python
ALPHA_VANTAGE_KEY = "personal_use_key"  # Fallback role only  
```
- **Usage**: Limited to ≤25/day fallback scenarios
- **Triggers**: Only when yfinance fails + Twelve Data unavailable  
- **Guard**: Daily exhaustion flag prevents re-queuing

### 🟡 Ambiguous Cases - Need Vendor Clarification

#### Tiingo - "Personal Use" Boundary
```python
# PROPOSED after vendor clarification:
TIINGO_HISTORICAL_KEY = "hist_..."   # Historical price data
TIINGO_NEWS_KEY = "news_..."         # News sentiment feeds
```
- **Status**: Awaiting clarification on multiple research projects
- **Fallback**: Single key until confirmed
- **Usage**: Conservative 900/day limit (10% buffer)

## Implementation Architecture

### Key Management Service
```python
class APIKeyManager:
    def get_key(self, provider: str, context: RequestContext) -> str:
        """Route requests to appropriate keys based on usage context"""
        
        if provider == "finnhub":
            return self._select_finnhub_key(context)
        elif provider == "newsapi":
            return self._get_single_key(provider)  # Never rotates
        elif provider == "fred":
            return self._select_fred_key(context.data_category)
            
    def _select_finnhub_key(self, context: RequestContext) -> str:
        if context.environment == "research":
            return FINNHUB_RESEARCH_KEY
        elif context.purpose == "trading_signal":  
            return FINNHUB_PROD_KEY
        else:
            raise ValueError("Invalid Finnhub context")
```

### Usage Ledger Design
```python
@dataclass
class KeyUsageLedger:
    provider: str
    key_id: str  # Hash of actual key for security
    calls_today: int
    calls_this_minute: int
    daily_limit: int
    minute_limit: int
    reset_time: datetime
    purpose: str  # "research", "production", "macro_data", etc.
```

### Quota Enforcement
```python
class QuotaEnforcer:
    def pre_flight_check(self, provider: str, context: RequestContext) -> bool:
        """Verify request won't exceed key limits"""
        key = self.key_manager.get_key(provider, context)
        ledger = self.get_ledger(provider, key)
        
        # Check daily limit
        if ledger.calls_today >= ledger.daily_limit:
            self.audit_logger.log_quota_exhausted(provider, key, "daily")
            return False
            
        # Check minute limit  
        if ledger.calls_this_minute >= ledger.minute_limit:
            return False
            
        return True
```

## Audit & Compliance Tracking

### Daily Usage Report
```json
{
  "date": "2025-09-03",
  "provider_usage": {
    "finnhub": {
      "research_key_xxx": {
        "calls": 1247,
        "limit": 1800,
        "purpose": "backtesting_analysis",
        "endpoints": ["/company-news", "/social-sentiment"]
      },
      "prod_key_yyy": {
        "calls": 892,
        "limit": 1800, 
        "purpose": "live_trading_signals",
        "endpoints": ["/quote", "/company-news"]
      }
    },
    "newsapi": {
      "single_key_zzz": {
        "calls": 87,
        "limit": 100,
        "purpose": "news_ingestion",
        "compliance_buffer": 13
      }
    }
  }
}
```

### Compliance Verification
- **Daily Audit**: All key usage logged with purpose classification
- **Violation Detection**: Automatic alerts if usage patterns suggest rotation
- **ToS Monitoring**: Regular review of provider terms for policy changes
- **Vendor Communication**: Log all multi-key clarification requests

## Risk Mitigation

### Technical Safeguards
1. **Hard-Coded Limits** - No dynamic key rotation for single-key providers
2. **Purpose Validation** - Requests must provide valid usage context  
3. **Quota Buffers** - 5-10% safety margin on all limits
4. **Audit Trails** - Immutable log of all key selection decisions

### Legal Protection  
1. **Documentation** - Clear business justification for each key
2. **Vendor Communication** - Written confirmation for ambiguous cases
3. **Usage Transparency** - Open to provider audit if requested
4. **Compliance Review** - Quarterly ToS review for all providers

## Migration Strategy

### Phase 1: Single Key Compliance (Week 1)
- Implement single-key enforcement for prohibited providers
- Add quota buffers and pre-flight checks  
- Deploy usage auditing

### Phase 2: Multi-Key Implementation (Week 2-3)  
- Deploy key routing for explicitly allowed providers
- Implement purpose-based partitioning logic
- Add compliance reporting

### Phase 3: Ambiguous Case Resolution (Week 4)
- Contact vendors for clarification on ambiguous policies
- Implement approved multi-key strategies
- Document final compliance posture

**Success Criteria**: Zero ToS violations, full audit compliance, optimized quota utilization within legal boundaries.