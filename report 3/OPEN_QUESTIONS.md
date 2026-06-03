# Open Questions - Blocking Issues & Clarifications Needed

## 🔴 Critical Blocking Questions

### 1. Alpha Vantage Reset Timing Confirmation
**Status**: 72-hour probe scheduled (Sept 1-3, 2025)
**Location**: `probes/probe_state.json:12-31`

**Hypothesis**:
- **H1**: UTC midnight reset (quota=25 after 00:01Z call)  
- **H2**: Rolling 24h window (quota remains <25)

**Impact**: Critical for daily pipeline scheduling
- If UTC midnight → can schedule at 6 PM ET with confidence
- If rolling 24h → must implement dynamic scheduling based on first call timestamp

**Resolution Timeline**: In progress, results expected Sept 3, 2025
**Contingency**: Assume most restrictive (rolling 24h) until confirmed

### 2. Universe Size & Geographic Distribution
**Missing Information**: 
- Approximate ticker universe size (50? 100? 500?)
- US vs Non-US stock distribution
- Portfolio vs Watchlist symbol counts

**Impact on Quota Allocation**:
```python
# DEPENDS on universe size:
TWELVE_DATA_ALLOCATION = {
    "intraday_bars": universe_size * 0.6,      # US + Non-US
    "indicators_remote": portfolio_size * 4,    # RSI, MACD, EMA, VWAP  
    "price_fallback": watchlist_size * 0.1,    # When yfinance fails
}

# Must not exceed 800/day total
total_needed = sum(TWELVE_DATA_ALLOCATION.values())
assert total_needed <= 720  # 10% safety buffer
```

**Questions**:
1. What is the current portfolio size?
2. What is the current watchlist size?  
3. Are there international stocks that need non-US data?
4. What's the growth trajectory for universe size?

**Resolution Needed**: Before Week 2 pipeline implementation

## 🟡 Medium Priority Questions

### 3. Target Cloud Platform Preference
**Options**: 
- Google Cloud Run (recommended for Python async)
- AWS Lambda (good serverless option)
- GitHub Actions (simple but limited runtime)

**Factors**:
- **Budget**: All are ~$20-50/month for expected usage
- **Runtime**: Cloud Run supports longer executions (60 min vs 15 min)
- **Secrets**: All support secure environment variable management
- **Monitoring**: Cloud Run has better observability integration

**Default Assumption**: Google Cloud Run unless specified otherwise

### 4. Data Storage Preference  
**Current Evidence**: 
- Disk-based caching in `data/` directory (yfinance, audit logs)
- Redis integration ready but optional
- No evidence of PostgreSQL/TimescaleDB

**Options**:
- **SQLite** (local file, simple, good for <100k records/day)
- **PostgreSQL** (better for larger scale, JSONB support)
- **Redis** (fast, good for counters/cache, not persistent)
- **Hybrid** (Redis for counters, SQLite/PostgreSQL for data)

**Questions**:
1. Expected data volume per day (MB/GB)?
2. Need for historical data retention (weeks/months/years)?
3. Query patterns (real-time lookups vs batch analysis)?
4. Preference for managed vs self-hosted?

**Default Assumption**: SQLite + Redis hybrid until scale demands PostgreSQL

### 5. Must-Have vs Nice-to-Have Features
**Current Feature Flags**: 25+ features, many disabled by default

**Prioritization Needed**:
```python
MUST_HAVE = [
    "enable_yf_prices",           # Core price data
    "enable_newsapi_ingestion",   # News breadth  
    "enable_symbol_intake",       # Portfolio management
    "enable_finbert_pipeline"     # Sentiment analysis
]

NICE_TO_HAVE = [
    "enable_correlation",         # Portfolio analytics
    "enable_gnn_scaffold",        # Advanced ML
    "enable_vector_search",       # Semantic search
    "enable_alt_forecasts"        # Prediction models
]
```

**Questions**:
1. Which features are business-critical vs experimental?
2. What's the priority order for new provider integrations?
3. Are there features that can be deprecated to reduce complexity?

## 🟠 Low Priority Questions

### 6. Vendor ToS Clarifications (Non-Blocking)
**Pending Vendor Questions**:
- **Tiingo**: Multiple keys for data partitioning allowed?
- **GNews**: Multiple applications same organization?  
- **Marketstack**: Research vs production key separation?
- **BLS**: Academic vs commercial registration differences?

**Impact**: Determines multi-key strategy implementation
**Timeline**: 2-4 weeks response time typical
**Workaround**: Use single keys conservatively until clarification

### 7. Advanced ML Model Preferences
**Current**: FinBERT sentiment, basic clustering, rule-based recommendations
**Enhancement Options**:
- **Transformers**: BERT/FinBERT fine-tuning for financial text
- **Time Series**: LSTM/GRU for price prediction
- **Ensemble**: Random Forest/XGBoost for recommendation scoring
- **External APIs**: TimeGPT, OpenAI API for analysis

**Questions**:  
1. Preference for on-premises vs API-based models?
2. Acceptable model inference latency (real-time vs batch)?
3. Model interpretability requirements?
4. Budget for external AI APIs?

### 8. Monitoring & Alerting Preferences
**Options**:
- **Logging**: Structured JSON, plain text, database storage
- **Metrics**: Prometheus, InfluxDB, simple file-based
- **Alerts**: Email, Slack, PagerDuty, webhooks
- **Dashboards**: Grafana, custom web app, simple reports

**Current Implementation**: JSONL structured logging to files

**Questions**:
1. Preferred notification channels for alerts?
2. Need for real-time dashboards vs daily reports?
3. Integration with existing monitoring tools?

## Resolution Timeline & Priorities

### Week 1 (Required)
- [ ] **Alpha Vantage reset timing** → Affects pipeline scheduling  
- [ ] **Universe size estimation** → Affects quota allocation math
- [ ] **Cloud platform selection** → Affects deployment planning

### Week 2 (Helpful)  
- [ ] **Data storage architecture** → Affects pipeline persistence layer
- [ ] **Feature prioritization** → Affects development focus
- [ ] **Monitoring preferences** → Affects observability implementation

### Week 3+ (Nice to Have)
- [ ] **Vendor ToS clarifications** → Affects multi-key strategy
- [ ] **ML model preferences** → Affects advanced analytics roadmap
- [ ] **Integration preferences** → Affects tool selection

## Decision-Making Framework

### When No Response Available
1. **Choose Conservative Option**: Single keys, smaller quotas, simpler architecture
2. **Design for Flexibility**: Make decisions easily reversible
3. **Document Assumptions**: Clear reasoning for future reference  
4. **Build Monitoring**: Track impacts of assumptions for validation

### Example Decision Tree - Universe Size
```
IF universe_size unknown:
  ASSUME 50 stocks (conservative)
  IMPLEMENT quota_allocation(universe_size) as configurable parameter
  ADD monitoring for actual usage vs allocation
  PLAN for easy scaling when real size known
```

## Information Gathering Strategy

### Questions for User/Stakeholder
```
Subject: Configuration Questions for Stock News Pipeline

Hi! I'm implementing the multi-provider data pipeline and need a few details 
to optimize the configuration:

1. **Portfolio Size**: How many stocks are typically in your portfolio/watchlist?
2. **Geographic Mix**: Do you track any international stocks (non-US)?  
3. **Cloud Preference**: Any preference between Google Cloud Run vs AWS Lambda?
4. **Priority Features**: Which features are most important vs experimental?

These will help me configure quotas and scheduling optimally. I can work with 
reasonable defaults, but specifics would help maximize efficiency.

Thanks!
```

### Technical Investigation Tasks  
- [ ] Analyze `data/portfolio.csv` and `data/watchlist.csv` for universe size
- [ ] Check existing log files for geographic distribution patterns
- [ ] Review git history for most actively developed features
- [ ] Estimate data volume from existing cache files

**Overall Approach**: Proceed with reasonable assumptions, build flexibility for changes, monitor actual usage to validate decisions.