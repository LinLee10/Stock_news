# Stonk News Pipeline

A comprehensive stock analysis pipeline that combines news sentiment analysis with machine learning forecasts for investment insights.

## Features

- **News Sentiment Analysis**: Multi-source news aggregation with FinBERT sentiment scoring
- **ML-Powered Forecasting**: Ensemble models (Random Forest, XGBoost, LSTM) for price predictions  
- **Portfolio Analytics**: Sector allocation, beta analysis, and benchmark comparison
- **Correlation Analysis**: Portfolio correlation heatmaps with Pearson correlation matrices
- **GNN Scaffold**: Graph Neural Network data loaders for financial graph analysis
- **Multi-Source Data**: Alpha Vantage, yfinance (free), and premium data sources
- **Email Reports**: Automated HTML reports with charts and insights
- **Feature Flags**: Safe rollout of new functionality with backward compatibility

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file:
```bash
cp .env.example config/secrets.env
```

Edit `config/secrets.env` with your API keys:
```bash
# Required
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
GMAIL_USER=your_email@gmail.com  
GMAIL_PASSWORD=your_app_password

# Optional: Enable additional features
ENABLE_YF_PRICES=true
ENABLE_PORTFOLIO_ANALYTICS=true
ENABLE_CORRELATION=true
ENABLE_EARNINGS_READS=true
```

### 3. Run the Pipeline

```bash
python main.py
```

Or start the API server:
```bash
python main.py --api
```

## Data Sources

### yfinance (Free) - Once-Per-Day

**No API key required** - Free historical stock data from Yahoo Finance with intelligent once-per-day refresh.

Enable yfinance with once-per-day rate limiting:
```bash
ENABLE_YF_PRICES=true
ENABLE_YF_DAILY_REFRESH=true
```

#### Key Features:
- **Once-per-day refresh** - Network calls limited to one daily window to avoid rate limits
- **Manifest-based tracking** - Uses `data/yf_bulk_cache/manifest.json` to prevent duplicate refreshes  
- **Lockfile protection** - Prevents concurrent refresh attempts
- **Graceful rate-limit handling** - Falls back to cached data when rate-limited
- **Fast-failing tests** - Test suite completes quickly with mocked network calls

#### Configuration Options:
```bash
# Basic yfinance settings
YF_CACHE_TTL_HOURS=24              # Cache TTL in hours (default: 24)
YF_MAX_RETRIES=2                   # Max retry attempts (reduced for free tier, default: 2)
YF_BACKOFF_BASE_SECONDS=2.0        # Base delay for backoff (default: 2.0)
YF_REFRESH_WINDOW_UTC_HOUR=2       # Daily refresh window UTC hour (default: 2 AM)
YF_DAILY_KEY="period=2y|interval=1d|auto_adjust=1"  # Parameters key for caching

# Feature flags
ENABLE_YF_PRICES=false             # Enable yfinance integration
ENABLE_YF_DAILY_REFRESH=false      # Enable once-per-day refresh guard
ENABLE_YF_PROFILES=false           # Fetch company profiles
ENABLE_YF_BACKOFF_DEBUG=false      # Debug retry/backoff logging

# Test mode (for development only)
YF_TEST_FAST_BACKOFF=0             # Set to '1' in tests to skip sleep delays
```

#### How Once-Per-Day Works:

1. **Daily Window Check**: Only refreshes during/after configured UTC hour (default: 2 AM)
2. **Manifest Guard**: Tracks refresh state in `data/yf_bulk_cache/manifest.json`
3. **Cache-Only Reads**: Normal pipeline reads only from cache, never makes network calls
4. **Bounded Retries**: Maximum 2 attempts per symbol, then graceful fallback
5. **Atomic Cache Updates**: Writes to `.tmp` files then renames for safe concurrent access

#### Manual CLI Usage:
```bash
# Check refresh status
python scripts/refresh_yf_cache.py --status

# Force manual refresh  
python scripts/refresh_yf_cache.py --force

# Refresh specific symbols
python scripts/refresh_yf_cache.py --symbols AAPL MSFT GOOGL

# Dry run (show what would happen)
python scripts/refresh_yf_cache.py --dry-run
```

#### Cache Files:
- **Manifest**: `data/yf_bulk_cache/manifest.json` - Tracks daily refresh state
- **Lock**: `data/yf_bulk_cache/.refresh.lock` - Prevents concurrent refreshes
- **History**: `data/yf_bulk_cache/{SYMBOL}_2y_1d.csv` - Cached OHLCV data
- **Profiles**: `data/yf_bulk_cache/profiles.csv` (if enabled)
- **HTTP Cache**: `data/yf_bulk_cache/yf_http_cache.sqlite`

#### Rate Limiting Behavior:
When rate-limited during the daily refresh:
- **Limited Retries**: 2 attempts max per symbol with 2→4 second exponential backoff
- **Fail Fast**: Skips rate-limited symbols quickly rather than hanging
- **Cache Fallback**: Pipeline continues using yesterday's cached data
- **No Pipeline Breaks**: Main execution never fails due to rate limits

#### Acceptance Criteria Met:
- ✅ **AC1**: Runs once/day max based on manifest guard; subsequent runs use cache only
- ✅ **AC2**: Rate-limited calls retry ≤2 times with 2s→4s backoff, then graceful skip
- ✅ **AC3**: Read path never makes network calls—only guarded refresh does
- ✅ **AC4**: Tests finish <30s with mocked network and zero sleep delays
- ✅ **AC5**: With flags disabled, behavior identical to original

## Smart Alerts (F06)

**Automated alerts for price moves, sentiment changes, and earnings proximity** - Get notified of significant market events.

Enable smart alerts:
```bash
ENABLE_SMART_ALERTS=true
```

### Key Features:
- **Price Movement Alerts** - Trigger on configurable percentage moves (default: 5%)
- **Sentiment Swing Alerts** - Detect significant sentiment changes from news analysis
- **Earnings Proximity Alerts** - Alert at 7, 3, and 1 days before earnings
- **Symbol-Specific Thresholds** - Custom alert levels for different stocks
- **Batched Daily Reports** - Alerts integrated into email reports
- **Anti-Spam Protection** - Cooldown periods prevent alert flooding

### Configuration File: `config/alerts.yaml`

```yaml
# Global defaults (applied to all symbols)
defaults:
  price_move_threshold_percent: 5.0      # Alert on moves > 5%
  sentiment_threshold_change: 0.3        # Alert on sentiment change > 0.3
  earnings_alert_days: [7, 3, 1]         # Alert N days before earnings
  batch_alerts: true                     # Include in daily email vs immediate
  immediate_email_severity: ["HIGH", "CRITICAL"]  # Send immediate emails
  
# Symbol-specific overrides
symbol_overrides:
  TSLA:
    price_move_threshold_percent: 8.0    # Higher threshold for volatile stocks
  AAPL:
    sentiment_threshold_change: 0.25     # Lower threshold for key holdings
```

### Alert Types & Severities:

**Price Movements:**
- **LOW** (3%+): Minor price moves
- **MEDIUM** (5%+): Significant price moves  
- **HIGH** (10%+): Major price moves
- **CRITICAL** (15%+): Extreme price moves

**Sentiment Changes:**
- **LOW** (0.2+): Minor sentiment shifts
- **MEDIUM** (0.3+): Notable sentiment changes
- **HIGH** (0.5+): Major sentiment swings
- **CRITICAL** (0.7+): Extreme sentiment reversals

**Earnings Proximity:**
- **LOW**: 7 days before earnings
- **MEDIUM**: 3 days before earnings
- **HIGH**: 1 day before earnings
- **CRITICAL**: Earnings day

### Email Integration:

Smart alerts appear in daily reports with:
- 📈📉 **Price alerts** with current/previous prices and percentage moves
- 💭 **Sentiment alerts** with change indicators and guidance
- 📊 **Earnings alerts** with countdown and preparation tips
- **Color-coded severity levels** for quick prioritization
- **Actionable guidance** for high-priority alerts
- **Disclaimer** about investment risk

### Alert Logic:

1. **Price Alerts**: Compare current price to N days ago (default: 1 day)
2. **Sentiment Alerts**: Compare current sentiment to N-day average (default: 7 days)
3. **Earnings Alerts**: Countdown to upcoming earnings dates
4. **Cooldown Protection**: 60-minute cooldown between alerts per symbol
5. **Threshold Customization**: Symbol-specific overrides for fine-tuning

### Example Alerts:

- `Price Alert: AAPL up 8.5%` - Apple stock jumped 8.5% in one day
- `Sentiment Alert: TSLA sentiment improved` - Tesla news sentiment turned positive
- `Earnings Alert: MSFT reports in 3 days` - Microsoft earnings approaching

### Performance & Security:

- **Fast Processing**: Alerts evaluate in <5 seconds for full portfolio
- **Rate Limited**: Maximum 50 alerts per hour globally
- **Anti-Spam**: Symbol cooldowns prevent duplicate notifications
- **Configurable**: All thresholds adjustable per symbol/globally
- **Safe Defaults**: Conservative thresholds prevent noise

### Testing:

```bash
# Run smart alerts unit tests
python -m pytest tests/test_smart_alerts.py -v

# Run smart alerts integration tests  
python -m pytest tests/integration/test_smart_alerts_integration.py -v
```

### Acceptance Criteria Met:
- ✅ **AC1**: Alerts batched into daily email reports with optional immediate delivery for high severity
- ✅ **AC2**: No alerts when `ENABLE_SMART_ALERTS=false` - feature completely disabled

## Enhanced Reports & Charts (F07)

**Professional visual upgrades to charts and email reports** - Enhanced layouts with optional analytics panes while maintaining backward compatibility.

### Key Features:
- **Enhanced Chart Layouts** - Optional panes for portfolio analytics, benchmarks, and alerts
- **Advanced Email Styling** - Professional gradients, enhanced tables, and visual indicators
- **Backward Compatibility** - Legacy structure unchanged when features disabled (AC1)
- **Modular Sections** - Enhanced sections append below legacy content (AC2)
- **Renderer Helpers** - Dedicated functions for benchmark, alert, and FinBERT sections
- **Performance Footer** - Generation metrics and feature indicators

### Enhanced Chart Functions:

**Legacy Function (unchanged):**
```python
from charts import create_collage

create_collage(tickers, price_data, forecast_data, "Portfolio Charts", "output.png")
```

**Enhanced Function (new):**
```python
from charts import create_enhanced_collage

create_enhanced_collage(
    tickers, price_data, forecast_data, 
    "Enhanced Portfolio Charts", "output.png",
    portfolio_analytics=analytics_data,    # F05 portfolio metrics pane
    benchmark_data=benchmark_performance,  # F05 benchmark comparison pane  
    smart_alerts=triggered_alerts,         # F06 alerts summary pane
    finbert_results=ai_analysis           # F04 FinBERT insights pane
)
```

### Chart Layout Enhancements:

**Standard Layout (flags OFF):**
- Stock price charts in 2-column grid
- Identical to legacy `create_collage` output
- No additional panes or features

**Enhanced Layout (flags ON):**
- Stock charts in upper rows (unchanged positioning)
- **Portfolio Analytics Pane** - Sector allocation pie chart + beta metrics
- **Benchmark Comparison Pane** - Performance bars with portfolio overlay  
- **Smart Alerts Pane** - Alert count visualization by severity
- **Feature Indicators** - Title shows enabled features and timestamp

### Email Report Enhancements:

**Enhanced Benchmark Analysis:**
- Gradient backgrounds and professional styling
- Trend indicators (🔥 bull, ❄️ bear, ⚖️ mixed) 
- Portfolio performance overlay with highlighting
- Color-coded returns (green positive, red negative)

**Enhanced Alert Dashboard:**
- Severity-based color coding and icons
- Alert type breakdown with counts
- Priority alerts section with actionable guidance
- Visual summary cards for each severity level

**Enhanced FinBERT Analysis:**  
- Portfolio sentiment overview with aggregate metrics
- Recommendation distribution visualization
- Confidence and conviction scoring
- Action breakdown with color-coded circles

**Performance Footer:**
- Generation timestamp and feature indicators
- Processing metrics and version information
- Professional gradient styling with feature badges

### Renderer Helper Functions:

```python
from email_report import (
    render_enhanced_benchmark_section,
    render_enhanced_alerts_section, 
    render_enhanced_finbert_section,
    render_performance_footer
)

# Create enhanced HTML sections programmatically
benchmark_html = render_enhanced_benchmark_section(benchmark_data, portfolio_analytics)
alerts_html = render_enhanced_alerts_section(smart_alerts) 
finbert_html = render_enhanced_finbert_section(finbert_results)
footer_html = render_performance_footer()
```

### Backward Compatibility:

**AC1 - Legacy Structure Unchanged:**
- With all feature flags OFF, PNG/HTML output identical to current structure
- No enhanced sections appear in reports
- Standard 2-column chart layout maintained
- CSS unchanged for legacy table elements

**AC2 - Enhanced Sections Append Below:**
- Enhanced sections appear AFTER all legacy content
- Legacy sections (30-day sentiment, headlines) come first
- F04/F05/F06 sections in middle  
- F07 enhanced sections at end before closing HTML
- Performance footer last

### Visual Styling:

**Professional Enhancements:**
- Linear gradient backgrounds for section headers
- Box shadows and rounded corners
- Color-coded severity indicators
- Emoji icons for visual context
- Responsive table layouts
- Typography improvements

**CSS Classes & IDs:**
- Stable element ordering for idempotency
- No conflicting styles with legacy elements
- Enhanced sections use distinct styling
- Backward compatible class names

### Testing:

```bash
# Run F07 upgrade tests
python -m pytest tests/test_f07_report_upgrades.py -v

# Run chart smoke tests  
python -m pytest tests/test_f07_chart_smoke.py -v
```

**Test Coverage:**
- Snapshot tests ensure HTML structure unchanged with flags OFF
- Enhanced features properly append with flags ON
- Chart generation works with all feature combinations
- Visual smoke tests for chart rendering
- Performance and thread safety validation

### Performance & Telemetry:

- **Render Time**: Enhanced sections add <1s to report generation
- **Chart Generation**: Optional panes add ~2s for complex layouts
- **Memory Usage**: Minimal impact with efficient matplotlib usage
- **File Sizes**: Enhanced charts ~20% larger due to additional panes

### Security & Idempotency:

- No new secrets or API keys required
- Deterministic HTML output with stable element ordering
- Same inputs produce identical enhanced reports
- No external dependencies for F07 features

### Acceptance Criteria Met:
- ✅ **AC1**: With all flags OFF, PNG/HTML match current legacy structure exactly
- ✅ **AC2**: With flags ON, enhanced sections append below legacy blocks without disruption

## REST API (F08)

**Secure REST API endpoints for programmatic access** - API key authentication, OpenAPI documentation, and comprehensive endpoint coverage.

Enable the REST API server:
```bash
ENABLE_API_ENDPOINTS=true
API_KEY=your_secure_api_key_here
```

Start the API server:
```bash
python api/app.py
# or
python main.py --api
```

### Key Features:
- **API Key Authentication** - Secure X-API-Key header-based authentication
- **OpenAPI Documentation** - Interactive Swagger UI and JSON specification
- **Feature Flag Integration** - Endpoints controlled by granular feature flags
- **Comprehensive Coverage** - All major functionality accessible via REST
- **Public Health Endpoints** - Health checks remain accessible without authentication

### Authentication:

**Protected Endpoints** require `X-API-Key` header:
```bash
curl -H "X-API-Key: your_api_key_here" http://localhost:8000/api/v1/recs?scope=watchlist
```

**Public Endpoints** (no authentication required):
- `/api/v1/health` - Health check and status
- `/api/v1/flags` - Feature flag states  
- `/api/v1/config` - Configuration debug (when debug enabled)
- `/api/v1/docs` - Swagger UI documentation
- `/api/v1/openapi.json` - OpenAPI specification

### API Endpoints:

#### Recommendations
- `GET /api/v1/recs?scope=watchlist|portfolio` - Generate buy/hold/reduce/exit recommendations
- **Parameters**: `scope`, `include_details`, `max_age_hours`
- **Authentication**: Required

#### Symbol Management  
- `POST /api/v1/symbols/intake` - Submit symbol for analysis
- `GET /api/v1/symbols/jobs/{job_id}` - Check job status
- `GET /api/v1/symbols/{symbol}/intake_status` - Get symbol intake progress
- `GET /api/v1/symbols/{symbol}/snapshot` - Get comprehensive symbol overview
- **Authentication**: Required

#### Earnings Analysis
- `GET /api/v1/earnings/upcoming?days=21` - Get upcoming earnings calendar
- `GET /api/v1/earnings/{symbol}/explain` - Detailed earnings analysis explanation
- **Authentication**: Required

#### Admin & Monitoring
- `GET /api/v1/admin/logs` - Recent audit logs with filtering
- `GET /api/v1/admin/logs/summary` - Aggregated operation summary
- `GET /api/v1/admin/logs/operation/{id}` - Specific operation details
- `GET /api/v1/admin/logs/export` - Export logs in JSONL format
- **Authentication**: Required

### OpenAPI Documentation:

**Swagger UI**: http://localhost:8000/api/v1/docs
**OpenAPI Spec**: http://localhost:8000/api/v1/openapi.json

The interactive Swagger UI provides:
- Complete endpoint documentation
- Parameter descriptions and validation
- Example requests and responses
- Built-in API testing capabilities
- Authentication scheme documentation

### Configuration:

**Required Environment Variables:**
```bash
# API Server
ENABLE_API_ENDPOINTS=true              # Enable REST API endpoints
API_KEY=your_secure_api_key_here      # API authentication key
API_HOST=localhost                     # Server host (optional, default: localhost)
API_PORT=8000                         # Server port (optional, default: 8000)
API_DEBUG=false                       # Debug mode (optional, default: false)

# Feature-Specific Endpoints
ENABLE_SYMBOL_INTAKE=true             # Enable symbol intake endpoints  
ENABLE_RECOS=true                     # Enable recommendations endpoints
ENABLE_EARNINGS_READS=true            # Enable earnings analysis endpoints
```

### Example Usage:

**Get Watchlist Recommendations:**
```bash
curl -H "X-API-Key: your_api_key" \
  "http://localhost:8000/api/v1/recs?scope=watchlist&include_details=true"
```

**Submit Symbol for Analysis:**
```bash
curl -X POST \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "company_name": "Tesla Inc", "priority": "high"}' \
  http://localhost:8000/api/v1/symbols/intake
```

**Check Upcoming Earnings:**
```bash
curl -H "X-API-Key: your_api_key" \
  "http://localhost:8000/api/v1/earnings/upcoming?days=14"
```

**Monitor System Logs:**
```bash
curl -H "X-API-Key: your_api_key" \
  "http://localhost:8000/api/v1/admin/logs?hours=24&feature=recommendations"
```

### Security Features:

**API Key Authentication:**
- Header-based authentication: `X-API-Key: your_key`
- Configurable per-environment API keys
- Invalid key attempts logged with source IP
- 401 Unauthorized for missing keys
- 403 Forbidden for invalid keys

**Request Validation:**
- Parameter validation for all endpoints
- JSON schema validation for POST requests  
- Proper HTTP status codes for all error conditions
- Detailed error messages with error codes

**Feature Flag Security:**
- API endpoints disabled by default (`ENABLE_API_ENDPOINTS=false`)
- Granular feature control per endpoint category
- Safe degradation when features disabled (503 Service Unavailable)

### Error Responses:

**Authentication Errors:**
```json
{
  "error": "API key required",
  "details": "Include X-API-Key header with valid API key", 
  "code": "MISSING_API_KEY"
}
```

**Feature Disabled:**
```json
{
  "error": "Recommendations feature is disabled",
  "code": "FEATURE_DISABLED"
}
```

**Validation Error:**
```json
{
  "error": "Validation failed",
  "details": {"ticker": ["Missing data for required field."]}
}
```

### Performance & Monitoring:

**Request Auditing:**
- All API requests logged with operation tracking
- Request/response times recorded  
- Feature usage metrics collected
- Error rates and authentication failures tracked

**Rate Limiting Protection:**
- Queue capacity limits for symbol intake
- Graceful handling of downstream service limits
- Proper HTTP status codes (429 Too Many Requests)

### Testing:

```bash
# Run API authentication tests
python -m pytest tests/test_f08_api_auth.py -v

# Run API integration tests
python -m pytest tests/integration/test_f08_api_integration.py -v
```

**Test Coverage:**
- API key authentication validation
- All endpoint authentication requirements
- Feature flag integration testing
- OpenAPI specification completeness
- Error handling and edge cases
- Integration workflow testing

### Acceptance Criteria Met:
- ✅ **Feature Flags & Config**: API controlled by `enable_api_endpoints=false` flag
- ✅ **OpenAPI Documentation**: Complete Swagger UI with interactive testing
- ✅ **API Key Authentication**: Secure header-based auth with proper error handling  
- ✅ **Health Endpoints Public**: `/health`, `/flags`, `/config` remain accessible
- ✅ **Protected Endpoints**: All data endpoints require valid X-API-Key header

## Async I/O + Caching + Retries (F09)

**Resilient network operations with asyncio gather, semaphore limits, exponential backoff, circuit breakers, and local result caching** - Makes daily runs resilient to transient failures.

Enable async I/O with resilience features:
```bash
ENABLE_ASYNC_IO=true
MAX_CONCURRENCY=5
BACKOFF_MAX=60
```

### Key Features:
- **Asyncio Gather with Semaphore Limits** - Concurrent network calls with configurable concurrency control
- **Exponential Backoff with Jitter** - Smart retry logic to handle rate limiting and temporary failures
- **Circuit Breakers per Host** - Prevent cascading failures with automatic failure detection and recovery
- **Local Result Caching** - In-memory cache with TTL (24h default) for idempotent requests
- **Graceful Degradation** - Falls back to sync behavior when async I/O disabled or fails

### Core F09 Pattern - Asyncio Gather with Semaphore Limits:

**News Scraping (Multiple Tickers Concurrently):**
```python
from news_scraper import scrape_headlines_resilient

# Automatically chooses async or sync based on feature flag
results = scrape_headlines_resilient(['AAPL', 'GOOGL', 'MSFT'], days=30)

# Or use async version directly
import asyncio
from news_scraper import scrape_headlines_async

results = await scrape_headlines_async(['AAPL', 'GOOGL', 'MSFT'], days=30)
```

**Price Data (Multiple Sources Concurrently):**
```python
from services.multi_source_data_manager import get_all_price_data_resilient

# Resilient wrapper (sync/async automatic)
price_data = get_all_price_data_resilient(lookback_days=90)

# Or async version directly
from services.multi_source_data_manager import get_all_price_data_async

price_data = await get_all_price_data_async(lookback_days=90)
```

**Direct Resilient HTTP Requests:**
```python
from services.retry_policies import make_resilient_requests

urls = [
    "https://api1.example.com/data",
    "https://api2.example.com/data", 
    "https://api3.example.com/data"
]

# Execute concurrently with circuit breakers and caching
results = await make_resilient_requests(
    urls, 
    method='GET',
    max_concurrency=5
)
```

### Circuit Breaker Protection:

Circuit breakers automatically detect failing services and prevent cascading failures:

**States:**
- **CLOSED** - Normal operation, requests pass through
- **OPEN** - Service failing, requests fail fast without network calls  
- **HALF_OPEN** - Testing recovery, limited requests allowed

**Configuration:**
```bash
CIRCUIT_FAILURE_THRESHOLD=5      # Failures before opening circuit
CIRCUIT_RECOVERY_TIMEOUT=60      # Seconds before attempting recovery
```

**Per-Host Circuit Breakers:**
- `news.google.com` - Google News RSS feeds
- `www.alphavantage.co` - Alpha Vantage API  
- `api.newsapi.org` - NewsAPI integration
- Each host has independent circuit breaker state

### Retry Logic with Exponential Backoff:

**Retry Configuration:**
```python
from services.retry_policies import AsyncRetryConfig

config = AsyncRetryConfig(
    max_retries=3,           # Maximum retry attempts
    base_delay=1.0,          # Base delay in seconds
    max_delay=60.0,          # Maximum delay cap
    exponential_base=2.0,    # Backoff multiplier
    jitter=True,             # Add randomness to prevent thundering herd
    backoff_strategy="exponential"  # exponential, linear, or fixed
)
```

**Retry Sequence Example:**
- Attempt 1: Immediate
- Attempt 2: ~1s delay (base_delay × 2^0)
- Attempt 3: ~2s delay (base_delay × 2^1) 
- Attempt 4: ~4s delay (base_delay × 2^2)
- Final failure after 4 attempts

### Local Result Caching:

**Idempotent Cache with TTL:**
```python
# Cache key = URL + params (excluding auth headers)
# TTL = 24 hours default
# Max entries = 1000 (LRU eviction)

CACHE_TTL=86400              # Cache TTL in seconds
```

**Cache Behavior:**
- **GET requests cached** - POST requests never cached
- **Authorization headers excluded** - Prevents credential leakage in cache keys
- **LRU eviction** - Oldest entries removed when at capacity
- **TTL expiration** - Automatic cleanup of expired entries

### Configuration:

**Environment Variables:**
```bash
# Core F09 Settings
ENABLE_ASYNC_IO=false            # Enable async I/O with resilience features
MAX_CONCURRENCY=5                # Maximum concurrent requests (semaphore limit)
BACKOFF_MAX=60                   # Maximum retry delay in seconds

# Circuit Breaker Settings  
CIRCUIT_FAILURE_THRESHOLD=5      # Failures before opening circuit
CIRCUIT_RECOVERY_TIMEOUT=60      # Recovery attempt timeout in seconds

# Caching Settings
CACHE_TTL=86400                  # Cache TTL in seconds (24 hours)
REQUEST_TIMEOUT=30               # HTTP request timeout in seconds
```

### Resilience Features:

**AC1: Daily run resilient to transient failures; retries capped; failures logged but do not abort**

**Failure Handling:**
- **Capped Retries** - Maximum retry attempts prevent infinite loops
- **Logged but Non-Blocking** - Failed requests logged as warnings, pipeline continues
- **Graceful Degradation** - Falls back to sync mode if async fails completely
- **Partial Success** - Pipeline completes successfully even if some requests fail

**Error Scenarios Handled:**
- **Network timeouts** - Retry with exponential backoff
- **Rate limiting** - Circuit breaker opens, subsequent requests fail fast
- **DNS failures** - Circuit breaker per host prevents repeated failures
- **Server errors (5xx)** - Trigger retries, client errors (4xx) do not
- **Connection errors** - Automatic retry with jitter to prevent thundering herd

### Performance & Telemetry:

**Performance Metrics:**
```python
from services.retry_policies import get_async_stats

stats = get_async_stats()
# Returns:
# - Circuit breaker states and failure counts per host
# - Cache hit rates and entry counts  
# - Average latency and request counts
# - Feature usage and configuration
```

**Key Metrics Tracked:**
- **Average Latency** - Mean request response time
- **Cache Hit Rate** - Percentage of requests served from cache
- **Circuit Breaker State** - CLOSED/OPEN/HALF_OPEN per host
- **Retry Attempts** - Total retries performed
- **Failure Rates** - Success vs failure percentages

**Telemetry Integration:**
```python
# Get comprehensive F09 performance stats
from services.multi_source_data_manager import async_data_manager

perf_stats = async_data_manager.get_async_performance_stats()
# Includes global async stats, config, and feature flag state
```

### Backward Compatibility:

**Automatic Fallback:**
- **Feature flag disabled** - Uses original sync behavior
- **Async failure** - Automatically falls back to sync implementation
- **Missing dependencies** - Gracefully degrades to sync mode
- **Configuration errors** - Defaults to safe sync behavior

**Wrapper Functions:**
```python
# These automatically choose async or sync based on ENABLE_ASYNC_IO
from news_scraper import scrape_headlines_resilient
from services.multi_source_data_manager import get_all_price_data_resilient

# Always work regardless of feature flag state
results = scrape_headlines_resilient(['AAPL'])
price_data = get_all_price_data_resilient()
```

## TimescaleDB Persistence (F10)

**Optional database persistence for time-series financial data** - Provides scalable TimescaleDB storage for prices, sentiment, and predictions with automatic schema management.

### Key Features:
- **Hypertable Schema** - Optimized time-partitioned tables for financial data
- **No-Op by Default** - Completely disabled unless both feature flag and DSN configured
- **Batch Operations** - Efficient bulk inserts with upsert conflict resolution
- **Auto Schema Creation** - DDL automatically executed on first connection
- **Connection Pooling** - Async connection management with health monitoring

### Database Schema:

**Three core hypertables for time-series data:**

```sql
-- Prices: OHLCV stock price data
CREATE TABLE prices (
    symbol VARCHAR(10) NOT NULL,
    date_recorded TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4),
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4),
    volume BIGINT,
    adjusted_close DECIMAL(12,4),
    PRIMARY KEY (symbol, date_recorded)
);

-- Sentiment: News sentiment analysis results
CREATE TABLE sentiment (
    symbol VARCHAR(10) NOT NULL,
    date_recorded TIMESTAMPTZ NOT NULL,
    sentiment_score DECIMAL(5,4),
    sentiment_label VARCHAR(20),
    confidence_score DECIMAL(5,4),
    article_count INTEGER DEFAULT 1,
    positive_mentions INTEGER DEFAULT 0,
    negative_mentions INTEGER DEFAULT 0,
    neutral_mentions INTEGER DEFAULT 0,
    PRIMARY KEY (symbol, date_recorded)
);

-- Predictions: ML model forecasts
CREATE TABLE predictions (
    symbol VARCHAR(10) NOT NULL,
    date_recorded TIMESTAMPTZ NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    predicted_value DECIMAL(12,4),
    confidence_score DECIMAL(5,4),
    model_version VARCHAR(20),
    features_used TEXT[],
    prediction_horizon_days INTEGER,
    actual_value DECIMAL(12,4),
    error_magnitude DECIMAL(12,4),
    PRIMARY KEY (symbol, date_recorded, prediction_type)
);
```

### Configuration:

**Environment Variables:**
```bash
# F10 TimescaleDB Settings
ENABLE_TIMESCALE_PERSISTENCE=false    # Enable database persistence
PG_DSN=postgres://user:pass@host:5432/db  # PostgreSQL connection string
```

**Requirements for Activation:**
- Feature flag: `ENABLE_TIMESCALE_PERSISTENCE=true`
- Database DSN: `PG_DSN` environment variable set
- TimescaleDB extension available on target database
- `asyncpg` Python library installed

### Usage Examples:

**Persist Price Data:**
```python
from services.db_adapter import persist_prices, PriceRecord
from datetime import datetime

# Create price records
prices = [
    PriceRecord(
        symbol="AAPL",
        date_recorded=datetime(2023, 12, 1),
        open_price=195.50,
        high_price=197.80,
        low_price=194.20,
        close_price=196.40,
        volume=50000000,
        adjusted_close=196.40
    )
]

# Persist to database (no-op if not configured)
records_saved = await persist_prices(prices)
```

**Persist Sentiment Data:**
```python
from services.db_adapter import persist_sentiment, SentimentRecord

# Create sentiment records
sentiments = [
    SentimentRecord(
        symbol="AAPL",
        date_recorded=datetime(2023, 12, 1),
        sentiment_score=0.75,
        sentiment_label="positive",
        confidence_score=0.92,
        article_count=15,
        positive_mentions=10,
        negative_mentions=2,
        neutral_mentions=3,
        data_source="finbert"
    )
]

# Persist to database
records_saved = await persist_sentiment(sentiments)
```

**Persist ML Predictions:**
```python
from services.db_adapter import persist_predictions, PredictionRecord

# Create prediction records
predictions = [
    PredictionRecord(
        symbol="AAPL",
        date_recorded=datetime(2023, 12, 1),
        prediction_type="price_movement",
        predicted_value=200.50,
        confidence_score=0.85,
        model_version="rf_v2.1",
        features_used=["price", "volume", "sentiment"],
        prediction_horizon_days=7
    )
]

# Persist to database
records_saved = await persist_predictions(predictions)
```

### Adapter Management:

**Initialize Database Connection:**
```python
from services.db_adapter import initialize_db_adapter, db_adapter

# Initialize global adapter
success = await initialize_db_adapter()
if success:
    print("TimescaleDB adapter ready")

# Check health status
health = await db_adapter.health_check()
print(f"Database healthy: {health['connection_healthy']}")
```

**Query Historical Data:**
```python
from datetime import datetime, timedelta

# Query price history
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()

prices = await db_adapter.query_prices("AAPL", start_date, end_date)
sentiment_data = await db_adapter.query_sentiment("AAPL", start_date, end_date)
predictions = await db_adapter.query_predictions("AAPL", "price_movement")
```

### Performance Features:

**TimescaleDB Optimizations:**
- **Time Partitioning** - Automatic chunking by date intervals
  - Prices: Monthly chunks (`INTERVAL '1 month'`)
  - Sentiment: Weekly chunks (`INTERVAL '1 week'`)
  - Predictions: Monthly chunks (`INTERVAL '1 month'`)
- **Indexing** - Optimized indexes for symbol + time queries
- **Compression** - Automatic compression of older data chunks
- **Connection Pooling** - Shared async connection pool (2-10 connections)

**Batch Operations:**
- **Upsert Conflict Resolution** - ON CONFLICT DO UPDATE for idempotent inserts
- **Bulk Processing** - Single transaction per batch for efficiency
- **Auto-Timestamps** - created_at/updated_at managed automatically

### No-Op Behavior:

**AC2: Adapters no-op unless flag & DSN present**

The system safely degrades when not configured:

```python
# These calls return 0 and log debug messages when disabled
records_saved = await persist_prices([...])    # Returns 0
records_saved = await persist_sentiment([...]) # Returns 0  
records_saved = await persist_predictions([...]) # Returns 0

# Query operations return empty lists
prices = await db_adapter.query_prices("AAPL", start, end)  # Returns []
```

**Disable Conditions:**
- `ENABLE_TIMESCALE_PERSISTENCE=false` (default)
- `PG_DSN` environment variable not set
- `asyncpg` library not installed
- Database connection failures

### Testing:

```bash
# Run F10 TimescaleDB tests
python -m pytest tests/test_f10_timescale_persistence.py -v

# Test schema generation
python -c "from services.timeseries_database_schema import create_hypertables; print(list(create_hypertables().keys()))"
```

### Acceptance Criteria Met:
- ✅ **AC1**: `services/timeseries_database_schema.py` exports `create_hypertables()` DDL strings for prices/sentiment/predictions
- ✅ **AC2**: Adapters no-op unless flag & DSN present - complete feature isolation when disabled

## Vector Search (F11)

**Optional semantic news store for similarity search** - Provides vector-based news search using Qdrant, pgvector, or CSV fallback with placeholder embeddings.

### Key Features:
- **Multi-Backend Support** - Qdrant (primary), pgvector (PostgreSQL), CSV fallback
- **Automatic Backend Selection** - Tries backends in priority order, falls back gracefully
- **Placeholder Embeddings** - Deterministic hash-based embeddings when sentence-transformers unavailable
- **News Integration** - Seamless integration with news deduplicator for automatic storage
- **Similarity Search API** - `search_similar(query, symbol?)` for semantic news discovery

### Vector Backends:

**Backend Priority Order:**
1. **Qdrant** - Dedicated vector database (cloud/self-hosted)
2. **pgvector** - PostgreSQL extension for vector operations  
3. **CSV Fallback** - Local CSV storage with in-memory search

**Storage Schema:**
```python
# All backends store:
NewsItem(
    headline: str,           # Article headline/title
    url: str,               # Source URL
    published_at: datetime, # Publication timestamp  
    symbol: str,           # Stock symbol (AAPL, TSLA, etc.)
    embedding: List[float], # 384-dim vector embedding
    content_hash: str      # 16-char deduplication hash
)
```

### Configuration:

**Environment Variables:**
```bash
# F11 Vector Search Settings
ENABLE_VECTOR_SEARCH=false          # Enable vector search functionality
QDRANT_URL=http://localhost:6333     # Qdrant server URL
QDRANT_KEY=your_api_key              # Qdrant API key (optional)
PG_DSN=postgres://...                # PostgreSQL DSN for pgvector (shared with F10)
```

**Activation Requirements:**
- Feature flag: `ENABLE_VECTOR_SEARCH=true`  
- **For Qdrant**: `QDRANT_URL` environment variable set
- **For pgvector**: `PG_DSN` environment variable set + `vector` extension
- **CSV Fallback**: Always available (no external dependencies)

### Usage Examples:

**Initialize Vector Store:**
```python
from services.vector_store import initialize_vector_store

# Initialize with best available backend
success = await initialize_vector_store()
if success:
    print("Vector search ready")
```

**Store News Items:**
```python
from services.vector_store import NewsItem, store_news_embeddings
from datetime import datetime

# Create news items
news_items = [
    NewsItem(
        headline="Apple Reports Record Q4 Revenue",
        url="https://example.com/apple-q4-2023",
        published_at=datetime(2023, 12, 1),
        symbol="AAPL"
    ),
    NewsItem(
        headline="Tesla Delivers 500K Vehicles in Q4",
        url="https://example.com/tesla-deliveries-q4",
        published_at=datetime(2023, 12, 2), 
        symbol="TSLA"
    )
]

# Store with automatic embedding generation
stored_count = await store_news_embeddings(news_items)
print(f"Stored {stored_count} news items")
```

**Search Similar News:**
```python
from services.vector_store import search_similar_news

# Search across all symbols
results = await search_similar_news("quarterly earnings report", limit=5)

# Search within specific symbol
apple_results = await search_similar_news("revenue growth", symbol="AAPL", limit=3)

# Process results
for result in results:
    print(f"Headline: {result.news_item.headline}")
    print(f"Symbol: {result.news_item.symbol}")
    print(f"Similarity: {result.similarity_score:.2f}")
    print(f"URL: {result.news_item.url}")
    print("---")
```

### Automatic News Integration:

**Integration with News Deduplicator:**

The system automatically stores embeddings when processing news through the deduplicator:

```python
from services.news_deduplicator import NewsDeduplicator

deduplicator = NewsDeduplicator()

# Create fingerprint with symbol for vector search
fingerprint = deduplicator.create_content_fingerprint(
    url="https://example.com/apple-news",
    title="Apple Announces New iPhone",
    content="Apple today announced...",
    symbol="AAPL"  # F11: Symbol enables vector search integration
)

# Store fingerprint - automatically creates vector embedding if enabled
deduplicator.store_fingerprint(fingerprint)
```

### Embedding Generation:

**Placeholder System:**
- **Real Models**: Supports sentence-transformers (e.g., `all-MiniLM-L6-v2`)
- **Fallback Mode**: Deterministic hash-based 384-dimensional embeddings
- **Consistency**: Same text always produces same embedding
- **Normalization**: All embeddings are L2-normalized for cosine similarity

```python
from services.vector_store import EmbeddingGenerator

generator = EmbeddingGenerator()
await generator.initialize()

# Generate embeddings for headlines
headlines = ["Apple stock rises", "Tesla reports earnings"]
embeddings = await generator.generate_embeddings(headlines)

# Each embedding is 384-dimensional normalized vector
for i, embedding in enumerate(embeddings):
    print(f"'{headlines[i]}' -> {len(embedding)}-dim vector")
```

### Backend-Specific Features:

**Qdrant (Recommended):**
- **Native Vector DB**: Purpose-built for similarity search
- **Scalability**: Handles millions of vectors efficiently  
- **Cloud/Self-Hosted**: Flexible deployment options
- **Advanced Indexing**: HNSW algorithm for fast approximate search

**pgvector (PostgreSQL):**  
- **SQL Integration**: Familiar PostgreSQL environment
- **ACID Compliance**: Transactional vector operations
- **Hybrid Queries**: Combine vector and traditional SQL queries
- **Shared Infrastructure**: Uses same database as F10 TimescaleDB

**CSV Fallback:**
- **Zero Dependencies**: Works without external services
- **Development**: Perfect for testing and development
- **Persistence**: Data survives between sessions
- **Basic Search**: Text matching + embedding similarity

### Performance & Telemetry:

**Vector Store Statistics:**
```python
from services.vector_store import vector_store

stats = await vector_store.get_stats()
# Returns:
# {
#   "backend": "qdrant",           # Active backend
#   "available": true,             # Backend health
#   "vector_count": 1500,          # Total stored vectors
#   "embedding_dim": 384,          # Embedding dimensions
#   "initialized": true            # Store ready status
# }
```

**Health Monitoring:**
- **Backend Status**: Track which backend is active
- **Vector Counts**: Monitor storage growth
- **Error Tracking**: Log backend failures and fallbacks
- **Performance**: Track embedding generation and search latency

### No-Op Behavior:

**AC2: Complete feature isolation when disabled**

The system safely degrades when not configured:

```python
# These calls return 0/empty when disabled
stored = await store_news_embeddings([...])      # Returns 0
results = await search_similar_news("query")     # Returns []

# Automatic storage is skipped in news deduplicator
# No vector operations occur when ENABLE_VECTOR_SEARCH=false
```

**Disable Conditions:**
- `ENABLE_VECTOR_SEARCH=false` (default)
- No backend configuration (no QDRANT_URL or PG_DSN)
- Missing dependencies (qdrant-client, asyncpg, vector extension)
- Backend connection failures

### Installation & Dependencies:

**Required Dependencies:**
```bash
# Core (always available)
pip install numpy

# Optional backend dependencies
pip install qdrant-client          # For Qdrant support
pip install asyncpg                # For pgvector support  
pip install sentence-transformers  # For real embeddings (optional)
```

**PostgreSQL Setup (pgvector):**
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Testing:

```bash
# Run F11 vector search tests
python -m pytest tests/test_f11_vector_search.py -v

# Test vector store initialization
python -c "import asyncio; from services.vector_store import initialize_vector_store; print(asyncio.run(initialize_vector_store()))"

# Test embedding generation
python -c "import asyncio; from services.vector_store import EmbeddingGenerator; g=EmbeddingGenerator(); asyncio.run(g.initialize()); print(len(asyncio.run(g.generate_embeddings(['test']))[0]))"
```

### Limitations:

**Current Implementation:**
- **No Real Models**: Uses placeholder embeddings by default
- **Basic Search**: CSV fallback uses simple text similarity
- **Memory Usage**: CSV backend loads all data into memory  
- **No Clustering**: Single-node operation only

**Future Enhancements:**
- Pre-trained sentence-transformers integration
- Advanced search filters (date range, sentiment, etc.)
- Hybrid search (keyword + semantic)
- Batch processing optimizations

### Acceptance Criteria Met:
- ✅ **AC1**: When enabled, store headline, url, published_at, symbol, embedding
- ✅ **AC2**: Provide `search_similar(query, symbol?)` API for future use

## Forecasting Extensions (F12)

**Optional baseline forecasters and TimeGPT stub integration** - Extends existing RF/XGB ensemble with ARIMA/Prophet baselines and TimeGPT stub for comprehensive model comparison.

### Key Features:
- **Keep Existing Models** - RF/XGB ensemble predictions unchanged when flags off
- **ARIMA/Prophet Baselines** - Optional statistical forecasting models for comparison
- **TimeGPT Stub** - Mock integration for next-generation forecasting API
- **Comparison Table** - Side-by-side MAE/RMSE logging to `data/forecast_comparison.csv`
- **Pluggable Registry** - Extensible forecaster architecture for future models

### Model Portfolio:

**Existing Models (Always Active):**
- **Random Forest + XGBoost Ensemble** - Tree-based models with market/peer features
- **LSTM** - Neural network for time series patterns with sentiment integration
- **Combined Predictions** - Ensemble averaging of RF/XGB/LSTM forecasts

**Optional Baselines (F12):**
- **ARIMA** - AutoRegressive Integrated Moving Average with automatic order selection
- **Prophet** - Facebook's time series forecasting with trend/seasonality decomposition
- **TimeGPT Stub** - Mock API integration for foundation model forecasting

### Configuration:

**Environment Variables:**
```bash
# F12 Forecasting Extensions
ENABLE_ALT_FORECASTS=false          # Enable ARIMA/Prophet baselines
ENABLE_TIMEGPT_STUB=false           # Enable TimeGPT stub integration
```

**Dependencies (Optional):**
```bash
# For ARIMA support
pip install statsmodels

# For Prophet support  
pip install prophet

# For TimeGPT (stub only - no actual API calls)
# No additional dependencies required
```

### Usage Examples:

**Existing Prediction API (Unchanged):**
```python
from prediction import train_predict_stock

# AC2: With flags off, existing predictions unchanged
result = train_predict_stock("AAPL")

# Returns existing structure:
# {
#   "dates": [...],
#   "history": DataFrame,
#   "predictions": [RF/XGB/LSTM ensemble],
#   "confidence": float,
#   "red_flag": bool
# }
```

**Enhanced Predictions with F12:**
```python
# Enable baseline forecasts
os.environ['ENABLE_ALT_FORECASTS'] = 'true'
os.environ['ENABLE_TIMEGPT_STUB'] = 'true'

result = train_predict_stock("AAPL")

# Extended structure with F12 data:
# {
#   "dates": [...],
#   "history": DataFrame,
#   "predictions": [RF/XGB/LSTM ensemble],  # Unchanged
#   "confidence": float,
#   "red_flag": bool,
#   "baseline_forecasts": [ARIMA_result, Prophet_result],
#   "timegpt_forecast": TimeGPT_result,
#   "forecast_comparison": [comparison_data]
# }
```

**Direct Baseline Usage:**
```python
from services.forecasting.baselines import run_baseline_forecasts

# Prepare price data
prices_by_symbol = {
    'AAPL': [
        {'date': datetime(2023, 12, 1), 'close': 195.50},
        {'date': datetime(2023, 12, 2), 'close': 196.40},
        # ... more historical data
    ]
}

# Run baseline forecasts
results = run_baseline_forecasts(prices_by_symbol, forecast_horizon=7)

for symbol, forecasts in results.items():
    for forecast in forecasts:
        if forecast.success:
            print(f"{forecast.model_name} for {symbol}:")
            print(f"  Predictions: {forecast.predictions}")
            print(f"  MAE: {forecast.mae:.4f}")
            print(f"  Runtime: {forecast.runtime_ms}ms")
```

**TimeGPT Stub Usage:**
```python
from services.forecasting.timegpt_stub import run_timegpt_forecast

# Run TimeGPT forecasts (stub - no API key needed)
timegpt_results = run_timegpt_forecast(prices_by_symbol, forecast_horizon=7)

for symbol, result in timegpt_results.items():
    if result.success:
        print(f"TimeGPT forecast for {symbol}: {result.predictions}")
        print(f"API credits used: {result.api_credits_used}")
```

### Baseline Model Details:

**ARIMA (AutoRegressive Integrated Moving Average):**
- **Automatic Order Selection** - Tests (p,d,q) configurations, selects best AIC
- **Stationarity Handling** - Differencing for trend removal
- **Confidence Intervals** - Statistical bounds on predictions
- **Minimum Data** - Requires 30+ data points for reliable forecasts

**Prophet (Facebook's Time Series Model):**
- **Trend Decomposition** - Linear/logistic growth trends
- **Seasonality** - Automatic weekly/yearly pattern detection
- **Holiday Effects** - Built-in holiday calendar support (disabled in stub)
- **Missing Data Robust** - Handles gaps in time series gracefully

**TimeGPT Stub:**
- **Foundation Model Simulation** - Mock next-generation forecasting API
- **Rate Limiting** - Simulates 100 calls/hour API limits
- **Latency Simulation** - Realistic 1.5s response times
- **Credit Tracking** - Mock API credit usage (1 credit per forecast)

### Comparison Table (AC1):

**Automatic CSV Generation:**

The system automatically saves forecast comparisons to `data/forecast_comparison.csv`:

```csv
symbol,model,forecast_day,prediction,confidence,mae,rmse,runtime_ms,timestamp
AAPL,RF_XGB_Ensemble,1,195.50,0.85,,,,"2023-12-01 10:00:00"
AAPL,ARIMA,1,196.20,,2.45,3.21,1500,"2023-12-01 10:00:00"
AAPL,Prophet,1,194.80,,2.78,3.45,2200,"2023-12-01 10:00:00"
AAPL,TimeGPT_Stub,1,195.90,,2.15,2.95,1800,"2023-12-01 10:00:00"
```

**Comparison Analysis:**
```python
from prediction import get_forecast_comparison_summary

summary = get_forecast_comparison_summary()
print(f"Models compared: {summary['models']}")
print(f"Total forecasts: {summary['total_forecasts']}")

# Model performance comparison
for model, perf in summary['model_performance'].items():
    print(f"{model}: MAE={perf['avg_mae']:.4f}, Runtime={perf['avg_runtime_ms']:.0f}ms")
```

### Performance & Runtime Budget:

**Baseline Model Runtimes:**
- **ARIMA**: 1-3 seconds (order selection + fitting)
- **Prophet**: 2-5 seconds (trend decomposition + seasonality)
- **TimeGPT Stub**: 1.5 seconds (simulated API latency)
- **RF/XGB/LSTM**: <1 second (existing, unchanged)

**Resource Management:**
- **Timeout Protection** - ARIMA limited to 30s, Prophet to 60s
- **Memory Efficient** - Streaming data processing, minimal memory overhead  
- **Error Isolation** - Baseline failures don't affect main predictions
- **Concurrent Safe** - Thread-safe operation for multiple symbols

### No-Op Behavior (AC2):

**Complete Backward Compatibility:**

When feature flags are disabled, existing predictions remain unchanged:

```python
# Flags disabled (default)
result = train_predict_stock("AAPL")

# Returns identical structure to pre-F12:
# - Same RF/XGB/LSTM predictions
# - Same confidence calculations  
# - Same red flag logic
# - No additional processing overhead
# - No baseline forecast keys in result
```

**Graceful Degradation:**
- **Missing Dependencies** - ARIMA/Prophet gracefully skip when statsmodels/prophet unavailable
- **Insufficient Data** - Baselines return failure results, don't crash main pipeline
- **Runtime Errors** - Individual model failures logged but don't affect other forecasts
- **CSV Write Failures** - Comparison table errors logged but don't break predictions

### Installation & Dependencies:

**Core Dependencies (Always Required):**
```bash
# Already in requirements.txt
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
```

**Optional Baseline Dependencies:**
```bash
# For ARIMA forecasting
pip install statsmodels>=0.13.0

# For Prophet forecasting
pip install prophet>=1.1.0

# Verify installation
python -c "import statsmodels; import prophet; print('Baselines ready')"
```

**Installation Notes:**
- **Prophet Requirements** - May need `gcc` compiler on some systems
- **Statsmodels** - Generally installs without issues on most platforms
- **Fallback Mode** - System works without optional dependencies (baselines disabled)

### Testing:

```bash
# Run F12 forecasting tests
python -m pytest tests/test_f12_forecasting_extensions.py -v

# Test baseline forecasters
python -c "
import asyncio
from services.forecasting.baselines import run_baseline_forecasts
from datetime import datetime, timedelta

prices = {'AAPL': [{'date': datetime.now() - timedelta(days=i), 'close': 100+i} for i in range(50)]}
results = run_baseline_forecasts(prices)
print('Baseline test:', len(results))
"

# Test TimeGPT stub
python -c "
import asyncio
from services.forecasting.timegpt_stub import get_timegpt_status
print('TimeGPT status:', get_timegpt_status())
"

# Test comparison table generation
python -c "from prediction import get_forecast_comparison_summary; print(get_forecast_comparison_summary())"
```

### Idempotency & Reproducibility:

**Deterministic Results:**
- **Same Seeds → Same Outputs** - All models use consistent random seeds
- **ARIMA** - Deterministic optimization with fixed starting parameters
- **Prophet** - Consistent initialization and convergence criteria
- **TimeGPT Stub** - Hash-based deterministic "forecasts" for testing

**Reproducible Comparisons:**
- **Consistent Data Preparation** - Same preprocessing across all models
- **Fixed Validation** - 20% holdout for MAE/RMSE calculation
- **Timestamp Precision** - Millisecond-level runtime tracking
- **Error Handling** - Consistent failure modes and error messages

### Future Enhancements:

**Real TimeGPT Integration:**
- Replace stub with actual API calls when TimeGPT becomes available
- Add authentication and credit management
- Implement advanced prompting strategies

**Additional Baselines:**
- **ETS (Error, Trend, Seasonality)** - Alternative statistical forecasting
- **VAR (Vector Autoregression)** - Multi-variate time series models
- **Neural Prophet** - Prophet with neural network components

**Advanced Comparison:**
- **Cross-Validation** - Time series CV for robust performance assessment
- **Statistical Significance** - Diebold-Mariano tests for forecast comparison
- **Ensemble Weights** - Dynamic model combination based on recent performance

### Acceptance Criteria Met:
- ✅ **AC1**: Comparison table saved to `data/forecast_comparison.csv`
- ✅ **AC2**: With flags off, existing predictions unchanged

### Testing:

```bash
# Run F09 async I/O tests
python -m pytest tests/test_f09_async_io.py -v

# Run F09 integration tests
python -m pytest tests/integration/test_f09_async_integration.py -v
```

**Test Coverage:**
- Retry sequence testing with exponential backoff
- Circuit breaker state transitions (CLOSED→OPEN→HALF_OPEN→CLOSED)
- Cache hit/miss scenarios and TTL expiration
- Semaphore concurrency limits and asyncio.gather patterns
- Error handling and graceful degradation
- Integration with news_scraper.py and multi_source_data_manager.py

### Security & Idempotency:

**Security:**
- **No credentials in cache keys** - Authorization headers excluded from cache key generation
- **Memory-only caching** - No sensitive data persisted to disk
- **Circuit breaker isolation** - Failing services don't impact others

**Idempotency:**
- **Deterministic cache keys** - Same URL+params always produce same key
- **GET-only caching** - Only idempotent requests cached
- **Consistent retry behavior** - Same inputs produce same retry sequences (excluding jitter)

### Acceptance Criteria Met:
- ✅ **AC1**: Daily run resilient to transient failures; retries capped; failures logged but do not abort
- ✅ **Asyncio Gather + Semaphore Limits**: Concurrent network calls with configurable MAX_CONCURRENCY  
- ✅ **Exponential Backoff**: Smart retry logic with jitter and maximum delay caps
- ✅ **Circuit Breakers**: Per-host failure detection and automatic recovery
- ✅ **Local Result Caching**: Idempotent cache with TTL=24h and LRU eviction

### Alpha Vantage (Primary)

Premium data source requiring API key. Set `ALPHA_VANTAGE_KEY` in your config.

### Additional Sources  

Enable premium sources with `ENABLE_PAID_SOURCES=true`:
- **Finnhub**: Real-time data and fundamentals
- **Polygon**: High-quality financial data  
- **EODHD**: End-of-day historical data

## Feature Flags

Control functionality with environment variables (all default to `false`):

```bash
# Core Features
ENABLE_YF_PRICES=false              # Use yfinance for historical data
ENABLE_YF_DAILY_REFRESH=false       # Once-per-day yfinance refresh guard
ENABLE_YF_PROFILES=false            # Fetch company profiles via yfinance
ENABLE_PORTFOLIO_ANALYTICS=false    # Sector allocation & beta analysis
ENABLE_FINBERT_PIPELINE=false       # Advanced sentiment analysis
ENABLE_90_DAY_SENTIMENT=false       # Extended sentiment windows
ENABLE_SMART_ALERTS=false           # Smart alerts for price moves, sentiment, and earnings

# Analytics Features (F13)
ENABLE_CORRELATION=false            # Portfolio correlation analysis with heatmaps
ENABLE_GNN_SCAFFOLD=false          # Graph Neural Network data loaders

# Earnings Features (F16)
ENABLE_EARNINGS_READS=false         # Upcoming earnings schedule and analysis

# Symbol Management (F17)
ENABLE_SYMBOL_INTAKE=false          # Dynamic symbol intake and registry management

# API Features
ENABLE_API_ENDPOINTS=false          # REST API server endpoints
ENABLE_SYMBOL_INTAKE=false          # Symbol intake API endpoints
ENABLE_RECOS=false                  # Recommendations API endpoints
ENABLE_EARNINGS_READS=false         # Earnings analysis API endpoints

# Async I/O Features
ENABLE_ASYNC_IO=false               # Async I/O with circuit breakers and caching

# Data Sources
ENABLE_MULTISOURCE_PRICES=false     # Multi-source price aggregation
ENABLE_PAID_SOURCES=false           # Premium data sources
ENABLE_NEWSAPI_INGESTION=false      # NewsAPI integration

# Debugging
ENABLE_YF_BACKOFF_DEBUG=false       # Detailed yfinance retry logs
ENABLE_DEBUG_MODE=false             # System-wide debug logging
```

## File Structure

```
├── data/
│   ├── yf_bulk_cache/          # yfinance cached data
│   ├── av_bulk_cache/          # Alpha Vantage cache
│   └── portfolio_metrics.csv   # Portfolio analytics
├── services/
│   ├── data_sources/
│   │   └── yfinance_provider.py
│   └── retry_policies.py       # Rate limiting utilities
├── config/
│   ├── secrets.env             # API keys & settings
│   └── feature_flags.py        # Feature flag management
└── analytics/
    ├── portfolio_metrics.py    # Portfolio analysis
    ├── correlation.py          # F13: Correlation analysis & heatmaps
    └── gnn_scaffold.py         # F13: Graph Neural Network data loaders
```

## Correlation Analysis (F13)

Compute Pearson correlation matrices across portfolio stocks with optional heatmap visualization.

### Configuration
```bash
ENABLE_CORRELATION=true    # Enable correlation analysis
```

### Features
- **Correlation Matrix**: Pearson correlation computation across aligned return windows  
- **Heatmap Generation**: Publication-ready correlation heatmaps saved as PNG
- **CSV Export**: Correlation matrices saved to `data/correlation.csv`
- **Email Integration**: Optional correlation heatmaps in email reports
- **Statistical Analysis**: Correlation statistics and outlier detection

### Usage
```python
from analytics.correlation import compute_portfolio_correlations, generate_correlation_heatmap

# Compute correlation matrix
correlation_matrix = compute_portfolio_correlations(price_data, lookback_days=252)

# Generate heatmap  
success = generate_correlation_heatmap(
    price_data,
    title="Portfolio Correlation Heatmap",
    save_path="charts/corr_heatmap.png"
)
```

### Output Files
- `data/correlation.csv`: Correlation matrix with metadata
- `charts/corr_heatmap.png`: Correlation heatmap visualization

## GNN Scaffold (F13)

Graph Neural Network data loader infrastructure for financial graph analysis.

### Configuration
```bash
ENABLE_GNN_SCAFFOLD=true    # Enable GNN data loaders
```

### Features
- **Graph Data Structures**: Nodes, edges, and adjacency matrices
- **Correlation Graphs**: Build graphs from stock correlations
- **Sector Graphs**: Connect stocks within same sectors
- **Extensible Architecture**: Abstract base classes for custom loaders
- **No Heavy Dependencies**: Lightweight scaffold without ML frameworks

### Usage
```python
from analytics.gnn_scaffold import GNNDataManager

manager = GNNDataManager()

# Create correlation-based graph
graph = manager.create_correlation_graph(
    symbols=['AAPL', 'MSFT', 'TSLA'], 
    price_data=price_data,
    correlation_threshold=0.5
)

# Create sector-based graph  
graph = manager.create_sector_graph(
    symbols=['AAPL', 'MSFT'],
    sector_data={'AAPL': 'Technology', 'MSFT': 'Technology'},
    price_data=price_data
)
```

# BEGIN F14_README
## Development Workflow

### Pre-commit Setup
Install and configure pre-commit hooks for consistent code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files (optional)
pre-commit run --all-files
```

The pre-commit configuration includes:
- **Black**: Code formatting with 127 character line length
- **isort**: Import sorting compatible with Black
- **flake8**: Linting with complexity checks
- **Security**: Bandit security scanning
- **File checks**: Trailing whitespace, YAML validation, large files

### Running Tests Locally

```bash
# Run all tests except e2e
pytest -k "not e2e"

# Run with coverage
pytest -k "not e2e" --cov=. --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests  
pytest tests/integration/
```

### CI Pipeline

The CI pipeline runs on:
- **Pull Requests**: Lint, format check, and non-e2e tests across Python 3.10-3.12
- **Nightly Schedule**: E2E tests with network mocking at 2 AM UTC
- **Push/Release**: Full deployment pipeline (existing)

All feature flags are disabled during CI smoke tests to ensure baseline functionality.
# END F14_README

## Earnings Analysis (F16)

Track upcoming earnings events for your portfolio and watchlist with implied move analysis.

### Configuration
```bash
ENABLE_EARNINGS_READS=true          # Enable earnings schedule tracking
EARNINGS_WINDOW_DAYS=14             # Days ahead to look for earnings (optional, default: 14)
```

### Features
- **Earnings Schedule**: Automatically fetch upcoming earnings for portfolio/watchlist symbols
- **Implied Moves**: Options-based implied volatility calculations for earnings reactions
- **Direction Analysis**: Bullish/Bearish/Neutral predictions with confidence scores  
- **Email Integration**: Clean table showing date, implied move %, and predicted direction
- **CSV Export**: Earnings data saved to `data/earnings_schedule.csv` with atomic writes

### Email Report Integration
When enabled, email reports include an "Upcoming Earnings" section with:
- **Symbol**: Stock ticker
- **Date**: Earnings announcement date
- **Implied Move**: Expected price movement percentage from options
- **Direction**: Predicted direction (Bullish/Bearish/Neutral) with color coding

### Data Files
- `data/earnings_schedule.csv`: Complete earnings schedule with analysis
- Atomic writes ensure data consistency during pipeline runs
- CSV includes: symbol, earnings_date, implied_move_pct, direction, confidence, risk_level

### Configuration Options
```bash
# Earnings window configuration
EARNINGS_WINDOW_DAYS=21             # Look ahead 21 days instead of default 14
```

## Symbol Intake (F17)

Dynamically add new ticker symbols to your pipeline without modifying config files.

### Configuration
```bash
ENABLE_SYMBOL_INTAKE=true           # Enable symbol intake processing
SYMBOL_INTAKE_LIST=NVDA,AMD,CRM     # Comma-separated list (optional)
SYMBOL_INTAKE_CSV=data/candidates.csv # Path to candidates CSV (optional)
```

### Features
- **Dynamic Symbol Addition**: Add symbols via environment variables or CSV files
- **Validation & Deduplication**: Validates ticker format and prevents duplicates
- **Registry Management**: Maintains persistent symbol registry with atomic writes
- **Pipeline Integration**: New symbols automatically included in current pipeline run
- **Audit Logging**: Detailed acceptance/rejection counts and processing metrics

### Usage Methods

#### Environment Variable
```bash
export SYMBOL_INTAKE_LIST="NVDA,AMD,CRM,INTC"
python main.py
```

#### CSV File
Create `data/symbol_candidates.csv`:
```csv
symbol
NVDA
AMD
CRM
INTC
```

```bash
export SYMBOL_INTAKE_CSV="data/symbol_candidates.csv"
python main.py
```

### Symbol Validation
Valid ticker formats:
- **Standard**: `AAPL`, `MSFT`, `GOOGL` (1-5 uppercase letters)
- **Class shares**: `BRK.B`, `BF-A` (dot or hyphen with 1-2 letters)

Invalid formats rejected:
- Numbers: `123`, mixed case: `aapl`
- Special characters: `AA@PL`, too long: `TOOLONG`

### Registry Management
- **File**: `data/tickers_registry.csv` with schema: `symbol,source,added_at`
- **Atomic writes**: Uses temp files to prevent corruption
- **Deduplication**: Against existing config (PORTFOLIO/WATCHLIST) and registry
- **Deterministic ordering**: Symbols sorted alphabetically for consistency

### Rollback
To remove symbols from intake:
```bash
# Edit registry CSV directly or programmatically filter
python -c "
import pandas as pd
df = pd.read_csv('data/tickers_registry.csv')
df = df[df['symbol'] != 'UNWANTED_SYMBOL']
df.to_csv('data/tickers_registry.csv', index=False)
"
```

## Compliance & Safety

### Disclaimers
All email reports and API responses include configurable disclaimers to ensure compliance and proper user expectations.

#### Configuration
```bash
# Environment variables for disclaimer customization
DISCLAIMER_TEXT="Your custom disclaimer text here"  # Optional
COMPANY_NAME="Your Company Name"                     # Optional
```

#### Default Behavior
- **Email Reports**: Include header and footer disclaimers with warning styling
- **API Root**: Returns disclaimer in JSON response with compliance notice
- **Default Text**: "This report/API provides research and educational data only. Not financial advice."

#### Disclaimer Locations
- Email header: Prominent warning box with yellow border
- Email footer: Company attribution and generation timestamp
- API root (`/`): JSON response with disclaimer, version, and endpoint information

### Security Features

#### Log Redaction
All audit logs automatically redact sensitive information:
- **Email addresses**: Replaced with `[REDACTED]`
- **API keys**: Show only first/last 4 characters (e.g., `abc1***xyz9`)
- **Secrets**: Password, token, and key assignments masked
- **Sensitive fields**: Automatic detection of credential field names

#### Usage
```python
from utils.redact import redact, safe_log_format

# Automatic redaction
safe_message = redact("User admin@company.com logged in with key abc123def456")
# Result: "User [REDACTED] logged in with key abc1***f456"

# Safe logging with context
logger.info(safe_log_format("Authentication failed", user="admin@test.com", api_key="secret"))
```

### Compliance Notes
- Disclaimers are always shown regardless of feature flag state
- Email reports include prominent warnings about automated analysis limitations
- API responses include compliance notices about consulting financial advisors
- All sensitive data is automatically scrubbed from system logs

## Microservices (Optional)

Enable microservices mode for containerized deployment with Docker Compose profiles.

### Configuration
```bash
ENABLE_MICROSERVICES_MODE=true
```

### Commands
```bash
# Start microservices with profile
docker compose --profile micro up

# Build and start services
docker compose --profile micro up --build

# Stop microservices
docker compose --profile micro down
```

### Services
- **API Gateway** - `http://localhost:8000` - Request routing and health checks
- **News Scraping Service** - `http://localhost:8001` - RSS feed aggregation

### Health Checks
```bash
# Check API Gateway
curl http://localhost:8000/healthz

# Check News Service
curl http://localhost:8001/healthz

# Get news via gateway
curl http://localhost:8000/news?limit=5
```

## Troubleshooting

### yfinance Rate Limits
```
YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')
```
**Solution**: This is expected with heavy usage. The system will:
1. Use cached data from previous runs
2. Retry with exponential backoff  
3. Fall back to Alpha Vantage if needed
4. Continue pipeline execution

### Empty Charts/Reports
**Cause**: No cached data available and network calls failing.
**Solution**: 
1. Ensure at least one data source is properly configured
2. Check API keys in `config/secrets.env`
3. Run with `ENABLE_DEBUG_MODE=true` for detailed logs

## Development

### Adding New Data Sources

1. Create provider in `services/data_sources/`
2. Add feature flag in `config/feature_flags.py` 
3. Wire into `services/multi_source_data_manager.py`
4. Update `prediction.py` integration
5. Add tests and documentation

### Testing

```bash
# Run with a single stock to test quickly
python -c "
from services.data_sources.yfinance_provider import create_yfinance_provider
provider = create_yfinance_provider()
data = provider.fetch_history(['AAPL'])
print(f'Retrieved {len(data)} datasets')
"
```

## License

This project is for educational and research purposes only. Not intended for production trading decisions.

---

## Support

- Check feature flags if functionality is missing
- Enable debug mode for troubleshooting: `ENABLE_DEBUG_MODE=true`
- Review logs for rate limiting or API issues
- Ensure all required API keys are configured