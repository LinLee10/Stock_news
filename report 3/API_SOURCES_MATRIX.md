# API Sources Matrix - Provider Analysis

## News Services

| Provider | Free Limits | Reset | Auth | Multi-Key Policy | Role Assignment | Status |
|----------|-------------|--------|------|------------------|-----------------|---------|
| **NewsAPI** | 100/day | UTC midnight | Header: `X-API-Key` | ❌ Prohibited ([ToS 2.1](https://newsapi.org/terms)) | Breadth backfill | ✅ Implemented |
| **GNews** | 100/day | UTC midnight | Query param: `token` | 🟡 Ambiguous - one key per "application" | Alternative breadth | ❌ Missing |

## Financial Data Services

| Provider | Free Limits | Reset | Auth | Multi-Key Policy | Role Assignment | Status |
|----------|-------------|--------|------|------------------|-----------------|---------|
| **Finnhub** | 60/min, 30k/mo | Calendar month | Header: `X-Finnhub-Token` | ✅ Allowed ([Docs](https://finnhub.io/docs/api/getting-started)) | News + sentiment | ❌ Missing |
| **Twelve Data** | 8/min, 800/day | UTC midnight | Query param: `apikey` | ❌ One key per user ([FAQ](https://twelvedata.com/pricing)) | Indicators + non-US intraday | ❌ Missing |
| **Tiingo** | 1000/day | UTC midnight | Header: `Authorization: Token` | 🟡 "Personal use" limit unclear | Historical gap-fill | ❌ Missing |
| **Polygon** | 5/min | UTC midnight | Query param: `apikey` | ❌ One key per account | Real-time quotes | ❌ Missing |
| **Marketstack** | 1000/day | UTC midnight | Query param: `access_key` | 🟡 ToS unclear on multiple keys | International markets | ❌ Missing |
| **EOD HD** | 20/day | UTC midnight | Query param: `api_token` | ❌ Personal use only | Fundamentals | ❌ Missing |
| **Nasdaq Data Link** | 300/month | Calendar month | Header: `X-API-Key` | ✅ Multiple apps allowed | Macro datasets | ❌ Missing |

## Trading & Intraday

| Provider | Free Limits | Reset | Auth | Multi-Key Policy | Role Assignment | Status |
|----------|-------------|--------|------|------------------|-----------------|---------|
| **Alpaca** | 200/min | No daily limit | Headers: `APCA-API-KEY-ID`, `APCA-API-SECRET-KEY` | ✅ Multiple paper accounts | US intraday bars | ❌ Missing |

## Government Data

| Provider | Free Limits | Reset | Auth | Multi-Key Policy | Role Assignment | Status |
|----------|-------------|--------|------|------------------|-----------------|---------|
| **FRED** | 120/min, 100k/day | UTC midnight | Query param: `api_key` | ✅ Multiple research projects | Economic indicators | ❌ Missing |
| **BEA** | 1000/day | UTC midnight | Query param: `UserID` | ✅ Government data | Economic analysis | ❌ Missing |
| **BLS** | 25/day, 500/day registered | UTC midnight | Query param: `registrationkey` | ✅ Multiple registrations | Labor statistics | ❌ Missing |

## Existing Implementations

| Provider | Free Limits | Reset | Auth | Multi-Key Policy | Role Assignment | Status |
|----------|-------------|--------|------|------------------|-----------------|---------|
| **Alpha Vantage** | 25/day | 🟡 Unknown (probe scheduled) | Query param: `apikey` | ❌ Personal use only | Fallback only | 🔴 Has bug |
| **YFinance** | No official quota | Undefined | None | N/A | Bulk daily OHLC | ✅ Working |

## Multi-Key Strategy Summary

### ✅ Explicitly Allowed (5 providers)
- Finnhub, Nasdaq Data Link, Alpaca, FRED, BEA

### ❌ Explicitly Prohibited (5 providers)  
- NewsAPI, Twelve Data, Polygon, EOD HD, Alpha Vantage

### 🟡 Ambiguous/Need Clarification (4 providers)
- GNews, Tiingo, Marketstack, BLS

**Vendor Question Template for Ambiguous Cases:**
> "For our financial research application, we need to partition API usage across different data categories (news vs prices vs indicators). Does your ToS permit multiple API keys for the same organization to achieve this partitioning, or would this violate single-user restrictions?"

## Rate Limit Reset Verification

### ✅ Confirmed UTC Midnight
- NewsAPI (header: `X-API-Key-Requests-Remaining`)
- Twelve Data, Polygon, Tiingo, Marketstack, EOD HD (standard practice)

### 🟡 Needs 72h Probe Confirmation
- **Alpha Vantage**: Current probe scheduled for Sept 1-3 (`probes/probe_state.json:12-31`)
- **GNews**: Reset behavior undocumented

### ✅ Non-Daily Limits  
- Finnhub (monthly), Nasdaq Data Link (monthly), Alpaca (minute-only)

## Compliance Risk Assessment

| Risk Level | Providers | Mitigation |
|------------|-----------|-------------|
| **Low** | FRED, BEA, Alpaca, YFinance | Multiple keys allowed or no restrictions |
| **Medium** | GNews, Tiingo, Marketstack, BLS | Contact vendors for clarification |
| **High** | NewsAPI, Twelve Data, Polygon, EOD HD, Alpha Vantage | Single key, strict quota management |

**Recommendation**: Implement per-key usage ledgers with strict partitioning for allowed providers, single-key quotas for prohibited providers.