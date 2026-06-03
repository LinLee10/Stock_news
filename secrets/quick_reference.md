# API Keys Quick Reference

## Loading Secrets in Code

```python
from secrets.load_secrets import get_financial_api_key, get_news_api_key, get_government_api_key

# Get specific API keys
newsapi_key = get_news_api_key('newsapi', 'production')
finnhub_key = get_financial_api_key('finnhub', 'production')  
fred_key = get_government_api_key('fred', 'production')
```

## Available Services

### News Services
- **NewsAPI**: `get_news_api_key('newsapi')`
- **GNews**: `get_news_api_key('gnews')`

### Financial Data Services  
- **Finnhub**: `get_financial_api_key('finnhub')`
- **Twelve Data**: `get_financial_api_key('twelvedata')`
- **Tiingo**: `get_financial_api_key('tiingo')`
- **Polygon**: `get_financial_api_key('polygon')`
- **Marketstack**: `get_financial_api_key('marketstack')`
- **EOD HD**: `get_financial_api_key('eodhd')`
- **Nasdaq Data Link**: `get_financial_api_key('nasdaq')`

### Alpaca Trading (Special - needs both keys)
```python
from secrets.load_secrets import get_alpaca_credentials
creds = get_alpaca_credentials('production')
api_key = creds['api_key']
secret_key = creds['secret_key']
```

### Government Data Services
- **FRED** (Federal Reserve): `get_government_api_key('fred')`
- **BEA** (Bureau of Economic Analysis): `get_government_api_key('bea')`
- **BLS** (Bureau of Labor Statistics): `get_government_api_key('bls')`

## Environment Loading

```bash
# Load production keys
source secrets/production.env

# Load development keys  
source secrets/development.env
```

## Your Actual API Keys

| Service | Key ID | Usage |
|---------|--------|-------|
| NewsAPI | `ec1f3d...ddb1a` | News articles |
| Finnhub | `d2rmiq...7eg` | Stock data, WebSocket |
| Twelve Data | `d964eb...4056` | Financial data |
| Alpaca Key | `PKZX86...9FGM` | Trading API |
| Alpaca Secret | `l0CbaX...CHfZ` | Trading API |
| Tiingo | `8eb637...f457` | Historical data |
| Polygon | `BttXTe...eC2g` | Real-time data |
| Marketstack | `42e38e...cf39` | Market data |
| EOD HD | `68b77a...8704` | End of day data |
| Nasdaq | `bcDX53...idP6` | Market data |
| GNews | `80a843...2de0` | Google News |
| FRED | `1e6fcf...22ac6` | Federal Reserve data |
| BEA | `2ADCA4...9357` | Economic indicators |
| BLS | `b32a35...898f` | Labor statistics |

## Security Notes

⚠️ **Never commit `production.env`** - it's gitignored  
🔄 **Rotate keys regularly** - especially if compromised  
📊 **Monitor usage** - track API call quotas  
🔒 **Use least privilege** - only necessary permissions