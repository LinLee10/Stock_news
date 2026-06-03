# API Keys and Secrets Management

This directory contains API keys and secrets for the Stonk News application.

⚠️ **SECURITY WARNING**: Never commit actual API keys to version control!

## File Structure

- `production.env` - Production API keys (DO NOT COMMIT)
- `development.env` - Development/testing API keys 
- `template.env` - Template with placeholder values
- `load_secrets.py` - Python utility to load secrets by environment

## Usage

### Loading in Python
```python
from secrets.load_secrets import load_secrets

# Load production secrets
secrets = load_secrets('production')
newsapi_key = secrets.get('NEWSAPI_KEY')

# Load development secrets  
secrets = load_secrets('development')
```

### Loading in Shell
```bash
# Load production environment
source secrets/production.env

# Load development environment
source secrets/development.env
```

## Security Best Practices

1. **Never commit actual keys** - Use `.gitignore` to exclude `production.env`
2. **Rotate keys regularly** - Update API keys periodically
3. **Use least privilege** - Only grant necessary permissions
4. **Monitor usage** - Track API usage for anomalies
5. **Separate environments** - Different keys for dev/prod

## API Key Management

Each service has its own prefix for easy identification:
- `NEWSAPI_*` - News API keys
- `FINANCIAL_*` - Financial data API keys  
- `GOVERNMENT_*` - Government data API keys
- `SOCIAL_*` - Social media API keys