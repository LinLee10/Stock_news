# Dry-Run Test Commands

## Setup Commands (One-time)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python3 -m pip install -U pip
python3 -m pip install -r requirements-dev.txt
```

## Quick Mode Switch Commands

### Switch to DRY_RUN Mode
```bash
# Easy switch (includes preflight checks)
./scripts/switch_to_dry_run.sh

# Manual switch
export $(grep -v '^#' .env.dryrun | xargs)
python3 scripts/preflight_dry_run.py
```

### Switch to REAL Mode
```bash
# Easy switch
./scripts/switch_to_real.sh

# Manual switch  
unset DRY_RUN DISABLE_EMAIL_SEND FORCE_CACHE_ONLY
export $(grep -v '^#' .env | xargs) 2>/dev/null || true
```

## Test Commands (DRY_RUN Mode)

```bash
# Verify test collection
pytest --collect-only -q

# Run all dry-run safe tests
pytest -q --maxfail=1

# Run specific test categories
pytest tests/unit/ -v          # Unit tests only
pytest -m "not e2e" -v         # Skip e2e tests
pytest tests/test_no_network.py -v  # Network blocking test

# Run with coverage
pytest --cov=services --cov=fakes -v
```

## Verification Commands

```bash
# Check network is blocked
python3 -c "
import socket
try:
    s = socket.socket()
    s.connect(('8.8.8.8', 80))
    print('❌ Network NOT blocked')
except:
    print('✅ Network blocked')
"

# Check fixtures exist
ls -la tests/fixtures/rss/

# Check email artifacts
ls -la artifacts/emails/
```

## Real Mode Commands

```bash
# Run integration tests
pytest tests/integration/ -v

# Run all tests except e2e
pytest -m "not e2e" -v

# Run main application
python3 main.py
```

## Monkeypatched Import Paths

When `DRY_RUN=1`, these imports are automatically patched:

- `services.alpha_vantage_manager.AlphaVantageManager` → `fakes.alpha_vantage.FakeAlphaVantageManager`
- `yfinance.download` → `fakes.yfinance.fake_download`
- `yfinance.Ticker` → `fakes.yfinance.FakeTicker`  
- `services.email_sender.EmailSender` → `fakes.email.FakeEmailSender`
- `feedparser.parse` → `fakes.rss.parse`
- `structlog.get_logger` → `fakes.structlog_shim.get_logger`
- `openai.ChatCompletion.create` → `fakes.llm.FakeLLMClient().chat_completion`
- `socket.socket` → Blocked with ConnectionError

## Acceptance Criteria Verification

✅ Network completely blocked - attempts fail with clear errors  
✅ RSS uses local fixtures only (`tests/fixtures/rss/*.xml`)  
✅ Email writes to `artifacts/emails/` - no network sending  
✅ Time frozen at 2024-01-15 10:00:00 for determinism  
✅ Random seed fixed at 42  
✅ All external APIs return fake but realistic data  
✅ Tests pass without external dependencies  

## Troubleshooting

If preflight fails:
```bash
# Check DRY_RUN is set
echo $DRY_RUN

# Check fixtures
python3 -c "from pathlib import Path; print(list(Path('tests/fixtures/rss').glob('*.xml')))"

# Check imports
python3 -c "from fakes.alpha_vantage import FakeAlphaVantageManager; print('✅ Fakes importable')"
```