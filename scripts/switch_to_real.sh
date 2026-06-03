#!/bin/bash
# Switch to REAL mode (production/integration)
set -e

echo "🌐 Switching to REAL mode..."

# Unset dry run environment
unset DRY_RUN
unset DISABLE_EMAIL_SEND
unset FORCE_CACHE_ONLY
unset NEWS_SOURCE
unset PRICE_SOURCE
unset MATPLOTLIB_BACKEND

# Load production environment if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment variables loaded from .env"
else
    echo "⚠️  No .env file found - using default production settings"
fi

echo "✅ REAL mode activated - network calls enabled"
echo ""
echo "Available commands:"
echo "  pytest tests/integration/    # Run integration tests"
echo "  pytest tests/ -m 'not e2e'   # Run all except e2e tests"
echo "  python3 main.py              # Run main application"