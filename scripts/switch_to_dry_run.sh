#!/bin/bash
# Switch to DRY_RUN mode
set -e

echo "🔧 Switching to DRY_RUN mode..."

# Export dry run environment
export $(grep -v '^#' .env.dryrun | xargs)
echo "✅ Environment variables loaded from .env.dryrun"

# Run preflight checks
echo "🚀 Running preflight checks..."
python3 scripts/preflight_dry_run.py

if [ $? -eq 0 ]; then
    echo "✅ DRY_RUN mode activated - safe to run tests"
    echo ""
    echo "Available commands:"
    echo "  pytest --collect-only -q    # Check test collection"
    echo "  pytest -q --maxfail=1       # Run dry tests"
    echo "  pytest -v tests/unit/        # Run unit tests only"
else
    echo "❌ Preflight checks failed - fix issues before running"
    exit 1
fi