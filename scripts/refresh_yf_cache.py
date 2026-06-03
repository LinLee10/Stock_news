#!/usr/bin/env python3
"""
YFinance Cache Refresh CLI

Optional CLI tool to manually trigger yfinance cache refresh.
Used by main.py when enable_yf_daily_refresh=True.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.yf_refresh_guard import YFDailyRefreshGuard
from config.tickers import PORTFOLIO, WATCHLIST
from config.feature_flags import is_yf_prices_enabled, is_yf_backoff_debug_enabled

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Refresh YFinance cache once per day')
    
    parser.add_argument(
        '--symbols',
        nargs='*',
        help='Symbols to refresh (default: PORTFOLIO + WATCHLIST)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh even if already done today'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true', 
        help='Show what would be done without executing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show refresh status and exit'
    )
    
    return parser.parse_args()


def get_default_symbols():
    """Get default symbols from config."""
    try:
        return sorted(set(PORTFOLIO + WATCHLIST))
    except Exception as e:
        logger.error(f"Failed to load default symbols: {e}")
        return ['AAPL', 'MSFT', 'GOOGL']  # Fallback


def show_status(guard: YFDailyRefreshGuard, symbols: list):
    """Show current refresh status."""
    manifest = guard._load_manifest()
    
    print("=" * 50)
    print("YFinance Cache Refresh Status")
    print("=" * 50)
    
    if manifest is None:
        print("Status: Never run")
        print(f"Symbols to refresh: {len(symbols)}")
        return
    
    print(f"Date: {manifest.date_utc}")
    print(f"Status: {manifest.status}")
    print(f"Attempts: {manifest.attempts}")
    print(f"Success: {manifest.success_count}/{len(manifest.symbols)}")
    print(f"Failed: {manifest.failed_count}/{len(manifest.symbols)}")
    print(f"Last update: {manifest.last_update}")
    print(f"Config key: {manifest.key}")
    
    # Check if symbols changed
    symbols_hash = guard._compute_symbols_hash(symbols)
    if symbols_hash != manifest.symbols_hash:
        print("⚠️  Symbols have changed since last run")
        print(f"   Cached: {len(manifest.symbols)} symbols")
        print(f"   Current: {len(symbols)} symbols")
    
    # Check if within window
    if guard._is_within_window():
        print("✅ Within refresh window")
    else:
        print(f"⏰ Before refresh window (hour {guard.config['window_hour']} UTC)")


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Set up logging level
    if args.verbose or is_yf_backoff_debug_enabled():
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if yfinance is enabled
    if not is_yf_prices_enabled():
        logger.warning("YFinance is disabled (ENABLE_YF_PRICES=false)")
        print("Enable yfinance with ENABLE_YF_PRICES=true in config/secrets.env")
        return 1
    
    # Get symbols
    symbols = args.symbols if args.symbols else get_default_symbols()
    logger.info(f"Working with {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    
    # Create guard
    guard = YFDailyRefreshGuard()
    
    # Show status if requested
    if args.status:
        show_status(guard, symbols)
        return 0
    
    # Dry run
    if args.dry_run:
        should_refresh, manifest = guard._should_refresh(symbols)
        print("=" * 50)
        print("Dry Run - YFinance Cache Refresh")
        print("=" * 50)
        print(f"Symbols: {len(symbols)}")
        print(f"Would refresh: {should_refresh}")
        
        if manifest:
            print(f"Current status: {manifest.status}")
            print(f"Attempts today: {manifest.attempts}")
        
        return 0
    
    # Force refresh by manipulating manifest
    if args.force:
        logger.info("Force refresh requested - clearing today's manifest")
        manifest = guard._load_manifest()
        if manifest and guard._is_today_utc(manifest.date_utc):
            manifest.status = "pending"
            manifest.attempts = 0
            guard._save_manifest(manifest)
    
    # Run refresh
    print("Starting YFinance cache refresh...")
    result = guard.run_once_per_day(symbols)
    
    print("=" * 50)
    print("Refresh Results")
    print("=" * 50)
    print(f"Status: {result['status']}")
    
    if result['status'] == 'skipped':
        print(f"Reason: {result.get('reason', 'unknown')}")
        if result.get('cache_available'):
            print("Previous cache data available")
    else:
        print(f"Attempts: {result.get('attempts', 0)}")
        print(f"Success: {result.get('success_count', 0)}/{result.get('symbols_requested', 0)}")
        print(f"Failed: {result.get('failed_count', 0)}/{result.get('symbols_requested', 0)}")
    
    # Return appropriate exit code
    if result['status'] in ['done', 'skipped']:
        return 0
    else:
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)