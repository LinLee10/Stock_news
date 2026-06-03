#!/usr/bin/env python3
"""Preflight checks for DRY_RUN mode"""
import os
import sys
import socket
from pathlib import Path

def check_dry_run_env():
    """Verify DRY_RUN=1 is set"""
    if os.getenv('DRY_RUN') != '1':
        print("❌ DRY_RUN=1 not set")
        return False
    print("✅ DRY_RUN=1 confirmed")
    return True

def check_socket_blocked():
    """Verify socket connections fail"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        s.close()
        print("❌ Socket not blocked - network access still available")
        return False
    except (socket.error, OSError):
        print("✅ Socket blocked - network access disabled")
        return True

def check_fixture_files():
    """Verify required fixture files exist"""
    required_fixtures = [
        "tests/fixtures/rss/google_news_sample.xml",
        "tests/fixtures/rss/general_tech_news.xml",
        "tests/fixtures/rss/tech_news_sample.xml",
        "tests/fixtures/rss/financial_news_sample.xml"
    ]
    
    all_exist = True
    for fixture_path in required_fixtures:
        if Path(fixture_path).exists():
            print(f"✅ {fixture_path}")
        else:
            print(f"⚠️  Missing (optional): {fixture_path}")
            # Don't fail for optional fixtures
    
    # Check that at least one RSS fixture exists
    rss_fixtures = list(Path("tests/fixtures/rss").glob("*.xml"))
    if not rss_fixtures:
        print("❌ No RSS fixtures found in tests/fixtures/rss/")
        all_exist = False
    else:
        print(f"✅ Found {len(rss_fixtures)} RSS fixture(s)")
    
    return all_exist

def check_imports():
    """Test critical imports don't fail"""
    try:
        # Test structlog shim
        sys.path.append('.')
        from fakes.structlog_shim import get_logger
        logger = get_logger("test")
        logger.info("Import test successful")
        
        # Test core modules
        from config.feature_flags import FeatureFlags
        from services.retry_policies import CircuitBreaker
        from fakes.alpha_vantage import FakeAlphaVantageManager
        
        print("✅ All critical imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Run all preflight checks"""
    print("🚀 DRY_RUN PREFLIGHT CHECKS")
    print("=" * 30)
    
    checks = [
        check_dry_run_env,
        check_socket_blocked,
        check_fixture_files,
        check_imports
    ]
    
    results = [check() for check in checks]
    
    print("\n" + "=" * 30)
    if all(results):
        print("✅ ALL CHECKS PASSED - Ready for dry run")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before running")
        return 1

if __name__ == "__main__":
    sys.exit(main())