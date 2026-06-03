#!/usr/bin/env python3
"""
Comprehensive test runner ensuring all tests complete in under 30 seconds.
Provides detailed timing and coverage information.
"""

import sys
import os
import unittest
import time
from pathlib import Path
from io import StringIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment
os.environ['TESTING'] = '1'
os.environ['YF_TEST_FAST_BACKOFF'] = '1'
os.environ['ENABLE_DEBUG_MODE'] = 'false'


class TimedTestResult(unittest.TextTestResult):
    """Test result class that tracks timing for each test."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_times = {}
        self.start_time = None
        
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
        
    def stopTest(self, test):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.test_times[str(test)] = elapsed
        super().stopTest(test)


class TimedTestRunner(unittest.TextTestRunner):
    """Test runner that provides timing information."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resultclass = TimedTestResult
        
    def run(self, test):
        result = super().run(test)
        
        # Print timing summary
        if hasattr(result, 'test_times') and result.test_times:
            print("\n" + "="*60)
            print("TEST TIMING SUMMARY")
            print("="*60)
            
            total_time = sum(result.test_times.values())
            print(f"Total test time: {total_time:.3f}s")
            
            # Show slowest tests
            sorted_times = sorted(result.test_times.items(), 
                                key=lambda x: x[1], reverse=True)
            
            print("\nSlowest tests:")
            for test_name, test_time in sorted_times[:10]:
                print(f"  {test_time:.3f}s - {test_name}")
                
            # Check performance requirement
            if total_time > 30.0:
                print(f"\n⚠️  WARNING: Test suite took {total_time:.1f}s (target: <30s)")
                print("Consider optimizing slow tests or adding more mocking")
            else:
                print(f"\n✅ Performance target met: {total_time:.1f}s < 30s")
                
        return result


def discover_tests():
    """Discover all tests in the test directory."""
    
    test_dir = Path(__file__).parent
    
    # Test categories with expected approximate counts
    test_categories = {
        'unit': 'Unit tests (fast, isolated)',
        'integration': 'Integration tests (mocked external calls)',
        'smoke': 'Smoke tests (basic functionality)'
    }
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    print("Discovering tests...")
    
    for category, description in test_categories.items():
        category_dir = test_dir / category
        if category_dir.exists():
            category_tests = loader.discover(str(category_dir), pattern='test_*.py')
            suite.addTest(category_tests)
            
            # Count tests in category
            test_count = category_tests.countTestCases()
            print(f"  {category}: {test_count} tests - {description}")
    
    # Also discover tests in root test directory
    root_tests = loader.discover(str(test_dir), pattern='test_*.py', top_level_dir=str(test_dir.parent))
    suite.addTest(root_tests)
    
    total_count = suite.countTestCases()
    print(f"\nTotal tests discovered: {total_count}")
    
    return suite


def run_performance_check():
    """Run a quick performance check before full test suite."""
    
    print("Running performance check...")
    start_time = time.time()
    
    # Import key modules to check for obvious performance issues
    try:
        import config.feature_flags
        import services.retry_policies
        from news_scraper import scrape_headlines
        from prediction import train_predict_stock
    except ImportError as e:
        print(f"⚠️  Import performance issue: {e}")
        return False
        
    elapsed = time.time() - start_time
    if elapsed > 5.0:
        print(f"⚠️  Slow imports detected: {elapsed:.1f}s")
        return False
    else:
        print(f"✅ Import performance: {elapsed:.3f}s")
        return True


def run_feature_flag_baseline_check():
    """Verify all feature flags are OFF by default."""
    
    print("\nVerifying feature flag defaults...")
    
    from config.feature_flags import FeatureFlags
    
    flags = FeatureFlags()
    all_flags = flags.get_all_flags()
    
    enabled_flags = [name for name, enabled in all_flags.items() if enabled]
    
    if enabled_flags:
        print(f"❌ ERROR: The following flags are enabled by default: {enabled_flags}")
        print("All flags must default to OFF to prevent regressions!")
        return False
    else:
        print(f"✅ All {len(all_flags)} feature flags default to OFF")
        return True


def run_critical_integration_check():
    """Run critical integration checks before full suite."""
    
    print("\nRunning critical integration checks...")
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Main pipeline can be imported
    total_checks += 1
    try:
        import main
        print("✅ Main pipeline imports successfully")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Main pipeline import failed: {e}")
    
    # Check 2: Feature flags system works
    total_checks += 1
    try:
        from config.feature_flags import feature_flags
        test_flag = 'enable_finbert_pipeline'
        original_state = feature_flags.is_enabled(test_flag)
        feature_flags.set_flag(test_flag, not original_state)
        new_state = feature_flags.is_enabled(test_flag)
        feature_flags.set_flag(test_flag, original_state)  # Reset
        
        if new_state != original_state:
            print("✅ Feature flag system works correctly")
            checks_passed += 1
        else:
            print("❌ Feature flag system not working")
    except Exception as e:
        print(f"❌ Feature flag test failed: {e}")
    
    # Check 3: Retry policies can be imported
    total_checks += 1
    try:
        from services.retry_policies import CircuitBreaker, retry_with_backoff
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        if cb.state == "CLOSED":
            print("✅ Circuit breaker system works")
            checks_passed += 1
        else:
            print("❌ Circuit breaker in wrong initial state")
    except Exception as e:
        print(f"❌ Circuit breaker test failed: {e}")
    
    success_rate = (checks_passed / total_checks) * 100
    print(f"\nCritical checks: {checks_passed}/{total_checks} passed ({success_rate:.0f}%)")
    
    return checks_passed == total_checks


def main():
    """Main test runner function."""
    
    print("="*60)
    print("COMPREHENSIVE TEST RUNNER")
    print("="*60)
    
    # Pre-flight checks
    if not run_performance_check():
        print("❌ Performance check failed - aborting test run")
        return 1
        
    if not run_feature_flag_baseline_check():
        print("❌ Feature flag baseline check failed - aborting test run")
        return 1
        
    if not run_critical_integration_check():
        print("❌ Critical integration check failed - aborting test run")
        return 1
    
    print("\n" + "="*60)
    print("RUNNING FULL TEST SUITE")
    print("="*60)
    
    # Discover and run all tests
    suite = discover_tests()
    
    if suite.countTestCases() == 0:
        print("❌ No tests found!")
        return 1
    
    # Run with timing
    runner = TimedTestRunner(verbosity=2, buffer=True)
    
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Total time: {total_time:.3f}s")
    
    # Performance assessment
    if total_time <= 30.0:
        print("✅ PERFORMANCE TARGET MET: Test suite completed in under 30 seconds")
    else:
        print("⚠️  PERFORMANCE TARGET MISSED: Test suite took longer than 30 seconds")
        print("Consider adding more mocking or optimizing slow tests")
    
    # Overall assessment
    if result.wasSuccessful():
        if total_time <= 30.0:
            print("\n🎉 ALL TESTS PASSED AND PERFORMANCE TARGET MET!")
            return 0
        else:
            print("\n✅ All tests passed but performance needs improvement")
            return 0
    else:
        print(f"\n❌ TEST FAILURES DETECTED")
        
        if result.failures:
            print(f"\nFailures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")
                
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())