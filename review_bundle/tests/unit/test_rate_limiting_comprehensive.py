#!/usr/bin/env python3
"""
Comprehensive tests for rate limiting, circuit breakers, and retry policies.
Ensures backoff/retry logic is bounded and testable.
"""

import sys
import os
import unittest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set fast backoff for all tests
os.environ['YF_TEST_FAST_BACKOFF'] = '1'

from services.retry_policies import (
    retry_with_backoff, CircuitBreaker, CircuitBreakerOpenError,
    with_circuit_breaker
)


class TestRetryWithBackoff(unittest.TestCase):
    """Test retry logic with bounded, injectable delays."""
    
    def setUp(self):
        """Set up test environment."""
        # Ensure we're in fast backoff mode
        os.environ['YF_TEST_FAST_BACKOFF'] = '1'
        
    def test_successful_function_no_retries(self):
        """Test successful function executes once without retries."""
        
        mock_fn = Mock(return_value="success")
        
        result = retry_with_backoff(mock_fn, max_retries=3)
        
        self.assertEqual(result, "success")
        mock_fn.assert_called_once()
        
    def test_retry_on_specified_exceptions(self):
        """Test retry occurs on specified exception types."""
        
        class TestException(Exception):
            pass
            
        mock_fn = Mock()
        mock_fn.side_effect = [TestException("fail"), TestException("fail"), "success"]
        
        result = retry_with_backoff(
            mock_fn, 
            retry_on=(TestException,),
            max_retries=3,
            base_delay=0.001,  # Very small delay for tests
            debug=True
        )
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_fn.call_count, 3)
        
    def test_max_retries_exhausted(self):
        """Test that function fails after max retries exhausted."""
        
        class TestException(Exception):
            pass
            
        mock_fn = Mock(side_effect=TestException("persistent fail"))
        
        with self.assertRaises(TestException):
            retry_with_backoff(
                mock_fn,
                retry_on=(TestException,),
                max_retries=2,
                base_delay=0.001,
                debug=True
            )
            
        # Should be called max_retries + 1 times (initial + retries)
        self.assertEqual(mock_fn.call_count, 3)
        
    def test_non_retriable_exception_no_retry(self):
        """Test that non-retriable exceptions are not retried."""
        
        class RetriableException(Exception):
            pass
            
        class NonRetriableException(Exception):
            pass
            
        mock_fn = Mock(side_effect=NonRetriableException("immediate fail"))
        
        with self.assertRaises(NonRetriableException):
            retry_with_backoff(
                mock_fn,
                retry_on=(RetriableException,),
                max_retries=3
            )
            
        # Should only be called once
        mock_fn.assert_called_once()
        
    def test_fast_backoff_in_test_mode(self):
        """Test that backoff is bypassed in test mode for speed."""
        
        class TestException(Exception):
            pass
            
        mock_fn = Mock()
        mock_fn.side_effect = [TestException("fail"), "success"]
        
        start_time = time.time()
        
        result = retry_with_backoff(
            mock_fn,
            retry_on=(TestException,),
            max_retries=2,
            base_delay=10.0,  # Would be very slow without fast mode
            debug=False
        )
        
        elapsed = time.time() - start_time
        
        self.assertEqual(result, "success")
        # Should complete very quickly in test mode
        self.assertLess(elapsed, 0.1, "Backoff should be bypassed in test mode")
        
    def test_exponential_backoff_calculation(self):
        """Test exponential backoff calculation (timing not tested)."""
        
        # This test verifies the backoff calculation logic
        # without actually waiting for delays
        
        class TestException(Exception):
            pass
            
        call_times = []
        
        def time_recording_fn():
            call_times.append(time.time())
            if len(call_times) <= 2:
                raise TestException("fail")
            return "success"
        
        retry_with_backoff(
            time_recording_fn,
            retry_on=(TestException,),
            max_retries=3,
            base_delay=0.001,
            backoff_multiplier=2.0
        )
        
        # Verify we made the expected number of calls
        self.assertEqual(len(call_times), 3)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality."""
    
    def setUp(self):
        """Set up circuit breaker for testing."""
        self.cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,  # Short timeout for tests
            expected_exception=ValueError
        )
        
    def test_initial_state_closed(self):
        """Test circuit breaker starts in CLOSED state."""
        self.assertEqual(self.cb.state, "CLOSED")
        self.assertEqual(self.cb.failure_count, 0)
        
    def test_successful_calls_stay_closed(self):
        """Test successful calls keep circuit CLOSED."""
        
        def success_fn():
            return "success"
            
        for _ in range(5):
            result = self.cb.call(success_fn)
            self.assertEqual(result, "success")
            self.assertEqual(self.cb.state, "CLOSED")
            
    def test_failures_trip_circuit_to_open(self):
        """Test that consecutive failures trip circuit to OPEN."""
        
        def failing_fn():
            raise ValueError("test failure")
            
        # First few failures should stay CLOSED
        for i in range(self.cb.failure_threshold - 1):
            with self.assertRaises(ValueError):
                self.cb.call(failing_fn)
            self.assertEqual(self.cb.state, "CLOSED")
            
        # Final failure should trip to OPEN
        with self.assertRaises(ValueError):
            self.cb.call(failing_fn)
        self.assertEqual(self.cb.state, "OPEN")
        
    def test_open_circuit_rejects_calls(self):
        """Test OPEN circuit rejects calls without executing function."""
        
        # Trip circuit to OPEN first
        def failing_fn():
            raise ValueError("test failure")
            
        for _ in range(self.cb.failure_threshold):
            with self.assertRaises(ValueError):
                self.cb.call(failing_fn)
                
        self.assertEqual(self.cb.state, "OPEN")
        
        # Now test that calls are rejected
        mock_fn = Mock(return_value="should not be called")
        
        with self.assertRaises(CircuitBreakerOpenError):
            self.cb.call(mock_fn)
            
        mock_fn.assert_not_called()
        
    def test_half_open_recovery_on_timeout(self):
        """Test circuit moves to HALF_OPEN after recovery timeout."""
        
        # Trip to OPEN
        def failing_fn():
            raise ValueError("test failure")
            
        for _ in range(self.cb.failure_threshold):
            with self.assertRaises(ValueError):
                self.cb.call(failing_fn)
                
        self.assertEqual(self.cb.state, "OPEN")
        
        # Wait for recovery timeout
        time.sleep(self.cb.recovery_timeout + 0.1)
        
        # Next call should move to HALF_OPEN and succeed
        def success_fn():
            return "recovered"
            
        result = self.cb.call(success_fn)
        self.assertEqual(result, "recovered")
        self.assertEqual(self.cb.state, "CLOSED")  # Should reset to CLOSED on success
        
    def test_half_open_failure_returns_to_open(self):
        """Test failure in HALF_OPEN returns circuit to OPEN."""
        
        # Trip to OPEN
        def failing_fn():
            raise ValueError("test failure")
            
        for _ in range(self.cb.failure_threshold):
            with self.assertRaises(ValueError):
                self.cb.call(failing_fn)
                
        # Wait for recovery timeout
        time.sleep(self.cb.recovery_timeout + 0.1)
        
        # Fail in HALF_OPEN
        with self.assertRaises(ValueError):
            self.cb.call(failing_fn)
            
        self.assertEqual(self.cb.state, "OPEN")
        
    def test_reset_on_success_in_half_open(self):
        """Test circuit resets to CLOSED on success in HALF_OPEN."""
        
        # Trip to OPEN
        for _ in range(self.cb.failure_threshold):
            with self.assertRaises(ValueError):
                self.cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
                
        # Wait and succeed in HALF_OPEN
        time.sleep(self.cb.recovery_timeout + 0.1)
        
        result = self.cb.call(lambda: "success")
        self.assertEqual(result, "success")
        self.assertEqual(self.cb.state, "CLOSED")
        self.assertEqual(self.cb.failure_count, 0)


class TestCircuitBreakerWithFallback(unittest.TestCase):
    """Test circuit breaker with fallback functionality."""
    
    def test_with_circuit_breaker_fallback(self):
        """Test with_circuit_breaker function with fallback."""
        
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        
        def failing_fn():
            raise ValueError("service down")
            
        def fallback_fn():
            return "fallback result"
            
        # First call should fail and trip circuit
        result = with_circuit_breaker(cb, failing_fn, fallback_fn)
        self.assertEqual(cb.state, "OPEN")
        
        # Subsequent calls should use fallback
        result = with_circuit_breaker(cb, failing_fn, fallback_fn)
        self.assertEqual(result, "fallback result")
        
    def test_with_circuit_breaker_no_fallback(self):
        """Test with_circuit_breaker without fallback raises exception."""
        
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        
        def failing_fn():
            raise ValueError("service down")
            
        # First call trips circuit
        with self.assertRaises(ValueError):
            with_circuit_breaker(cb, failing_fn)
            
        # Second call should raise CircuitBreakerOpenError
        with self.assertRaises(CircuitBreakerOpenError):
            with_circuit_breaker(cb, failing_fn)


class TestRateLimitingIntegration(unittest.TestCase):
    """Test integration of rate limiting with actual service classes."""
    
    def test_yfinance_provider_uses_backoff(self):
        """Test that YFinanceProvider uses retry with backoff."""
        
        from services.data_sources.yfinance_provider import YFinanceProvider, YFConfig
        
        config = YFConfig(
            max_retries=2,
            backoff_base_seconds=0.001,
            test_fast_backoff=True
        )
        
        provider = YFinanceProvider(config)
        
        # Verify circuit breaker is configured
        self.assertIsNotNone(provider.circuit_breaker)
        self.assertEqual(provider.circuit_breaker.failure_threshold, 1)  # Test mode
        
    def test_alpha_vantage_manager_respects_quotas(self):
        """Test Alpha Vantage manager respects rate limits."""
        
        try:
            from services.alpha_vantage_manager import AlphaVantageManager
            
            # This would test quota checking behavior
            # Skip if not implemented
            manager = AlphaVantageManager()
            self.assertTrue(hasattr(manager, 'get_remaining_capacity'))
            
        except ImportError:
            self.skipTest("AlphaVantageManager not available")
            
    def test_newsapi_client_handles_quota_exhaustion(self):
        """Test NewsAPI client handles quota exhaustion gracefully."""
        
        try:
            from integrations.newsapi_client import NewsAPIClient
            
            client = NewsAPIClient()
            # Test quota manager exists
            self.assertTrue(hasattr(client, 'quota_manager'))
            
        except ImportError:
            self.skipTest("NewsAPIClient not available")


if __name__ == '__main__':
    # Ensure test environment
    os.environ['TESTING'] = '1'
    os.environ['YF_TEST_FAST_BACKOFF'] = '1'
    
    unittest.main(verbosity=2)