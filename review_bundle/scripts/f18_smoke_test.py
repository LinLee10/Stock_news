#!/usr/bin/env python3
"""
F18 Microservices Smoke Test
Tests /healthz and /news endpoints for API Gateway and News Scraping Service
"""

import requests
import json
import sys
from datetime import datetime

def test_endpoint(url, description):
    """Test a single endpoint"""
    try:
        print(f"Testing {description}: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Success: {response.status_code}")
            print(f"  ✓ Response: {json.dumps(data, indent=2)[:200]}...")
            return True
        else:
            print(f"  ✗ Failed: {response.status_code}")
            print(f"  ✗ Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Connection Error: {e}")
        return False
    except json.JSONDecodeError:
        print(f"  ✗ Invalid JSON response")
        return False

def main():
    """Run F18 microservices smoke test"""
    print("F18 Microservices Smoke Test")
    print("=" * 40)
    print(f"Started at: {datetime.utcnow().isoformat()}Z")
    print()
    
    # Test endpoints
    tests = [
        ("http://localhost:8000/healthz", "API Gateway Health"),
        ("http://localhost:8001/healthz", "News Service Health"),
        ("http://localhost:8000/news?limit=3", "Gateway News Proxy"),
        ("http://localhost:8001/news?limit=3", "Direct News Service")
    ]
    
    results = []
    for url, description in tests:
        success = test_endpoint(url, description)
        results.append(success)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 40)
    print(f"SMOKE TEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All F18 microservices are healthy!")
        sys.exit(0)
    else:
        print("✗ Some F18 microservices failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()