#!/usr/bin/env python3
"""
Test script to diagnose Alpha Vantage API timeout issues
"""
import time
import requests
from datetime import datetime
from config.config import ALPHA_VANTAGE_KEY

def test_alpha_vantage_api():
    """Test Alpha Vantage API with explicit timeout and diagnostics"""
    
    print(f"🔍 Testing Alpha Vantage API - {datetime.now()}")
    print(f"API Key: {ALPHA_VANTAGE_KEY[:8]}..." if ALPHA_VANTAGE_KEY else "No API key found!")
    
    # Test URL that our system is likely calling
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'AAPL',  # Simple test symbol
        'apikey': ALPHA_VANTAGE_KEY,
        'outputsize': 'compact',
        'datatype': 'json'
    }
    
    print(f"🌐 Testing URL: {url}")
    print(f"📦 Params: {params}")
    
    try:
        print("⏰ Starting request with 30s timeout...")
        start_time = time.time()
        
        response = requests.get(
            url, 
            params=params, 
            timeout=30,  # Same as config
            headers={'User-Agent': 'StonkNews/1.0'}
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Response received in {elapsed:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        print(f"📏 Response Length: {len(response.text)} chars")
        
        # Check if response contains actual data or error
        if response.status_code == 200:
            json_data = response.json()
            if "Error Message" in json_data:
                print(f"❌ API Error: {json_data['Error Message']}")
            elif "Information" in json_data:
                print(f"ℹ️  API Info: {json_data['Information']}")  
            elif "Time Series (Daily)" in json_data:
                print(f"✅ Success: Got {len(json_data['Time Series (Daily)'])} data points")
            else:
                print(f"⚠️  Unexpected response format: {list(json_data.keys())}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("⏰ TIMEOUT: Request timed out after 30 seconds")
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 CONNECTION ERROR: {e}")
    except requests.exceptions.RequestException as e:
        print(f"🚨 REQUEST ERROR: {e}")
    except Exception as e:
        print(f"💥 UNEXPECTED ERROR: {e}")

def test_smart_cache_manager():
    """Test the SmartCacheManager that might be causing issues"""
    print(f"\n🗃️  Testing SmartCacheManager integration...")
    
    try:
        from services.smart_cache_manager import SmartCacheManager
        from services.data_sources.price_provider import PriceProvider
        
        print("📦 Initializing SmartCacheManager...")
        cache_manager = SmartCacheManager()
        
        print("📦 Initializing PriceProvider...")  
        provider = PriceProvider()
        
        print("✅ Initialization successful")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 API Timeout Diagnostic Test\n")
    test_alpha_vantage_api()
    test_smart_cache_manager()
    print(f"\n✅ Test completed - {datetime.now()}")