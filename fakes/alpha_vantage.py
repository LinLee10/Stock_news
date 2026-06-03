#!/usr/bin/env python3
"""Fake Alpha Vantage manager for DRY_RUN mode"""
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class FakeAPIResponse:
    success: bool
    data: Optional[Dict[str, Any]]
    error_message: Optional[str] = None
    response_time_ms: int = 50

class FakeAlphaVantageManager:
    """Fake Alpha Vantage manager returning deterministic data"""
    
    def __init__(self, api_key: str = "FAKE_KEY"):
        self.api_key = api_key
        self.call_count = 0
    
    async def start(self):
        """No-op start"""
        pass
    
    async def stop(self):
        """No-op stop"""
        pass
    
    async def get_daily_data(self, ticker: str, **kwargs) -> FakeAPIResponse:
        """Return fake daily OHLCV data"""
        self.call_count += 1
        
        # Generate 30 days of fake OHLCV data
        dates = pd.date_range(
            end=datetime.now(timezone.utc),
            periods=30,
            freq='D'
        )
        
        # Deterministic prices based on ticker
        base_price = hash(ticker) % 100 + 50  # 50-150 range
        
        data = {
            "Time Series (Daily)": {
                date.strftime("%Y-%m-%d"): {
                    "1. open": f"{base_price + (i % 5)}",
                    "2. high": f"{base_price + (i % 5) + 2}",
                    "3. low": f"{base_price + (i % 5) - 1}",
                    "4. close": f"{base_price + (i % 7)}",
                    "5. volume": f"{1000000 + (i * 50000)}"
                }
                for i, date in enumerate(dates)
            }
        }
        
        return FakeAPIResponse(success=True, data=data)
    
    async def get_quote(self, ticker: str, **kwargs) -> FakeAPIResponse:
        """Return fake quote data"""
        self.call_count += 1
        base_price = hash(ticker) % 100 + 50
        
        data = {
            "Global Quote": {
                "01. symbol": ticker,
                "02. open": f"{base_price}",
                "03. high": f"{base_price + 2}",
                "04. low": f"{base_price - 1}",
                "05. price": f"{base_price + 1}",
                "06. volume": "1500000",
                "07. latest trading day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "08. previous close": f"{base_price}",
                "09. change": "1.00",
                "10. change percent": "1.00%"
            }
        }
        
        return FakeAPIResponse(success=True, data=data)