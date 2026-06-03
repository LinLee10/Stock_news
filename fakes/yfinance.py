#!/usr/bin/env python3
"""Fake yfinance module for DRY_RUN mode"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Union


def fake_download(tickers: Union[str, List[str]], 
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 period: str = "1mo",
                 interval: str = "1d",
                 **kwargs) -> pd.DataFrame:
    """Fake yfinance.download() returning deterministic data"""
    
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Generate date range
    if start and end:
        dates = pd.date_range(start=start, end=end, freq='D')
    else:
        # Default to 30 days
        dates = pd.date_range(
            end=datetime.now(),
            periods=30,
            freq='D'
        )
    
    # Filter to business days only
    dates = dates[dates.weekday < 5]
    
    if len(tickers) == 1:
        # Single ticker - return simple DataFrame
        ticker = tickers[0]
        base_price = hash(ticker) % 100 + 50
        
        data = {
            'Open': [base_price + (i % 5) for i in range(len(dates))],
            'High': [base_price + (i % 5) + 2 for i in range(len(dates))],
            'Low': [base_price + (i % 5) - 1 for i in range(len(dates))],
            'Close': [base_price + (i % 7) for i in range(len(dates))],
            'Volume': [1000000 + (i * 50000) for i in range(len(dates))],
            'Adj Close': [base_price + (i % 7) for i in range(len(dates))]
        }
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        return df
    
    else:
        # Multiple tickers - return MultiIndex DataFrame
        data = {}
        
        for ticker in tickers:
            base_price = hash(ticker) % 100 + 50
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                data[(col, ticker)] = [
                    base_price + (i % 5) + ({'High': 2, 'Low': -1}.get(col, 0))
                    for i in range(len(dates))
                ]
        
        # Create MultiIndex columns
        columns = pd.MultiIndex.from_tuples(
            [(col, ticker) for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] 
             for ticker in tickers],
            names=['Price', 'Ticker']
        )
        
        df = pd.DataFrame(data, index=dates, columns=columns)
        df.index.name = 'Date'
        return df


class FakeTicker:
    """Fake yfinance.Ticker class"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self._base_price = hash(ticker) % 100 + 50
    
    def history(self, 
               period: str = "1mo",
               interval: str = "1d",
               start: Optional[str] = None,
               end: Optional[str] = None,
               **kwargs) -> pd.DataFrame:
        """Fake ticker.history() method"""
        
        if start and end:
            dates = pd.date_range(start=start, end=end, freq='D')
        elif period == "1d":
            dates = pd.date_range(
                end=datetime.now(),
                periods=1,
                freq='D'
            )
        elif period == "5d":
            dates = pd.date_range(
                end=datetime.now(),
                periods=5,
                freq='D'
            )
        elif period == "1mo":
            dates = pd.date_range(
                end=datetime.now(),
                periods=30,
                freq='D'
            )
        elif period == "1y":
            dates = pd.date_range(
                end=datetime.now(),
                periods=252,  # Trading days in a year
                freq='D'
            )
        else:
            dates = pd.date_range(
                end=datetime.now(),
                periods=30,
                freq='D'
            )
        
        # Filter to business days
        dates = dates[dates.weekday < 5]
        
        data = {
            'Open': [self._base_price + (i % 5) for i in range(len(dates))],
            'High': [self._base_price + (i % 5) + 2 for i in range(len(dates))],
            'Low': [self._base_price + (i % 5) - 1 for i in range(len(dates))],
            'Close': [self._base_price + (i % 7) for i in range(len(dates))],
            'Volume': [1000000 + (i * 50000) for i in range(len(dates))],
        }
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        return df
    
    @property
    def info(self) -> dict:
        """Fake ticker info"""
        return {
            'symbol': self.ticker,
            'longName': f'{self.ticker} Inc.',
            'sector': 'Technology',
            'industry': 'Software',
            'marketCap': 1000000000,
            'regularMarketPrice': self._base_price,
            'currency': 'USD'
        }


# For backwards compatibility
download = fake_download
Ticker = FakeTicker