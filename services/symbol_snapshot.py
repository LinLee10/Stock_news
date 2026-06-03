#!/usr/bin/env python3
"""
Symbol snapshot service for generating comprehensive symbol overviews
Provides price cards, chart data, headlines, and earnings information
"""
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import existing services
from prediction import fetch_price_history
from news_scraper import scrape_headlines
from config.feature_flags import is_news_corroboration_enabled

logger = logging.getLogger(__name__)

class SymbolSnapshotService:
    """Service for generating comprehensive symbol snapshots"""
    
    def __init__(self):
        self.data_path = Path("data")
    
    def generate_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate comprehensive snapshot for a symbol"""
        try:
            symbol = symbol.upper()
            
            # Get price data and chart
            price_card = self._get_price_card(symbol)
            if not price_card:
                logger.warning(f"No price data available for {symbol}")
                return None
            
            # Get chart data (last 10 trading days)
            chart_data = self._get_chart_data(symbol)
            
            # Get recent headlines with sentiment
            headlines = self._get_headlines_with_sentiment(symbol)
            
            # Get earnings information
            earnings_info = self._get_earnings_info(symbol)
            
            # Compile snapshot
            snapshot = {
                'symbol': symbol,
                'generated_at': datetime.utcnow().isoformat(),
                'price_card': price_card,
                'chart_data': chart_data,
                'headlines': headlines,
                'earnings': earnings_info,
                'data_sources': {
                    'price_data': 'alpha_vantage_cached' if price_card else 'unavailable',
                    'news_data': 'google_news_alpha_vantage',
                    'earnings_data': 'manual_calendar' if earnings_info else 'unavailable'
                }
            }
            
            return snapshot
            
        except Exception as e:
            logger.exception(f"Error generating snapshot for {symbol}")
            return None
    
    def _get_price_card(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price information and key metrics"""
        try:
            # Fetch recent price history
            price_df = fetch_price_history(symbol, period="30d")
            if price_df.empty:
                return None
            
            # Calculate key metrics
            latest_price = float(price_df['Close'].iloc[-1])
            
            # Calculate price changes
            price_changes = {}
            for days, label in [(1, '1d'), (5, '5d'), (30, '30d')]:
                if len(price_df) > days:
                    old_price = float(price_df['Close'].iloc[-(days+1)])
                    change = latest_price - old_price
                    change_pct = (change / old_price) * 100
                    price_changes[label] = {
                        'change': round(change, 2),
                        'change_pct': round(change_pct, 2)
                    }
            
            # Calculate volatility (30-day)
            returns = price_df['Close'].pct_change().dropna()
            volatility = float(returns.std() * (252 ** 0.5)) if len(returns) > 1 else 0.0
            
            # Price ranges
            high_52w = float(price_df['Close'].max()) if len(price_df) >= 252 else float(price_df['Close'].max())
            low_52w = float(price_df['Close'].min()) if len(price_df) >= 252 else float(price_df['Close'].min())
            
            price_card = {
                'current_price': latest_price,
                'currency': 'USD',
                'price_changes': price_changes,
                'metrics': {
                    'volatility_annualized': round(volatility, 4),
                    'high_52w': high_52w,
                    'low_52w': low_52w,
                    'volume_avg': None  # Would need volume data
                },
                'last_updated': price_df['Date'].iloc[-1].isoformat() if not price_df.empty else None
            }
            
            return price_card
            
        except Exception as e:
            logger.error(f"Error getting price card for {symbol}: {e}")
            return None
    
    def _get_chart_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get 10-day chart data for mini chart"""
        try:
            price_df = fetch_price_history(symbol, period="30d")
            if price_df.empty:
                return []
            
            # Get last 10 trading days
            chart_df = price_df.tail(10).copy()
            
            chart_data = []
            for _, row in chart_df.iterrows():
                chart_data.append({
                    'date': row['Date'].strftime('%Y-%m-%d'),
                    'close': float(row['Close']),
                    'timestamp': int(row['Date'].timestamp())
                })
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error getting chart data for {symbol}: {e}")
            return []
    
    def _get_headlines_with_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        """Get top 5 recent headlines with sentiment scores and confidence"""
        try:
            # Scrape recent headlines (7 days)
            headlines_data = scrape_headlines([symbol], days=7)
            symbol_data = headlines_data.get(symbol, {})
            
            headlines = symbol_data.get('headlines', [])
            if not headlines:
                return []
            
            # Sort by date (most recent first) and take top 5
            headlines_sorted = sorted(headlines, key=lambda x: x[2], reverse=True)[:5]
            
            result = []
            for title, url, date in headlines_sorted:
                # Basic sentiment analysis (would be enhanced with clustering if enabled)
                sentiment_score = self._calculate_basic_sentiment(title)
                confidence = 0.7  # Basic confidence
                
                # Enhanced scoring if news corroboration is enabled
                if is_news_corroboration_enabled():
                    # Future: Enhanced sentiment with clustering and consensus
                    confidence = 0.85  # Higher confidence with corroboration
                
                headline_data = {
                    'title': title,
                    'url': url,
                    'published_date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    'sentiment': {
                        'score': sentiment_score,
                        'confidence': confidence,
                        'label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
                    },
                    'source': self._extract_source_from_url(url)
                }
                result.append(headline_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting headlines for {symbol}: {e}")
            return []
    
    def _calculate_basic_sentiment(self, title: str) -> float:
        """Basic sentiment calculation (placeholder for more sophisticated analysis)"""
        positive_words = ['gain', 'rise', 'up', 'bull', 'buy', 'strong', 'growth', 'profit']
        negative_words = ['loss', 'fall', 'down', 'bear', 'sell', 'weak', 'decline', 'risk']
        
        title_lower = title.lower()
        pos_count = sum(1 for word in positive_words if word in title_lower)
        neg_count = sum(1 for word in negative_words if word in title_lower)
        
        if pos_count > neg_count:
            return 0.3
        elif neg_count > pos_count:
            return -0.3
        else:
            return 0.0
    
    def _extract_source_from_url(self, url: str) -> str:
        """Extract source name from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            # Remove 'www.' and take domain name
            domain = domain.replace('www.', '')
            return domain.split('.')[0] if domain else 'unknown'
        except:
            return 'unknown'
    
    def _get_earnings_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get earnings calendar information"""
        try:
            # Check if earnings calendar exists
            earnings_calendar_path = self.data_path / "earnings_calendar.csv"
            if not earnings_calendar_path.exists():
                return None
            
            # Load earnings calendar
            calendar_df = pd.read_csv(earnings_calendar_path)
            symbol_earnings = calendar_df[calendar_df['ticker'].str.upper() == symbol.upper()]
            
            if symbol_earnings.empty:
                return None
            
            # Get next earnings date
            today = pd.Timestamp.now().date()
            future_earnings = symbol_earnings[pd.to_datetime(symbol_earnings['earnings_date']).dt.date >= today]
            
            if future_earnings.empty:
                return None
            
            next_earnings = future_earnings.iloc[0]
            
            # Check for implied move data
            implied_move = self._get_implied_move(symbol)
            
            earnings_info = {
                'next_earnings_date': next_earnings['earnings_date'],
                'confirmed': bool(next_earnings.get('confirmed', False)),
                'time_of_day': next_earnings.get('time_of_day', 'unknown'),
                'fiscal_quarter': next_earnings.get('fiscal_quarter', 'unknown'),
                'implied_move_pct': implied_move,
                'days_until': (pd.to_datetime(next_earnings['earnings_date']).date() - today).days
            }
            
            return earnings_info
            
        except Exception as e:
            logger.error(f"Error getting earnings info for {symbol}: {e}")
            return None
    
    def _get_implied_move(self, symbol: str) -> Optional[float]:
        """Get options implied move percentage if available"""
        try:
            # Check earnings stats table
            earnings_stats_path = self.data_path / "earnings_stats.csv"
            if not earnings_stats_path.exists():
                return None
            
            stats_df = pd.read_csv(earnings_stats_path)
            symbol_stats = stats_df[stats_df['ticker'].str.upper() == symbol.upper()]
            
            if symbol_stats.empty:
                return None
            
            implied_move = symbol_stats.iloc[0].get('implied_move_pct')
            return float(implied_move) if pd.notna(implied_move) else None
            
        except Exception as e:
            logger.error(f"Error getting implied move for {symbol}: {e}")
            return None