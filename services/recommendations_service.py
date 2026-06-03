#!/usr/bin/env python3
"""
Recommendations service for integrating reco_engine with portfolio/watchlist data
Provides financial context and position management
"""
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from config.tickers import PORTFOLIO, WATCHLIST
from services.reco_engine import generate_symbol_recommendation
from prediction import fetch_price_history
from news_scraper import scrape_headlines

logger = logging.getLogger(__name__)

class RecommendationsService:
    """Service for generating contextualized recommendations"""
    
    def __init__(self):
        self.data_path = Path("data")
        self.portfolio_path = self.data_path / "portfolio.csv"
        self.watchlist_path = self.data_path / "watchlist.csv"
    
    def generate_watchlist_recommendations(self, 
                                         include_details: bool = False,
                                         max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """Generate recommendations for watchlist symbols"""
        try:
            recommendations = []
            
            for symbol in WATCHLIST:
                try:
                    # Gather data for recommendation engine
                    technical_data = self._get_technical_data(symbol)
                    momentum_data = self._get_momentum_data(symbol)
                    news_data = self._get_news_data(symbol)
                    earnings_data = self._get_earnings_data(symbol)
                    
                    # For watchlist, create mock portfolio context (no position)
                    portfolio_data = {
                        'quantity': 0.0,
                        'cost_basis': 0.0,
                        'current_value': 0.0,
                        'position_pct': 0.0,
                        'unrealized_pnl_pct': 0.0
                    }
                    
                    # Generate recommendation
                    rec = generate_symbol_recommendation(
                        symbol, technical_data, momentum_data, news_data, earnings_data, portfolio_data
                    )
                    
                    # Add watchlist-specific context
                    rec['context'] = 'watchlist'
                    rec['current_price'] = technical_data['price_current']
                    rec['position_status'] = 'not_held'
                    
                    if include_details:
                        rec['details'] = {
                            'technical_signals': self._summarize_technical_signals(technical_data),
                            'momentum_signals': self._summarize_momentum_signals(momentum_data),
                            'news_sentiment': news_data['consensus_14d'],
                            'earnings_proximity': earnings_data.get('hours_until_earnings', 'N/A')
                        }
                    
                    recommendations.append(rec)
                    
                except Exception as e:
                    logger.warning(f"Error generating recommendation for {symbol}: {e}")
                    continue
            
            # Sort by confidence descending
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.exception("Error generating watchlist recommendations")
            return []
    
    def generate_portfolio_recommendations(self,
                                         include_details: bool = False,
                                         max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """Generate recommendations for portfolio positions with P/L context"""
        try:
            recommendations = []
            portfolio_positions = self._load_portfolio_positions()
            
            for symbol in PORTFOLIO:
                try:
                    # Get position data
                    position = portfolio_positions.get(symbol, {})
                    
                    # Gather data for recommendation engine
                    technical_data = self._get_technical_data(symbol)
                    momentum_data = self._get_momentum_data(symbol)
                    news_data = self._get_news_data(symbol)
                    earnings_data = self._get_earnings_data(symbol)
                    
                    # Calculate portfolio context with real position data
                    portfolio_data = self._calculate_portfolio_context(symbol, position, technical_data)
                    
                    # Generate recommendation
                    rec = generate_symbol_recommendation(
                        symbol, technical_data, momentum_data, news_data, earnings_data, portfolio_data
                    )
                    
                    # Add portfolio-specific context
                    rec['context'] = 'portfolio'
                    rec['current_price'] = technical_data['price_current']
                    rec['position_status'] = 'held'
                    rec['position_details'] = {
                        'quantity': portfolio_data['quantity'],
                        'cost_basis': portfolio_data['cost_basis'],
                        'current_value': portfolio_data['current_value'],
                        'unrealized_pnl': portfolio_data['current_value'] - (portfolio_data['quantity'] * portfolio_data['cost_basis']),
                        'unrealized_pnl_pct': portfolio_data['unrealized_pnl_pct'],
                        'position_pct_of_portfolio': portfolio_data['position_pct'],
                        'days_held': position.get('days_held', 'N/A')
                    }
                    
                    if include_details:
                        rec['details'] = {
                            'technical_signals': self._summarize_technical_signals(technical_data),
                            'momentum_signals': self._summarize_momentum_signals(momentum_data),
                            'news_sentiment': news_data['consensus_14d'],
                            'earnings_proximity': earnings_data.get('hours_until_earnings', 'N/A'),
                            'risk_metrics': self._calculate_risk_metrics(portfolio_data, technical_data)
                        }
                    
                    recommendations.append(rec)
                    
                except Exception as e:
                    logger.warning(f"Error generating recommendation for {symbol}: {e}")
                    continue
            
            # Sort by conviction and position size
            recommendations.sort(key=lambda x: (x['confidence'], x['position_details']['position_pct_of_portfolio']), reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.exception("Error generating portfolio recommendations")
            return []
    
    def _load_portfolio_positions(self) -> Dict[str, Dict[str, Any]]:
        """Load portfolio positions from CSV file"""
        try:
            if not self.portfolio_path.exists():
                logger.warning("Portfolio file not found, using empty positions")
                return {}
            
            df = pd.read_csv(self.portfolio_path)
            positions = {}
            
            for _, row in df.iterrows():
                symbol = row.get('ticker', row.get('Ticker', '')).upper()
                if symbol:
                    positions[symbol] = {
                        'quantity': float(row.get('quantity', row.get('Quantity', 0))),
                        'cost_basis': float(row.get('cost_basis', row.get('CostBasis', 0))),
                        'purchase_date': row.get('purchase_date', row.get('PurchaseDate', '')),
                        'days_held': self._calculate_days_held(row.get('purchase_date', ''))
                    }
            
            return positions
            
        except Exception as e:
            logger.error(f"Error loading portfolio positions: {e}")
            return {}
    
    def _calculate_days_held(self, purchase_date_str: str) -> int:
        """Calculate days held from purchase date string"""
        try:
            if not purchase_date_str:
                return 0
            purchase_date = pd.to_datetime(purchase_date_str).date()
            return (datetime.now().date() - purchase_date).days
        except:
            return 0
    
    def _get_technical_data(self, symbol: str) -> Dict[str, float]:
        """Get technical indicator data for symbol"""
        try:
            # Fetch price history
            price_df = fetch_price_history(symbol, period="90d")
            if price_df.empty:
                raise ValueError(f"No price data for {symbol}")
            
            current_price = float(price_df['Close'].iloc[-1])
            
            # Calculate moving averages
            price_df['MA20'] = price_df['Close'].rolling(20).mean()
            price_df['MA50'] = price_df['Close'].rolling(50).mean()
            price_df['MA200'] = price_df['Close'].rolling(200).mean()
            
            # Calculate RSI
            price_df['RSI'] = self._calculate_rsi(price_df['Close'])
            
            # Calculate MACD
            price_df['MACD_hist'] = self._calculate_macd_histogram(price_df['Close'])
            
            return {
                'price_current': current_price,
                'dma_20': float(price_df['MA20'].iloc[-1]) if not pd.isna(price_df['MA20'].iloc[-1]) else current_price,
                'dma_50': float(price_df['MA50'].iloc[-1]) if not pd.isna(price_df['MA50'].iloc[-1]) else current_price,
                'dma_200': float(price_df['MA200'].iloc[-1]) if not pd.isna(price_df['MA200'].iloc[-1]) else current_price,
                'rsi': float(price_df['RSI'].iloc[-1]) if not pd.isna(price_df['RSI'].iloc[-1]) else 50.0,
                'macd_histogram': float(price_df['MACD_hist'].iloc[-1]) if not pd.isna(price_df['MACD_hist'].iloc[-1]) else 0.0,
                'volatility': float(price_df['Close'].pct_change().std() * (252 ** 0.5)) if len(price_df) > 1 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting technical data for {symbol}: {e}")
            # Return neutral defaults
            return {
                'price_current': 100.0, 'dma_20': 100.0, 'dma_50': 100.0, 'dma_200': 100.0,
                'rsi': 50.0, 'macd_histogram': 0.0, 'volatility': 0.0
            }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd_histogram(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD histogram"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        return histogram
    
    def _get_momentum_data(self, symbol: str) -> Dict[str, float]:
        """Get momentum vs sector data (simplified for now)"""
        try:
            price_df = fetch_price_history(symbol, period="180d")
            if price_df.empty:
                raise ValueError(f"No price data for {symbol}")
            
            # Calculate momentum periods
            current = price_df['Close'].iloc[-1]
            
            # 1-month momentum (approximate)
            month_1_price = price_df['Close'].iloc[-22] if len(price_df) >= 22 else price_df['Close'].iloc[0]
            momentum_1m = (current - month_1_price) / month_1_price
            
            # 3-month momentum
            month_3_price = price_df['Close'].iloc[-66] if len(price_df) >= 66 else price_df['Close'].iloc[0]
            momentum_3m = (current - month_3_price) / month_3_price
            
            # 6-month momentum
            month_6_price = price_df['Close'].iloc[-132] if len(price_df) >= 132 else price_df['Close'].iloc[0]
            momentum_6m = (current - month_6_price) / month_6_price
            
            return {
                'momentum_1m': momentum_1m,
                'momentum_3m': momentum_3m,
                'momentum_6m': momentum_6m
            }
            
        except Exception as e:
            logger.error(f"Error getting momentum data for {symbol}: {e}")
            return {'momentum_1m': 0.0, 'momentum_3m': 0.0, 'momentum_6m': 0.0}
    
    def _get_news_data(self, symbol: str) -> Dict[str, Any]:
        """Get news consensus data"""
        try:
            headlines_data = scrape_headlines([symbol], days=14)
            symbol_data = headlines_data.get(symbol, {})
            
            # Calculate consensus from daily sentiment
            daily_sentiment = symbol_data.get('daily_sentiment', {})
            if daily_sentiment:
                consensus = sum(daily_sentiment.values()) / len(daily_sentiment)
                
                # Determine trend direction (simplified)
                recent_days = list(daily_sentiment.values())[-7:] if len(daily_sentiment) >= 7 else list(daily_sentiment.values())
                if len(recent_days) >= 2:
                    trend = "rising" if recent_days[-1] > recent_days[0] else "falling"
                else:
                    trend = "stable"
            else:
                consensus = 0.0
                trend = "stable"
            
            return {
                'consensus_14d': consensus,
                'trend_direction': trend,
                'confidence': 0.7  # Base confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting news data for {symbol}: {e}")
            return {'consensus_14d': 0.0, 'trend_direction': 'stable', 'confidence': 0.5}
    
    def _get_earnings_data(self, symbol: str) -> Dict[str, Any]:
        """Get earnings calendar data"""
        try:
            earnings_calendar_path = self.data_path / "earnings_calendar.csv"
            if not earnings_calendar_path.exists():
                return {'hours_until_earnings': None}
            
            calendar_df = pd.read_csv(earnings_calendar_path)
            symbol_earnings = calendar_df[calendar_df['ticker'].str.upper() == symbol.upper()]
            
            if symbol_earnings.empty:
                return {'hours_until_earnings': None}
            
            # Get next earnings
            today = pd.Timestamp.now()
            future_earnings = symbol_earnings[pd.to_datetime(symbol_earnings['earnings_date']) >= today]
            
            if future_earnings.empty:
                return {'hours_until_earnings': None}
            
            next_earnings_date = pd.to_datetime(future_earnings.iloc[0]['earnings_date'])
            hours_until = (next_earnings_date - today).total_seconds() / 3600
            
            return {
                'next_earnings_date': next_earnings_date,
                'hours_until_earnings': hours_until,
                'confirmed': bool(future_earnings.iloc[0].get('confirmed', False))
            }
            
        except Exception as e:
            logger.error(f"Error getting earnings data for {symbol}: {e}")
            return {'hours_until_earnings': None}
    
    def _calculate_portfolio_context(self, symbol: str, position: Dict[str, Any], technical_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio context for recommendation engine"""
        try:
            quantity = position.get('quantity', 0.0)
            cost_basis = position.get('cost_basis', 0.0)
            current_price = technical_data['price_current']
            
            current_value = quantity * current_price
            total_invested = quantity * cost_basis
            
            # Calculate P/L
            unrealized_pnl_pct = ((current_value - total_invested) / total_invested) if total_invested > 0 else 0.0
            
            # Estimate position as percentage of portfolio (simplified)
            # In practice, this would come from portfolio management system
            position_pct = min(current_value / 100000.0, 0.5)  # Assume $100k portfolio, cap at 50%
            
            return {
                'quantity': quantity,
                'cost_basis': cost_basis,
                'current_value': current_value,
                'position_pct': position_pct,
                'unrealized_pnl_pct': unrealized_pnl_pct
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio context for {symbol}: {e}")
            return {
                'quantity': 0.0, 'cost_basis': 0.0, 'current_value': 0.0,
                'position_pct': 0.0, 'unrealized_pnl_pct': 0.0
            }
    
    def _summarize_technical_signals(self, technical_data: Dict[str, float]) -> Dict[str, str]:
        """Summarize technical signals for details"""
        price = technical_data['price_current']
        ma20 = technical_data['dma_20']
        ma50 = technical_data['dma_50']
        rsi = technical_data['rsi']
        
        trend = "bullish" if price > ma20 > ma50 else "bearish" if price < ma20 < ma50 else "mixed"
        rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        
        return {
            'trend': trend,
            'rsi_signal': rsi_signal,
            'price_vs_ma20': f"{((price - ma20) / ma20 * 100):+.1f}%"
        }
    
    def _summarize_momentum_signals(self, momentum_data: Dict[str, float]) -> Dict[str, str]:
        """Summarize momentum signals for details"""
        mom_1m = momentum_data['momentum_1m']
        mom_3m = momentum_data['momentum_3m']
        
        return {
            '1_month': f"{mom_1m * 100:+.1f}%",
            '3_month': f"{mom_3m * 100:+.1f}%",
            'trend': "positive" if mom_1m > 0 and mom_3m > 0 else "negative" if mom_1m < 0 and mom_3m < 0 else "mixed"
        }
    
    def _calculate_risk_metrics(self, portfolio_data: Dict[str, float], technical_data: Dict[str, float]) -> Dict[str, Any]:
        """Calculate risk metrics for portfolio positions"""
        return {
            'position_risk': "high" if portfolio_data['position_pct'] > 0.15 else "normal",
            'volatility': f"{technical_data['volatility'] * 100:.1f}%",
            'unrealized_pnl_risk': "profit_taking" if portfolio_data['unrealized_pnl_pct'] > 0.2 else "stop_loss" if portfolio_data['unrealized_pnl_pct'] < -0.1 else "normal"
        }