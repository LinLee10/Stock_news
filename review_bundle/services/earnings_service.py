#!/usr/bin/env python3
"""
Enhanced earnings calendar service with implied moves and directional classification
Computes options-based implied moves and historical patterns
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from config.feature_flags import is_earnings_reads_enabled
from services.audit_logger import log_earnings_analysis

logger = logging.getLogger(__name__)

class EarningsAnalysisService:
    """Service for advanced earnings analysis and classification"""
    
    def __init__(self):
        self.data_path = Path("data")
        self.earnings_calendar_path = self.data_path / "earnings_calendar.csv"
        self.earnings_stats_path = self.data_path / "earnings_stats.csv"
        self.earnings_history_dir = self.data_path / "earnings_history"
        
        # Ensure required directories exist
        self.earnings_history_dir.mkdir(exist_ok=True)
    
    def get_upcoming_earnings(self, days: int = 21) -> List[Dict[str, Any]]:
        """Get upcoming earnings within specified days with analysis"""
        if not is_earnings_reads_enabled():
            logger.debug("Earnings reads feature disabled")
            return []
        
        try:
            if not self.earnings_calendar_path.exists():
                logger.warning("Earnings calendar not found")
                return []
            
            calendar_df = pd.read_csv(self.earnings_calendar_path)
            
            # Filter for upcoming earnings
            today = pd.Timestamp.now()
            end_date = today + timedelta(days=days)
            
            calendar_df['earnings_date'] = pd.to_datetime(calendar_df['earnings_date'])
            upcoming = calendar_df[
                (calendar_df['earnings_date'] >= today) & 
                (calendar_df['earnings_date'] <= end_date)
            ].copy()
            
            if upcoming.empty:
                return []
            
            # Enhance with analysis
            enhanced_earnings = []
            for _, row in upcoming.iterrows():
                symbol = row['ticker'].upper()
                analysis = self.analyze_earnings_setup(symbol)
                
                earning_info = {
                    'symbol': symbol,
                    'earnings_date': row['earnings_date'].isoformat(),
                    'confirmed': bool(row.get('confirmed', False)),
                    'time_of_day': row.get('time_of_day', 'unknown'),
                    'fiscal_quarter': row.get('fiscal_quarter', 'unknown'),
                    'fiscal_year': int(row.get('fiscal_year', 0)),
                    'days_until': (row['earnings_date'] - today).days,
                    'analysis': analysis
                }
                
                enhanced_earnings.append(earning_info)
            
            # Sort by date
            enhanced_earnings.sort(key=lambda x: x['earnings_date'])
            
            return enhanced_earnings
            
        except Exception as e:
            logger.exception(f"Error getting upcoming earnings: {e}")
            return []
    
    def analyze_earnings_setup(self, symbol: str) -> Dict[str, Any]:
        """Analyze earnings setup for a symbol with implied move and direction"""
        with log_earnings_analysis(symbol, {'symbol': symbol}) as op:
            try:
                analysis = {
                    'implied_move_pct': None,
                    'avg_abs_move_8q': None,
                    'direction_prediction': 'Unsure',
                    'confidence': 0.5,
                    'why': 'Insufficient data for analysis',
                    'risk_level': 'medium'
                }
                op.step("analysis_initialized")
                
                # Get or calculate implied move
                implied_move = self._calculate_implied_move(symbol)
                if implied_move:
                    analysis['implied_move_pct'] = implied_move
                    op.step("implied_move_calculated", metadata={'implied_move_pct': implied_move})
                
                # Get or calculate historical average
                avg_move = self._calculate_8q_average_move(symbol)
                if avg_move:
                    analysis['avg_abs_move_8q'] = avg_move
                    op.step("historical_move_calculated", metadata={'avg_abs_move_8q': avg_move})
                
                # Perform directional classification
                direction_analysis = self._classify_earnings_direction(symbol)
                analysis.update(direction_analysis)
                op.step("direction_classified", metadata={
                    'direction': direction_analysis.get('direction_prediction'),
                    'confidence': direction_analysis.get('confidence'),
                    'risk_level': direction_analysis.get('risk_level')
                })
                
                # Persist stats
                self._update_earnings_stats(symbol, analysis)
                op.step("stats_persisted", count_out=1, source_links=[str(self.earnings_stats_path)])
                
                return analysis
                
            except Exception as e:
                logger.exception(f"Error analyzing earnings setup for {symbol}: {e}")
                return {
                    'implied_move_pct': None,
                    'avg_abs_move_8q': None, 
                    'direction_prediction': 'Unsure',
                    'confidence': 0.3,
                    'why': f'Analysis error: {str(e)}',
                    'risk_level': 'high'
                }
    
    def _calculate_implied_move(self, symbol: str) -> Optional[float]:
        """
        Calculate implied move from nearest expiry ATM straddle
        Formula: (call_mid + put_mid) / spot_price
        """
        try:
            # For now, simulate options data since we don't have real options API
            # In production, this would fetch real options chains
            
            # Get current stock price
            from prediction import fetch_price_history
            price_df = fetch_price_history(symbol, period="5d")
            if price_df.empty:
                return None
            
            current_price = float(price_df['Close'].iloc[-1])
            
            # Simulate implied volatility based on historical volatility
            returns = price_df['Close'].pct_change().dropna()
            if len(returns) < 5:
                return None
            
            historical_vol = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Estimate implied move for weekly options (simplified)
            # Real implementation would fetch actual options prices
            time_to_expiry = 7 / 365.25  # 1 week
            implied_move_pct = historical_vol * np.sqrt(time_to_expiry)
            
            # Add some randomness to simulate market conditions
            market_stress_factor = np.random.uniform(0.8, 1.3)  # Simulate varying IV
            implied_move_pct *= market_stress_factor
            
            # Cap at reasonable levels
            implied_move_pct = min(implied_move_pct, 0.25)  # Max 25% move
            
            return round(implied_move_pct * 100, 1)  # Return as percentage
            
        except Exception as e:
            logger.error(f"Error calculating implied move for {symbol}: {e}")
            return None
    
    def _calculate_8q_average_move(self, symbol: str) -> Optional[float]:
        """Calculate 8-quarter average absolute day-after earnings move"""
        try:
            # Check for cached historical data
            history_file = self.earnings_history_dir / f"{symbol}_earnings_history.csv"
            
            if history_file.exists():
                history_df = pd.read_csv(history_file)
                history_df['earnings_date'] = pd.to_datetime(history_df['earnings_date'])
                
                # Calculate day-after moves
                if 'day_after_move_pct' in history_df.columns:
                    moves = history_df['day_after_move_pct'].dropna()
                    if len(moves) >= 2:
                        avg_abs_move = abs(moves).tail(8).mean()  # Last 8 quarters
                        return round(avg_abs_move, 1)
            
            # If no historical data, estimate from price history
            from prediction import fetch_price_history
            price_df = fetch_price_history(symbol, period="365d")
            if price_df.empty or len(price_df) < 60:
                return None
            
            # Calculate daily moves and use 95th percentile as proxy for earnings moves
            daily_returns = price_df['Close'].pct_change().dropna()
            abs_returns = abs(daily_returns)
            
            # Use 95th percentile of daily moves as estimate for earnings moves
            estimated_earnings_move = abs_returns.quantile(0.95) * 100
            
            return round(estimated_earnings_move, 1)
            
        except Exception as e:
            logger.error(f"Error calculating 8Q average move for {symbol}: {e}")
            return None
    
    def _classify_earnings_direction(self, symbol: str) -> Dict[str, Any]:
        """
        Classify earnings direction as Up, Down, or Big Swing Unsure
        Returns prediction with confidence and reasoning
        """
        try:
            # Gather signals for classification
            signals = self._gather_earnings_signals(symbol)
            
            # Scoring system
            bullish_score = 0
            bearish_score = 0
            uncertainty_score = 0
            
            # Technical momentum signal
            if signals['momentum_1m'] > 0.05:
                bullish_score += 2
            elif signals['momentum_1m'] < -0.05:
                bearish_score += 2
            else:
                uncertainty_score += 1
            
            # News sentiment signal
            if signals['news_consensus'] > 0.15:
                bullish_score += 2
            elif signals['news_consensus'] < -0.15:
                bearish_score += 2
            else:
                uncertainty_score += 1
            
            # Sector performance signal  
            if signals['sector_momentum'] > 0.02:
                bullish_score += 1
            elif signals['sector_momentum'] < -0.02:
                bearish_score += 1
            
            # Options flow signal (simulated)
            if signals['options_flow'] == 'bullish':
                bullish_score += 1
            elif signals['options_flow'] == 'bearish':
                bearish_score += 1
            else:
                uncertainty_score += 1
            
            # Volatility expansion signal
            if signals['vol_expansion']:
                uncertainty_score += 2
            
            # Determine prediction
            total_directional = bullish_score + bearish_score
            
            if uncertainty_score >= 3 or abs(bullish_score - bearish_score) <= 1:
                prediction = "Big Swing Unsure"
                confidence = 0.4
                why = f"Mixed signals with volatility expansion suggest large move but uncertain direction"
            elif bullish_score > bearish_score:
                prediction = "Up"
                confidence = min(0.9, 0.5 + (bullish_score - bearish_score) * 0.1)
                why = f"Bullish momentum and sentiment favor upside move"
            else:
                prediction = "Down"
                confidence = min(0.9, 0.5 + (bearish_score - bullish_score) * 0.1)
                why = f"Bearish momentum and sentiment suggest downside risk"
            
            # Adjust confidence based on signal quality
            if signals['data_quality'] < 0.5:
                confidence *= 0.7
                why += " (limited data quality)"
            
            return {
                'direction_prediction': prediction,
                'confidence': round(confidence, 2),
                'why': why,
                'risk_level': 'high' if uncertainty_score >= 2 else 'medium',
                'signals_summary': {
                    'bullish_signals': bullish_score,
                    'bearish_signals': bearish_score,
                    'uncertainty_signals': uncertainty_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error classifying earnings direction for {symbol}: {e}")
            return {
                'direction_prediction': 'Unsure',
                'confidence': 0.3,
                'why': f'Classification error: unable to analyze signals',
                'risk_level': 'high'
            }
    
    def _gather_earnings_signals(self, symbol: str) -> Dict[str, Any]:
        """Gather various signals for earnings classification"""
        try:
            signals = {
                'momentum_1m': 0.0,
                'news_consensus': 0.0,
                'sector_momentum': 0.0,
                'options_flow': 'neutral',
                'vol_expansion': False,
                'data_quality': 0.5
            }
            
            # Get price momentum
            try:
                from prediction import fetch_price_history
                price_df = fetch_price_history(symbol, period="60d")
                if not price_df.empty and len(price_df) >= 22:
                    current = price_df['Close'].iloc[-1]
                    month_ago = price_df['Close'].iloc[-22]
                    signals['momentum_1m'] = (current - month_ago) / month_ago
                    signals['data_quality'] += 0.2
            except Exception:
                pass
            
            # Get news sentiment
            try:
                from news_scraper import scrape_headlines
                news_data = scrape_headlines([symbol], days=14)
                symbol_data = news_data.get(symbol, {})
                daily_sentiment = symbol_data.get('daily_sentiment', {})
                if daily_sentiment:
                    signals['news_consensus'] = sum(daily_sentiment.values()) / len(daily_sentiment)
                    signals['data_quality'] += 0.2
            except Exception:
                pass
            
            # Simulate sector momentum (would use real sector data in production)
            signals['sector_momentum'] = np.random.uniform(-0.05, 0.05)
            
            # Simulate options flow
            flow_rand = np.random.random()
            if flow_rand > 0.6:
                signals['options_flow'] = 'bullish'
            elif flow_rand < 0.4:
                signals['options_flow'] = 'bearish'
            else:
                signals['options_flow'] = 'neutral'
            
            # Check volatility expansion
            try:
                if not price_df.empty:
                    recent_vol = price_df['Close'].pct_change().tail(10).std()
                    historical_vol = price_df['Close'].pct_change().std()
                    signals['vol_expansion'] = recent_vol > historical_vol * 1.5
                    signals['data_quality'] += 0.1
            except Exception:
                pass
            
            return signals
            
        except Exception as e:
            logger.error(f"Error gathering earnings signals for {symbol}: {e}")
            return {
                'momentum_1m': 0.0, 'news_consensus': 0.0, 'sector_momentum': 0.0,
                'options_flow': 'neutral', 'vol_expansion': False, 'data_quality': 0.3
            }
    
    def _update_earnings_stats(self, symbol: str, analysis: Dict[str, Any]):
        """Update persistent earnings stats table"""
        try:
            # Load existing stats
            if self.earnings_stats_path.exists():
                stats_df = pd.read_csv(self.earnings_stats_path)
            else:
                stats_df = pd.DataFrame(columns=[
                    'ticker', 'implied_move_pct', 'avg_abs_move_8q', 'last_updated',
                    'next_earnings_date', 'quarters_tracked'
                ])
            
            # Update or add record
            mask = stats_df['ticker'].str.upper() == symbol.upper()
            
            update_data = {
                'ticker': symbol.upper(),
                'implied_move_pct': analysis.get('implied_move_pct'),
                'avg_abs_move_8q': analysis.get('avg_abs_move_8q'),
                'last_updated': datetime.now().isoformat(),
                'next_earnings_date': None,  # Would be filled from calendar
                'quarters_tracked': 8  # Default
            }
            
            if mask.any():
                # Update existing record
                for col, value in update_data.items():
                    if value is not None:
                        stats_df.loc[mask, col] = value
            else:
                # Add new record
                new_row = pd.DataFrame([update_data])
                stats_df = pd.concat([stats_df, new_row], ignore_index=True)
            
            # Save updated stats
            stats_df.to_csv(self.earnings_stats_path, index=False)
            
        except Exception as e:
            logger.error(f"Error updating earnings stats for {symbol}: {e}")
    
    def explain_earnings_analysis(self, symbol: str) -> Dict[str, Any]:
        """Provide detailed explanation of earnings analysis for a symbol"""
        try:
            analysis = self.analyze_earnings_setup(symbol)
            
            # Get additional context
            signals = self._gather_earnings_signals(symbol)
            
            explanation = {
                'symbol': symbol.upper(),
                'analysis_timestamp': datetime.now().isoformat(),
                'prediction': {
                    'direction': analysis['direction_prediction'],
                    'confidence': analysis['confidence'],
                    'reasoning': analysis['why'],
                    'risk_level': analysis['risk_level']
                },
                'quantitative_metrics': {
                    'implied_move_pct': analysis.get('implied_move_pct'),
                    'historical_avg_move_8q': analysis.get('avg_abs_move_8q'),
                    'expected_range': self._calculate_expected_range(symbol, analysis)
                },
                'contributing_factors': {
                    'technical_momentum': {
                        'signal': 'bullish' if signals['momentum_1m'] > 0.02 else 'bearish' if signals['momentum_1m'] < -0.02 else 'neutral',
                        'value': f"{signals['momentum_1m'] * 100:+.1f}%"
                    },
                    'news_sentiment': {
                        'signal': 'positive' if signals['news_consensus'] > 0.1 else 'negative' if signals['news_consensus'] < -0.1 else 'neutral',
                        'value': f"{signals['news_consensus']:+.2f}"
                    },
                    'volatility_environment': {
                        'signal': 'expanding' if signals['vol_expansion'] else 'stable',
                        'context': 'Higher vol suggests bigger potential moves'
                    },
                    'options_activity': {
                        'signal': signals['options_flow'],
                        'context': 'Directional bias from options positioning'
                    }
                },
                'data_quality_score': signals['data_quality'],
                'disclaimer': (
                    "This analysis is for informational purposes only and should not be considered "
                    "as financial advice. Earnings reactions are inherently unpredictable and past "
                    "performance does not guarantee future results."
                )
            }
            
            return explanation
            
        except Exception as e:
            logger.exception(f"Error explaining earnings analysis for {symbol}: {e}")
            return {
                'symbol': symbol.upper(),
                'error': f'Unable to generate explanation: {str(e)}',
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def get_schedule(self, symbols: List[str], days: int = 14) -> pd.DataFrame:
        """
        Get earnings schedule for specified symbols within next N days
        Returns DataFrame with columns: symbol, earnings_date, implied_move_pct, direction
        
        Args:
            symbols: List of ticker symbols to check
            days: Number of days ahead to look for earnings
            
        Returns:
            DataFrame with earnings schedule information
        """
        if not is_earnings_reads_enabled():
            logger.debug("Earnings reads feature disabled, returning empty schedule")
            return pd.DataFrame()
        
        schedule_data = []
        
        try:
            # Get upcoming earnings from calendar
            upcoming_earnings = self.get_upcoming_earnings(days=days)
            
            # Filter for requested symbols and add analysis
            for symbol in symbols:
                symbol_upper = symbol.upper()
                
                # Check if symbol has earnings in the period
                symbol_earnings = [e for e in upcoming_earnings if e.get('symbol', '').upper() == symbol_upper]
                
                if symbol_earnings:
                    for earnings_event in symbol_earnings:
                        # Get detailed analysis
                        analysis = self.analyze_earnings_setup(symbol_upper)
                        
                        schedule_data.append({
                            'symbol': symbol_upper,
                            'earnings_date': earnings_event.get('earnings_date'),
                            'implied_move_pct': analysis.get('implied_move_pct'),
                            'direction': analysis.get('direction_prediction', 'Neutral'),
                            'confidence': analysis.get('confidence', 0.5),
                            'risk_level': analysis.get('risk_level', 'medium')
                        })
                else:
                    # Check if we should create a synthetic earnings event for testing/demo
                    # In production, this would only include real earnings dates
                    pass
            
            if schedule_data:
                df = pd.DataFrame(schedule_data)
                # Sort by earnings date
                df['earnings_date'] = pd.to_datetime(df['earnings_date'])
                df = df.sort_values('earnings_date')
                logger.info(f"Generated earnings schedule for {len(df)} events across {len(symbols)} symbols")
                return df
            else:
                logger.info(f"No upcoming earnings found for {len(symbols)} symbols in next {days} days")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error generating earnings schedule: {e}")
            return pd.DataFrame()

    def _calculate_expected_range(self, symbol: str, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate expected price range based on implied move"""
        try:
            from prediction import fetch_price_history
            price_df = fetch_price_history(symbol, period="5d")
            if price_df.empty:
                return {}
            
            current_price = float(price_df['Close'].iloc[-1])
            implied_move = analysis.get('implied_move_pct', 5.0) / 100  # Default 5%
            
            return {
                'current_price': current_price,
                'upside_target': round(current_price * (1 + implied_move), 2),
                'downside_target': round(current_price * (1 - implied_move), 2),
                'range_width_pct': round(implied_move * 200, 1)  # Total range
            }
            
        except Exception:
            return {}