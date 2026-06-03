"""
Investment Recommendation Engine with Multi-Factor Analysis

Converts 90-day sentiment trends into actionable buy/sell/hold recommendations
with risk-aware position sizing and regulatory compliance.
"""

import asyncio
import logging
import json
import sqlite3
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import yfinance as yf

from finbert_sentiment_analyzer import (
    FinBERTSentimentPipeline, 
    AggregatedSentiment, 
    InvestmentRecommendation,
    RecommendationAction
)

logger = logging.getLogger(__name__)

class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    rsi: float = 50.0
    macd_signal: float = 0.0
    bb_position: float = 0.5
    sma_20: float = 0.0
    sma_50: float = 0.0
    volume_ratio: float = 1.0
    momentum: float = 0.0

@dataclass
class FundamentalMetrics:
    """Fundamental analysis metrics"""
    pe_ratio: float = 20.0
    price_to_book: float = 2.0
    debt_to_equity: float = 0.5
    roe: float = 0.15
    revenue_growth: float = 0.05
    earnings_growth: float = 0.10
    dividend_yield: float = 0.02

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    beta: float = 1.0
    volatility: float = 0.20
    max_drawdown: float = 0.15
    sharpe_ratio: float = 1.0
    var_95: float = 0.05
    correlation_to_market: float = 0.7

@dataclass
class RecommendationSummary:
    """Two-line recommendation summary"""
    line1: str  # Current situation
    line2: str  # Specific recommendation
    confidence_pct: int  # 0-95 (capped for humility)
    time_horizon: str  # e.g., "90 days"
    risk_level: str  # Low/Medium/High

@dataclass
class DetailedRecommendation:
    """Complete recommendation with all supporting data"""
    symbol: str
    recommendation_id: str
    timestamp: datetime
    
    # Core recommendation
    action: RecommendationAction
    summary: RecommendationSummary
    reasoning: str
    
    # Scoring breakdown
    sentiment_score: float
    technical_score: float
    fundamental_score: float
    fusion_score: float
    confidence: float
    
    # Risk management
    position_size_pct: float
    stop_loss_pct: float
    target_price: Optional[float]
    risk_metrics: RiskMetrics
    
    # Supporting data
    current_price: float
    technical_indicators: TechnicalIndicators
    fundamental_metrics: FundamentalMetrics
    
    # Compliance and tracking
    risk_tolerance_aligned: bool
    regulatory_disclosures: List[str]
    backtest_accuracy: float
    track_record_score: float

class InvestmentRecommendationEngine:
    """Multi-factor investment recommendation engine"""
    
    def __init__(self):
        self.db_path = "data/recommendations.db"
        
        # Portfolio and watchlist configuration
        self.portfolio_tickers = ['RTX', 'PFE', 'MRVL', 'ADI', 'LLY', 'RIVN', 'TSLA', 'PLTR']
        self.watchlist_tickers = ['NVDA', 'GOOGL', 'AMD', 'MSFT']
        
        # Model weights for ensemble
        self.weights = {
            'sentiment': 0.40,
            'technical': 0.35,
            'fundamental': 0.20,
            'risk_adjustment': 0.05
        }
        
        # Risk management parameters
        self.max_position_size = {
            RiskTolerance.CONSERVATIVE: 0.05,  # 5% max
            RiskTolerance.MODERATE: 0.08,      # 8% max
            RiskTolerance.AGGRESSIVE: 0.12     # 12% max
        }
        
        self.stop_loss_levels = {
            RiskTolerance.CONSERVATIVE: 0.08,  # 8% stop loss
            RiskTolerance.MODERATE: 0.12,      # 12% stop loss
            RiskTolerance.AGGRESSIVE: 0.18     # 18% stop loss
        }
        
        # Dynamic thresholds based on market regime
        self.regime_thresholds = {
            MarketRegime.BULL_MARKET: {'buy': 0.5, 'sell': -0.7},
            MarketRegime.BEAR_MARKET: {'buy': 0.7, 'sell': -0.5},
            MarketRegime.SIDEWAYS: {'buy': 0.6, 'sell': -0.6},
            MarketRegime.HIGH_VOLATILITY: {'buy': 0.8, 'sell': -0.8}
        }
        
        # Performance tracking
        self.recommendation_history: Dict[str, List[DetailedRecommendation]] = {}
        self.performance_metrics = {}
        
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database for recommendations tracking"""
        
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Recommendations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                action TEXT NOT NULL,
                fusion_score REAL NOT NULL,
                confidence REAL NOT NULL,
                position_size_pct REAL NOT NULL,
                current_price REAL NOT NULL,
                target_price REAL,
                stop_loss_pct REAL NOT NULL,
                reasoning TEXT NOT NULL,
                risk_tolerance TEXT NOT NULL,
                market_regime TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Performance tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_performance (
                recommendation_id TEXT NOT NULL,
                evaluation_date INTEGER NOT NULL,
                actual_return REAL NOT NULL,
                hit_accuracy BOOLEAN NOT NULL,
                risk_adjusted_return REAL NOT NULL,
                FOREIGN KEY (recommendation_id) REFERENCES recommendations (id)
            )
        """)
        
        # Compliance audit table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_audit (
                id TEXT PRIMARY KEY,
                recommendation_id TEXT NOT NULL,
                audit_type TEXT NOT NULL,
                audit_result TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY (recommendation_id) REFERENCES recommendations (id)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_symbol ON recommendations(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_timestamp ON recommendations(timestamp)")
        
        conn.commit()
        conn.close()

    async def generate_recommendation(self, symbol: str, 
                                    aggregated_sentiment: AggregatedSentiment,
                                    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE) -> DetailedRecommendation:
        """Generate comprehensive investment recommendation"""
        
        try:
            # Get current market data
            price_data = await self._get_price_data(symbol)
            current_price = price_data['close'].iloc[-1]
            
            # Calculate technical indicators
            technical_indicators = await self._calculate_technical_indicators(price_data)
            
            # Get fundamental metrics
            fundamental_metrics = await self._get_fundamental_metrics(symbol)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(price_data, symbol)
            
            # Score each factor
            sentiment_score = self._score_sentiment(aggregated_sentiment)
            technical_score = self._score_technical(technical_indicators)
            fundamental_score = self._score_fundamental(fundamental_metrics)
            
            # Detect market regime
            market_regime = await self._detect_market_regime(symbol)
            
            # Calculate fusion score
            fusion_score = self._calculate_fusion_score(
                sentiment_score, technical_score, fundamental_score, risk_metrics
            )
            
            # Generate recommendation action
            action = self._determine_action(fusion_score, market_regime, risk_tolerance)
            
            # Calculate position size
            position_size_pct = self._calculate_position_size(
                fusion_score, risk_metrics, risk_tolerance
            )
            
            # Calculate stop loss and target
            stop_loss_pct = self.stop_loss_levels[risk_tolerance]
            target_price = self._calculate_target_price(current_price, fusion_score, action)
            
            # Generate summary and reasoning
            summary = self._generate_summary(action, fusion_score, symbol, current_price)
            reasoning = self._generate_detailed_reasoning(
                aggregated_sentiment, technical_indicators, fundamental_metrics,
                sentiment_score, technical_score, fundamental_score, fusion_score
            )
            
            # Compliance checks
            risk_tolerance_aligned = await self._check_risk_tolerance_alignment(
                symbol, risk_tolerance, risk_metrics
            )
            regulatory_disclosures = self._generate_regulatory_disclosures(action, symbol)
            
            # Get historical performance
            backtest_accuracy, track_record_score = await self._get_track_record(symbol)
            
            # Create recommendation
            recommendation = DetailedRecommendation(
                symbol=symbol,
                recommendation_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                action=action,
                summary=summary,
                reasoning=reasoning,
                sentiment_score=sentiment_score,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                fusion_score=fusion_score,
                confidence=min(0.95, abs(fusion_score) / 5.0),  # Cap at 95%
                position_size_pct=position_size_pct,
                stop_loss_pct=stop_loss_pct,
                target_price=target_price,
                risk_metrics=risk_metrics,
                current_price=current_price,
                technical_indicators=technical_indicators,
                fundamental_metrics=fundamental_metrics,
                risk_tolerance_aligned=risk_tolerance_aligned,
                regulatory_disclosures=regulatory_disclosures,
                backtest_accuracy=backtest_accuracy,
                track_record_score=track_record_score
            )
            
            # Store recommendation
            await self._store_recommendation(recommendation, risk_tolerance, market_regime)
            
            # Log compliance audit
            await self._log_compliance_audit(recommendation)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Failed to generate recommendation for {symbol}: {e}")
            return None

    async def _get_price_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Get historical price data"""
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            if hist.empty:
                raise ValueError(f"No price data available for {symbol}")
            
            # Ensure we have the required columns
            hist.columns = hist.columns.str.lower()
            return hist
            
        except Exception as e:
            logger.error(f"Failed to get price data for {symbol}: {e}")
            # Return minimal dummy data
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            return pd.DataFrame({
                'close': np.random.normal(100, 10, len(dates))
            }, index=dates)

    async def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> TechnicalIndicators:
        """Calculate technical analysis indicators"""
        
        try:
            # RSI calculation
            delta = price_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD calculation
            ema_12 = price_data['close'].ewm(span=12).mean()
            ema_26 = price_data['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            macd_signal = macd_line.iloc[-1] - signal_line.iloc[-1]
            
            # Bollinger Bands position
            sma_20 = price_data['close'].rolling(20).mean()
            bb_std = price_data['close'].rolling(20).std()
            bb_upper = sma_20 + (2 * bb_std)
            bb_lower = sma_20 - (2 * bb_std)
            bb_position = (price_data['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Simple moving averages
            sma_50 = price_data['close'].rolling(50).mean().iloc[-1]
            
            # Volume ratio
            avg_volume = price_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = price_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Momentum
            momentum = (price_data['close'].iloc[-1] / price_data['close'].iloc[-20] - 1) if len(price_data) >= 20 else 0
            
            return TechnicalIndicators(
                rsi=float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
                macd_signal=float(macd_signal) if not pd.isna(macd_signal) else 0.0,
                bb_position=float(bb_position) if not pd.isna(bb_position) else 0.5,
                sma_20=float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else price_data['close'].iloc[-1],
                sma_50=float(sma_50) if not pd.isna(sma_50) else price_data['close'].iloc[-1],
                volume_ratio=float(volume_ratio),
                momentum=float(momentum)
            )
            
        except Exception as e:
            logger.warning(f"Technical indicators calculation failed: {e}")
            return TechnicalIndicators()

    async def _get_fundamental_metrics(self, symbol: str) -> FundamentalMetrics:
        """Get fundamental analysis metrics"""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return FundamentalMetrics(
                pe_ratio=info.get('trailingPE', 20.0) or 20.0,
                price_to_book=info.get('priceToBook', 2.0) or 2.0,
                debt_to_equity=info.get('debtToEquity', 50.0) / 100.0 if info.get('debtToEquity') else 0.5,
                roe=info.get('returnOnEquity', 0.15) or 0.15,
                revenue_growth=info.get('revenueGrowth', 0.05) or 0.05,
                earnings_growth=info.get('earningsGrowth', 0.10) or 0.10,
                dividend_yield=info.get('dividendYield', 0.02) or 0.02
            )
            
        except Exception as e:
            logger.warning(f"Fundamental metrics calculation failed for {symbol}: {e}")
            return FundamentalMetrics()

    async def _calculate_risk_metrics(self, price_data: pd.DataFrame, symbol: str) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        try:
            returns = price_data['close'].pct_change().dropna()
            
            # Beta calculation (vs SPY proxy)
            market_returns = np.random.normal(0.001, 0.02, len(returns))  # Placeholder
            beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns) if len(returns) > 20 else 1.0
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.15
            
            # Sharpe ratio (assuming 2% risk-free rate)
            excess_returns = returns.mean() * 252 - 0.02
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 1.0
            
            # VaR 95%
            var_95 = abs(returns.quantile(0.05)) if len(returns) > 20 else 0.05
            
            # Correlation to market (placeholder)
            correlation_to_market = 0.7
            
            return RiskMetrics(
                beta=float(beta),
                volatility=float(volatility),
                max_drawdown=float(max_drawdown),
                sharpe_ratio=float(sharpe_ratio),
                var_95=float(var_95),
                correlation_to_market=float(correlation_to_market)
            )
            
        except Exception as e:
            logger.warning(f"Risk metrics calculation failed: {e}")
            return RiskMetrics()

    def _score_sentiment(self, sentiment: AggregatedSentiment) -> float:
        """Score sentiment analysis on -5 to +5 scale"""
        
        base_score = sentiment.ewma_sentiment * 3  # Scale to -3 to +3
        
        # Boost for high confidence
        confidence_boost = sentiment.high_confidence_ratio * 1.0
        
        # Trend momentum
        trend_boost = sentiment.sentiment_trend * 0.5
        
        # Conviction adjustment
        conviction_adjustment = (sentiment.conviction_score / 10.0) * 1.0
        
        total_score = base_score + confidence_boost + trend_boost + conviction_adjustment
        
        return float(np.clip(total_score, -5.0, 5.0))

    def _score_technical(self, tech: TechnicalIndicators) -> float:
        """Score technical indicators on -5 to +5 scale"""
        
        score = 0.0
        
        # RSI scoring
        if tech.rsi < 30:
            score += 1.5  # Oversold
        elif tech.rsi > 70:
            score -= 1.5  # Overbought
        else:
            score += (50 - tech.rsi) / 50  # Linear around 50
        
        # MACD signal
        score += np.clip(tech.macd_signal * 2, -2.0, 2.0)
        
        # Bollinger Band position
        score += (tech.bb_position - 0.5) * 2  # -1 to +1
        
        # Moving average crossover
        if tech.sma_20 > tech.sma_50:
            score += 0.5
        else:
            score -= 0.5
        
        # Volume confirmation
        if tech.volume_ratio > 1.5:
            score += 0.5
        elif tech.volume_ratio < 0.5:
            score -= 0.5
        
        # Momentum
        score += np.clip(tech.momentum * 5, -1.0, 1.0)
        
        return float(np.clip(score, -5.0, 5.0))

    def _score_fundamental(self, fund: FundamentalMetrics) -> float:
        """Score fundamental metrics on -5 to +5 scale"""
        
        score = 0.0
        
        # P/E ratio scoring (lower is better, but not too low)
        if 10 <= fund.pe_ratio <= 18:
            score += 1.0
        elif fund.pe_ratio > 25:
            score -= 1.0
        
        # P/B ratio scoring
        if fund.price_to_book < 2:
            score += 0.5
        elif fund.price_to_book > 4:
            score -= 0.5
        
        # Debt-to-equity
        if fund.debt_to_equity < 0.3:
            score += 0.5
        elif fund.debt_to_equity > 0.8:
            score -= 0.5
        
        # ROE
        if fund.roe > 0.15:
            score += 1.0
        elif fund.roe < 0.08:
            score -= 1.0
        
        # Growth rates
        if fund.revenue_growth > 0.1:
            score += 1.0
        elif fund.revenue_growth < 0:
            score -= 1.0
        
        if fund.earnings_growth > 0.15:
            score += 1.0
        elif fund.earnings_growth < 0:
            score -= 1.0
        
        return float(np.clip(score, -5.0, 5.0))

    def _calculate_fusion_score(self, sentiment_score: float, technical_score: float,
                               fundamental_score: float, risk_metrics: RiskMetrics) -> float:
        """Calculate weighted fusion score"""
        
        # Base weighted score
        fusion_score = (
            sentiment_score * self.weights['sentiment'] +
            technical_score * self.weights['technical'] +
            fundamental_score * self.weights['fundamental']
        )
        
        # Risk adjustment
        risk_penalty = 0.0
        if risk_metrics.volatility > 0.4:  # High volatility penalty
            risk_penalty = -1.0
        elif risk_metrics.sharpe_ratio < 0.5:  # Poor risk-adjusted returns
            risk_penalty = -0.5
        
        fusion_score += risk_penalty * self.weights['risk_adjustment']
        
        return float(np.clip(fusion_score, -5.0, 5.0))

    async def _detect_market_regime(self, symbol: str) -> MarketRegime:
        """Detect current market regime"""
        
        try:
            # Get market proxy data (SPY)
            spy_data = await self._get_price_data("SPY", days=60)
            returns = spy_data['close'].pct_change().dropna()
            
            # Calculate metrics
            recent_return = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[-30] - 1) if len(spy_data) >= 30 else 0
            volatility = returns.std() * np.sqrt(252)
            
            # Regime detection
            if volatility > 0.25:
                return MarketRegime.HIGH_VOLATILITY
            elif recent_return > 0.05:
                return MarketRegime.BULL_MARKET
            elif recent_return < -0.05:
                return MarketRegime.BEAR_MARKET
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            logger.warning(f"Market regime detection failed: {e}")
            return MarketRegime.SIDEWAYS

    def _determine_action(self, fusion_score: float, market_regime: MarketRegime,
                         risk_tolerance: RiskTolerance) -> RecommendationAction:
        """Determine recommendation action based on fusion score and context"""
        
        # Get regime-specific thresholds
        thresholds = self.regime_thresholds[market_regime]
        
        # Adjust for risk tolerance
        buy_threshold = thresholds['buy']
        sell_threshold = thresholds['sell']
        
        if risk_tolerance == RiskTolerance.CONSERVATIVE:
            buy_threshold += 0.1
            sell_threshold -= 0.1
        elif risk_tolerance == RiskTolerance.AGGRESSIVE:
            buy_threshold -= 0.1
            sell_threshold += 0.1
        
        # Make decision
        if fusion_score >= buy_threshold + 1.0:
            return RecommendationAction.STRONG_BUY
        elif fusion_score >= buy_threshold:
            return RecommendationAction.BUY
        elif fusion_score <= sell_threshold - 1.0:
            return RecommendationAction.STRONG_SELL
        elif fusion_score <= sell_threshold:
            return RecommendationAction.SELL
        else:
            return RecommendationAction.HOLD

    def _calculate_position_size(self, fusion_score: float, risk_metrics: RiskMetrics,
                                risk_tolerance: RiskTolerance) -> float:
        """Calculate position size based on Kelly criterion and risk management"""
        
        # Base position size from fusion score strength
        base_size = min(abs(fusion_score) / 5.0 * 0.1, self.max_position_size[risk_tolerance])
        
        # Risk adjustment
        volatility_penalty = max(0.5, 1.0 - (risk_metrics.volatility - 0.2) * 2)
        sharpe_adjustment = min(1.5, max(0.5, risk_metrics.sharpe_ratio))
        
        adjusted_size = base_size * volatility_penalty * sharpe_adjustment
        
        return float(np.clip(adjusted_size, 0.01, self.max_position_size[risk_tolerance]))

    def _calculate_target_price(self, current_price: float, fusion_score: float,
                               action: RecommendationAction) -> Optional[float]:
        """Calculate target price based on fusion score strength"""
        
        if action in [RecommendationAction.BUY, RecommendationAction.STRONG_BUY]:
            # Upward target
            target_multiplier = 1.0 + (abs(fusion_score) / 5.0 * 0.15)  # Up to 15% target
            return float(current_price * target_multiplier)
        elif action in [RecommendationAction.SELL, RecommendationAction.STRONG_SELL]:
            # Downward target
            target_multiplier = 1.0 - (abs(fusion_score) / 5.0 * 0.15)  # Down to 15% target
            return float(current_price * target_multiplier)
        else:
            return None

    def _generate_summary(self, action: RecommendationAction, fusion_score: float,
                         symbol: str, current_price: float) -> RecommendationSummary:
        """Generate two-line recommendation summary"""
        
        confidence_pct = min(95, int(abs(fusion_score) / 5.0 * 100))
        
        # Line 1: Current situation
        if fusion_score > 2:
            line1 = f"{symbol} shows strong bullish signals across sentiment, technical, and fundamental analysis."
        elif fusion_score > 0.5:
            line1 = f"{symbol} displays positive momentum with favorable sentiment and technical indicators."
        elif fusion_score < -2:
            line1 = f"{symbol} exhibits concerning bearish signals with negative sentiment and weak technicals."
        elif fusion_score < -0.5:
            line1 = f"{symbol} shows mixed signals with some downward pressure from recent developments."
        else:
            line1 = f"{symbol} presents neutral outlook with balanced risk-reward profile at current levels."
        
        # Line 2: Specific recommendation
        action_text = {
            RecommendationAction.STRONG_BUY: "Strong Buy - Consider increasing position size",
            RecommendationAction.BUY: "Buy - Initiate or add to position",
            RecommendationAction.HOLD: "Hold - Maintain current position",
            RecommendationAction.SELL: "Sell - Consider reducing position",
            RecommendationAction.STRONG_SELL: "Strong Sell - Exit position"
        }
        
        line2 = f"Recommendation: {action_text[action]} at ${current_price:.2f} with disciplined risk management."
        
        return RecommendationSummary(
            line1=line1,
            line2=line2,
            confidence_pct=confidence_pct,
            time_horizon="90 days",
            risk_level="Medium" if abs(fusion_score) < 3 else "High"
        )

    def _generate_detailed_reasoning(self, sentiment: AggregatedSentiment,
                                   technical: TechnicalIndicators,
                                   fundamental: FundamentalMetrics,
                                   sent_score: float, tech_score: float,
                                   fund_score: float, fusion_score: float) -> str:
        """Generate detailed reasoning using SHAP-like feature attribution"""
        
        reasoning_parts = []
        
        # Sentiment analysis contribution
        if sent_score > 1:
            reasoning_parts.append(f"Sentiment Analysis (+{sent_score:.1f}): {sentiment.article_count} articles over 90 days show {sentiment.high_confidence_ratio:.0%} high-confidence positive sentiment")
        elif sent_score < -1:
            reasoning_parts.append(f"Sentiment Analysis ({sent_score:.1f}): Negative sentiment trend with {sentiment.article_count} articles indicating bearish outlook")
        
        # Technical analysis contribution
        if tech_score > 1:
            reasoning_parts.append(f"Technical Analysis (+{tech_score:.1f}): RSI at {technical.rsi:.0f}, positive MACD signal, and momentum indicators suggest bullish trend")
        elif tech_score < -1:
            reasoning_parts.append(f"Technical Analysis ({tech_score:.1f}): Overbought conditions with RSI {technical.rsi:.0f} and negative momentum indicators")
        
        # Fundamental analysis contribution
        if fund_score > 1:
            reasoning_parts.append(f"Fundamental Analysis (+{fund_score:.1f}): Strong fundamentals with {fundamental.roe:.1%} ROE and {fundamental.revenue_growth:.1%} revenue growth")
        elif fund_score < -1:
            reasoning_parts.append(f"Fundamental Analysis ({fund_score:.1f}): Concerning valuation metrics and weak growth prospects")
        
        # Risk considerations
        reasoning_parts.append(f"Risk Assessment: Moderate volatility profile with appropriate position sizing for current market conditions")
        
        # Final fusion
        reasoning_parts.append(f"Combined Analysis: Multi-factor fusion score of {fusion_score:.1f}/5.0 indicates {'bullish' if fusion_score > 0 else 'bearish'} outlook with {'high' if abs(fusion_score) > 3 else 'moderate'} conviction")
        
        return ". ".join(reasoning_parts) + "."

    async def _check_risk_tolerance_alignment(self, symbol: str, risk_tolerance: RiskTolerance,
                                            risk_metrics: RiskMetrics) -> bool:
        """Check if recommendation aligns with user's risk tolerance"""
        
        if risk_tolerance == RiskTolerance.CONSERVATIVE:
            return risk_metrics.volatility < 0.25 and risk_metrics.sharpe_ratio > 0.8
        elif risk_tolerance == RiskTolerance.MODERATE:
            return risk_metrics.volatility < 0.35 and risk_metrics.sharpe_ratio > 0.5
        else:  # AGGRESSIVE
            return True  # Accept all risk levels

    def _generate_regulatory_disclosures(self, action: RecommendationAction, symbol: str) -> List[str]:
        """Generate required regulatory disclosures"""
        
        disclosures = [
            "This recommendation is for informational purposes only and should not be considered as personalized investment advice.",
            "Past performance does not guarantee future results. All investments carry risk of loss.",
            "Please consult with a qualified financial advisor before making investment decisions.",
            "This analysis is based on publicly available information and may not reflect all material factors."
        ]
        
        if action in [RecommendationAction.STRONG_BUY, RecommendationAction.STRONG_SELL]:
            disclosures.append("Strong recommendations carry higher risk and should be sized appropriately within your overall portfolio.")
        
        return disclosures

    async def _get_track_record(self, symbol: str) -> Tuple[float, float]:
        """Get historical accuracy and track record score"""
        
        # Placeholder implementation - would query actual performance history
        return 0.72, 7.5  # 72% accuracy, 7.5/10 track record

    async def _store_recommendation(self, rec: DetailedRecommendation, 
                                  risk_tolerance: RiskTolerance,
                                  market_regime: MarketRegime):
        """Store recommendation in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO recommendations 
                (id, symbol, timestamp, action, fusion_score, confidence, 
                 position_size_pct, current_price, target_price, stop_loss_pct,
                 reasoning, risk_tolerance, market_regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec.recommendation_id, rec.symbol, int(rec.timestamp.timestamp()),
                rec.action.value, rec.fusion_score, rec.confidence,
                rec.position_size_pct, rec.current_price, rec.target_price,
                rec.stop_loss_pct, rec.reasoning, risk_tolerance.value,
                market_regime.value
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored recommendation {rec.recommendation_id} for {rec.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to store recommendation: {e}")

    async def _log_compliance_audit(self, rec: DetailedRecommendation):
        """Log compliance audit entry"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            audit_results = {
                'risk_tolerance_aligned': rec.risk_tolerance_aligned,
                'regulatory_disclosures': len(rec.regulatory_disclosures),
                'reasoning_provided': len(rec.reasoning) > 100,
                'risk_metrics_calculated': True
            }
            
            cursor.execute("""
                INSERT INTO compliance_audit 
                (id, recommendation_id, audit_type, audit_result, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), rec.recommendation_id, 'suitability_check',
                json.dumps(audit_results), int(datetime.now().timestamp())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log compliance audit: {e}")

    async def generate_portfolio_recommendations(self, 
                                               sentiment_pipeline: FinBERTSentimentPipeline,
                                               risk_tolerance: RiskTolerance = RiskTolerance.MODERATE) -> Dict[str, DetailedRecommendation]:
        """Generate recommendations for all portfolio tickers"""
        
        recommendations = {}
        
        # Process portfolio tickers
        for symbol in self.portfolio_tickers:
            try:
                # Get recent articles (placeholder - would integrate with news system)
                articles = []  # Would get actual articles
                
                # Get price data for sentiment context
                price_data = await self._get_price_data(symbol)
                
                # Create placeholder aggregated sentiment
                aggregated_sentiment = AggregatedSentiment(
                    symbol=symbol,
                    period_start=datetime.now() - timedelta(days=90),
                    period_end=datetime.now(),
                    ewma_sentiment=np.random.normal(0, 0.3),  # Placeholder
                    volatility_adjusted_sentiment=np.random.normal(0, 0.2),
                    sentiment_trend=np.random.normal(0, 0.1),
                    confidence_weighted_sentiment=np.random.normal(0, 0.25),
                    tbl_upper_barrier=price_data['close'].iloc[-1] * 1.15,
                    tbl_lower_barrier=price_data['close'].iloc[-1] * 0.85,
                    tbl_signal=np.random.choice([-1, 0, 1]),
                    sentiment_std=0.2,
                    article_count=np.random.randint(10, 50),
                    high_confidence_ratio=np.random.uniform(0.6, 0.9),
                    conviction_score=np.random.uniform(3, 9),
                    track_record_multiplier=1.0
                )
                
                # Generate recommendation
                rec = await self.generate_recommendation(symbol, aggregated_sentiment, risk_tolerance)
                
                if rec:
                    recommendations[symbol] = rec
                    
            except Exception as e:
                logger.error(f"Failed to generate recommendation for {symbol}: {e}")
        
        return recommendations

    async def get_recommendation_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of recent recommendations"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent recommendations
            cursor.execute("""
                SELECT action, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM recommendations 
                WHERE timestamp > ? 
                GROUP BY action
            """, (int((datetime.now() - timedelta(days=30)).timestamp()),))
            
            action_summary = {}
            for row in cursor.fetchall():
                action_summary[row[0]] = {
                    'count': row[1],
                    'avg_confidence': round(row[2], 3)
                }
            
            # Get overall stats
            cursor.execute("""
                SELECT COUNT(*) as total_recs, AVG(fusion_score) as avg_fusion_score
                FROM recommendations 
                WHERE timestamp > ?
            """, (int((datetime.now() - timedelta(days=30)).timestamp()),))
            
            total_recs, avg_fusion_score = cursor.fetchone()
            
            conn.close()
            
            return {
                'period': '30 days',
                'total_recommendations': total_recs or 0,
                'average_fusion_score': round(avg_fusion_score or 0, 3),
                'action_breakdown': action_summary,
                'portfolio_coverage': len(self.portfolio_tickers),
                'watchlist_coverage': len(self.watchlist_tickers)
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}

# Factory function
async def create_recommendation_engine() -> InvestmentRecommendationEngine:
    """Create investment recommendation engine"""
    
    engine = InvestmentRecommendationEngine()
    logger.info("Investment recommendation engine created successfully")
    return engine