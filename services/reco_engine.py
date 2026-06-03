#!/usr/bin/env python3
"""
Recommendation engine module for generating buy/hold/reduce/exit signals
Uses technical indicators, momentum, news consensus, and portfolio context
"""
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from services.audit_logger import log_recommendation_generation

logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicators:
    """Technical indicator data structure"""
    price_current: float
    dma_20: float
    dma_50: float 
    dma_200: float
    rsi: float
    macd_histogram: float
    volume_avg: Optional[float] = None
    volatility: Optional[float] = None

@dataclass
class MomentumData:
    """Momentum vs sector data structure"""
    momentum_1m: float  # 1-month momentum vs sector
    momentum_3m: float  # 3-month momentum vs sector
    momentum_6m: float  # 6-month momentum vs sector

@dataclass
class NewsConsensus:
    """News consensus data structure"""
    consensus_14d: float  # 14-day consensus score
    trend_direction: str  # "rising", "falling", "stable"
    confidence: float     # Confidence in consensus

@dataclass
class EarningsContext:
    """Earnings window context"""
    next_earnings_date: Optional[datetime]
    hours_until_earnings: Optional[float]
    confirmed: bool = False

@dataclass
class PortfolioContext:
    """Portfolio position context"""
    quantity: float
    cost_basis: float
    current_value: float
    position_pct: float  # Percentage of total portfolio
    unrealized_pnl_pct: float  # P/L percentage

@dataclass
class Recommendation:
    """Recommendation output structure"""
    symbol: str
    action: str          # "buy", "hold", "reduce", "exit"
    confidence: float    # 0.0 to 1.0
    why: str            # One sentence explanation
    horizon_days: int   # Recommendation time horizon
    next_check_date: datetime

class RecommendationEngine:
    """Core recommendation engine with configurable rules"""
    
    def __init__(self, rules_path: Optional[str] = None):
        self.rules_path = rules_path or "config/recommendation_rules.yaml"
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict[str, Any]:
        """Load recommendation rules from YAML configuration"""
        try:
            rules_file = Path(self.rules_path)
            if not rules_file.exists():
                logger.error(f"Rules file not found: {self.rules_path}")
                return self._get_default_rules()
            
            with open(rules_file, 'r') as f:
                rules = yaml.safe_load(f)
            
            logger.info(f"Loaded recommendation rules from {self.rules_path}")
            return rules
            
        except Exception as e:
            logger.exception(f"Error loading rules file: {e}")
            return self._get_default_rules()
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Fallback default rules if YAML file is unavailable"""
        return {
            'technical': {'trend': {'bullish_threshold': 0.02, 'bearish_threshold': -0.02}},
            'news': {'positive_threshold': 0.2, 'negative_threshold': -0.2},
            'earnings': {'high_risk': 24, 'medium_risk': 48, 'safe_window': 72},
            'rules': {
                'hold': {'conditions': [], 'confidence_base': 0.5, 'horizon_days': 14}
            }
        }
    
    def generate_recommendation(self,
                              symbol: str,
                              technical: TechnicalIndicators,
                              momentum: MomentumData,
                              news: NewsConsensus,
                              earnings: EarningsContext,
                              portfolio: PortfolioContext) -> Recommendation:
        """
        Generate recommendation based on all input factors
        Pure function with no external dependencies
        """
        try:
            # Evaluate each component
            trend_regime = self._evaluate_trend_regime(technical)
            rsi_band = self._evaluate_rsi_band(technical.rsi)
            macd_signal = self._evaluate_macd_signal(technical.macd_histogram)
            momentum_signals = self._evaluate_momentum_signals(momentum)
            news_signal = self._evaluate_news_signal(news)
            earnings_risk = self._evaluate_earnings_risk(earnings)
            portfolio_signal = self._evaluate_portfolio_context(portfolio)
            
            # Create evaluation context
            context = {
                'trend_regime': trend_regime,
                'rsi_band': rsi_band,
                'macd_signal': macd_signal,
                'momentum_1m': momentum_signals['1m'],
                'momentum_3m': momentum_signals['3m'],
                'momentum_6m': momentum_signals['6m'],
                'news_consensus': news.consensus_14d,
                'news_trend': news.trend_direction,
                'earnings_risk': earnings_risk,
                'position_size': portfolio_signal['size'],
                'position_pnl': portfolio.unrealized_pnl_pct
            }
            
            # Evaluate rules in priority order
            recommendation = self._evaluate_rules(symbol, context)
            
            return recommendation
            
        except Exception as e:
            logger.exception(f"Error generating recommendation for {symbol}: {e}")
            # Return safe default
            return Recommendation(
                symbol=symbol,
                action="hold",
                confidence=0.3,
                why="Error in analysis, defaulting to hold",
                horizon_days=7,
                next_check_date=datetime.now() + timedelta(days=7)
            )
    
    def _evaluate_trend_regime(self, technical: TechnicalIndicators) -> str:
        """Evaluate trend regime based on moving averages"""
        current = technical.price_current
        dma_20 = technical.dma_20
        dma_50 = technical.dma_50
        dma_200 = technical.dma_200
        
        bullish_thresh = self.rules['technical']['trend']['bullish_threshold']
        bearish_thresh = self.rules['technical']['trend']['bearish_threshold']
        
        # Check if price is above/below DMAs
        above_20 = (current - dma_20) / dma_20 > bullish_thresh
        above_50 = (current - dma_50) / dma_50 > bullish_thresh
        above_200 = (current - dma_200) / dma_200 > bullish_thresh
        
        below_20 = (current - dma_20) / dma_20 < bearish_thresh
        below_50 = (current - dma_50) / dma_50 < bearish_thresh
        below_200 = (current - dma_200) / dma_200 < bearish_thresh
        
        # Determine trend
        if above_20 and above_50 and above_200:
            return "positive"
        elif below_20 and below_50 and below_200:
            return "negative"
        elif below_20 and below_50:
            return "broken"
        else:
            return "mixed"
    
    def _evaluate_rsi_band(self, rsi: float) -> str:
        """Evaluate RSI band position"""
        oversold = self.rules['technical']['rsi']['oversold']
        overbought = self.rules['technical']['rsi']['overbought']
        neutral_min = self.rules['technical']['rsi']['neutral_min']
        neutral_max = self.rules['technical']['rsi']['neutral_max']
        
        if rsi <= oversold:
            return "oversold"
        elif rsi >= overbought:
            return "overbought"
        elif neutral_min <= rsi <= neutral_max:
            return "neutral"
        elif rsi < neutral_min:
            return "oversold_zone"
        else:
            return "overbought_zone"
    
    def _evaluate_macd_signal(self, macd_histogram: float) -> str:
        """Evaluate MACD histogram signal"""
        pos_thresh = self.rules['technical']['macd']['positive_momentum']
        neg_thresh = self.rules['technical']['macd']['negative_momentum']
        
        if macd_histogram > pos_thresh:
            return "positive"
        elif macd_histogram < neg_thresh:
            return "negative"
        else:
            return "neutral"
    
    def _evaluate_momentum_signals(self, momentum: MomentumData) -> Dict[str, str]:
        """Evaluate momentum signals vs sector"""
        def classify_momentum(value: float) -> str:
            strong_pos = self.rules['momentum']['strong_positive']
            weak_pos = self.rules['momentum']['weak_positive']
            weak_neg = self.rules['momentum']['weak_negative']
            strong_neg = self.rules['momentum']['strong_negative']
            
            if value >= strong_pos:
                return "strong_positive"
            elif value >= weak_pos:
                return "positive"
            elif value >= weak_neg:
                return "neutral"
            elif value >= strong_neg:
                return "negative"
            else:
                return "strong_negative"
        
        return {
            '1m': classify_momentum(momentum.momentum_1m),
            '3m': classify_momentum(momentum.momentum_3m),
            '6m': classify_momentum(momentum.momentum_6m)
        }
    
    def _evaluate_news_signal(self, news: NewsConsensus) -> str:
        """Evaluate news consensus signal"""
        pos_thresh = self.rules['news']['positive_threshold']
        neg_thresh = self.rules['news']['negative_threshold']
        
        if news.consensus_14d >= pos_thresh:
            return "positive"
        elif news.consensus_14d <= neg_thresh:
            return "negative"
        else:
            return "neutral"
    
    def _evaluate_earnings_risk(self, earnings: EarningsContext) -> str:
        """Evaluate earnings risk window"""
        if not earnings.next_earnings_date or not earnings.hours_until_earnings:
            return "safe"
        
        hours = earnings.hours_until_earnings
        high_risk = self.rules['earnings']['high_risk']
        medium_risk = self.rules['earnings']['medium_risk']
        safe_window = self.rules['earnings']['safe_window']
        
        if hours <= high_risk:
            return "high"
        elif hours <= medium_risk:
            return "medium"
        elif hours <= safe_window:
            return "low"
        else:
            return "safe"
    
    def _evaluate_portfolio_context(self, portfolio: PortfolioContext) -> Dict[str, str]:
        """Evaluate portfolio position context"""
        overweight = self.rules['portfolio']['overweight_threshold']
        underweight = self.rules['portfolio']['underweight_threshold']
        profit_target = self.rules['portfolio']['profit_target']
        stop_loss = self.rules['portfolio']['stop_loss']
        
        # Position size classification
        if portfolio.position_pct > overweight:
            size_signal = "overweight"
        elif portfolio.position_pct < underweight:
            size_signal = "underweight"
        else:
            size_signal = "normal"
        
        # P/L classification
        if portfolio.unrealized_pnl_pct >= profit_target:
            pnl_signal = "take_profit"
        elif portfolio.unrealized_pnl_pct <= stop_loss:
            pnl_signal = "stop_loss"
        else:
            pnl_signal = "normal"
        
        return {
            'size': size_signal,
            'pnl': pnl_signal
        }
    
    def _evaluate_rules(self, symbol: str, context: Dict[str, Any]) -> Recommendation:
        """Evaluate rules in priority order and return best match"""
        
        # Define rule priority (most specific first)
        rule_priority = [
            'exit_stop_loss', 'exit', 'strong_buy', 'buy', 
            'reduce_overweight', 'reduce', 'hold_earnings', 'hold'
        ]
        
        for rule_name in rule_priority:
            if rule_name not in self.rules['rules']:
                continue
                
            rule = self.rules['rules'][rule_name]
            if self._matches_conditions(context, rule.get('conditions', [])):
                return self._create_recommendation(symbol, rule_name, rule, context)
        
        # Fallback to hold if no rules match
        return self._create_recommendation(symbol, 'hold', self.rules['rules']['hold'], context)
    
    def _matches_conditions(self, context: Dict[str, Any], conditions: List[str]) -> bool:
        """Check if context matches all rule conditions"""
        for condition in conditions:
            if not self._evaluate_condition(context, condition):
                return False
        return True
    
    def _evaluate_condition(self, context: Dict[str, Any], condition: str) -> bool:
        """Evaluate a single condition against context"""
        try:
            # Parse condition format: "field: operator value"
            parts = condition.split(': ')
            if len(parts) != 2:
                return False
            
            field, expression = parts
            field = field.strip()
            expression = expression.strip()
            
            field_value = context.get(field)
            if field_value is None:
                return False
            
            # Parse operators
            if expression.startswith('>='):
                threshold = float(expression[2:].strip())
                return field_value >= threshold
            elif expression.startswith('<='):
                threshold = float(expression[2:].strip())
                return field_value <= threshold
            elif expression.startswith('>'):
                threshold = float(expression[1:].strip())
                return field_value > threshold
            elif expression.startswith('<'):
                threshold = float(expression[1:].strip())
                return field_value < threshold
            elif expression.startswith('==') or expression.startswith('='):
                target = expression[2:].strip().strip('"\'')
                return str(field_value) == target
            else:
                # Direct string comparison
                target = expression.strip('"\'')
                return str(field_value) == target
                
        except Exception as e:
            logger.warning(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _create_recommendation(self, symbol: str, rule_name: str, rule: Dict[str, Any], context: Dict[str, Any]) -> Recommendation:
        """Create recommendation from matched rule"""
        action = rule_name.split('_')[0]  # Extract action from rule name
        if action not in ['buy', 'hold', 'reduce', 'exit']:
            action = 'hold'
        
        base_confidence = rule.get('confidence_base', 0.5)
        horizon_days = rule.get('horizon_days', 14)
        
        # Generate explanation
        why = self._generate_explanation(rule_name, context)
        
        # Calculate next check date
        next_check = datetime.now() + timedelta(days=min(horizon_days, 30))
        
        return Recommendation(
            symbol=symbol,
            action=action,
            confidence=base_confidence,
            why=why,
            horizon_days=horizon_days,
            next_check_date=next_check
        )
    
    def _generate_explanation(self, rule_name: str, context: Dict[str, Any]) -> str:
        """Generate one-sentence explanation for recommendation"""
        explanations = {
            'buy': f"Positive trend with {context.get('news_consensus', 0):.2f} news consensus and safe earnings window",
            'strong_buy': f"Strong momentum and positive sentiment with {context.get('momentum_1m', 'N/A')} 1-month performance",
            'hold': f"Mixed signals with {context.get('trend_regime', 'unknown')} trend regime",
            'hold_earnings': f"Holding due to earnings risk in {context.get('earnings_risk', 'unknown')} window",
            'reduce': f"Negative trend with falling news consensus and overweight position",
            'reduce_overweight': f"Reducing overweight position before earnings uncertainty",
            'exit': f"Broken trend with strongly negative news consensus ({context.get('news_consensus', 0):.2f})",
            'exit_stop_loss': f"Stop loss triggered at {context.get('position_pnl', 0):.1%} loss"
        }
        
        return explanations.get(rule_name, "Rule-based recommendation")

# Convenience functions for integration
def generate_symbol_recommendation(symbol: str,
                                 technical_data: Dict[str, float],
                                 momentum_data: Dict[str, float],
                                 news_data: Dict[str, Any],
                                 earnings_data: Dict[str, Any],
                                 portfolio_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Convenience function to generate recommendation from raw data dictionaries
    """
    with log_recommendation_generation(symbol, {
        'has_position': portfolio_data.get('quantity', 0) > 0,
        'current_price': technical_data.get('price_current'),
        'rsi': technical_data.get('rsi')
    }) as op:
        engine = RecommendationEngine()
        op.step("engine_initialized")
        
        # Convert dictionaries to dataclasses
        technical = TechnicalIndicators(
            price_current=technical_data['price_current'],
            dma_20=technical_data['dma_20'],
            dma_50=technical_data['dma_50'],
            dma_200=technical_data['dma_200'],
            rsi=technical_data['rsi'],
            macd_histogram=technical_data['macd_histogram'],
            volume_avg=technical_data.get('volume_avg'),
            volatility=technical_data.get('volatility')
        )
        
        momentum = MomentumData(
            momentum_1m=momentum_data['momentum_1m'],
            momentum_3m=momentum_data['momentum_3m'],
            momentum_6m=momentum_data['momentum_6m']
        )
        
        news = NewsConsensus(
            consensus_14d=news_data['consensus_14d'],
            trend_direction=news_data['trend_direction'],
            confidence=news_data['confidence']
        )
        
        earnings = EarningsContext(
            next_earnings_date=earnings_data.get('next_earnings_date'),
            hours_until_earnings=earnings_data.get('hours_until_earnings'),
            confirmed=earnings_data.get('confirmed', False)
        )
        
        portfolio = PortfolioContext(
            quantity=portfolio_data['quantity'],
            cost_basis=portfolio_data['cost_basis'],
            current_value=portfolio_data['current_value'],
            position_pct=portfolio_data['position_pct'],
            unrealized_pnl_pct=portfolio_data['unrealized_pnl_pct']
        )
        
        op.step("data_converted", count_in=5, metadata={
            'earnings_proximity': earnings.hours_until_earnings,
            'position_pct': portfolio.position_pct,
            'news_consensus': news.consensus_14d
        })
        
        # Generate recommendation
        recommendation = engine.generate_recommendation(
            symbol, technical, momentum, news, earnings, portfolio
        )
        
        op.step("recommendation_generated", count_out=1, metadata={
            'action': recommendation.action,
            'confidence': recommendation.confidence,
            'horizon_days': recommendation.horizon_days
        })
        
        # Convert to dictionary for JSON serialization
        result = {
            'symbol': recommendation.symbol,
            'action': recommendation.action,
            'confidence': recommendation.confidence,
            'why': recommendation.why,
            'horizon_days': recommendation.horizon_days,
            'next_check_date': recommendation.next_check_date.isoformat()
        }
        
        op.step("result_serialized", count_out=1)
        return result