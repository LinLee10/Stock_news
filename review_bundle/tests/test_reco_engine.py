#!/usr/bin/env python3
"""
Unit tests for recommendation engine
Pure function tests with canned scenarios and no external calls
"""
import unittest
import tempfile
import yaml
from datetime import datetime, timedelta
from pathlib import Path

from services.reco_engine import (
    RecommendationEngine, TechnicalIndicators, MomentumData, 
    NewsConsensus, EarningsContext, PortfolioContext,
    generate_symbol_recommendation
)

class TestRecommendationEngine(unittest.TestCase):
    """Test cases for RecommendationEngine with canned scenarios"""
    
    def setUp(self):
        """Set up test environment with temporary rules file"""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_path = Path(self.temp_dir) / "test_rules.yaml"
        
        # Create test rules file
        test_rules = {
            'technical': {
                'trend': {'bullish_threshold': 0.02, 'bearish_threshold': -0.02},
                'rsi': {'oversold': 30, 'overbought': 70, 'neutral_min': 40, 'neutral_max': 60},
                'macd': {'positive_momentum': 0.1, 'negative_momentum': -0.1}
            },
            'momentum': {
                'strong_positive': 0.15, 'weak_positive': 0.05, 
                'weak_negative': -0.05, 'strong_negative': -0.15
            },
            'news': {
                'positive_threshold': 0.2, 'negative_threshold': -0.2, 'falling_threshold': -0.1
            },
            'earnings': {'high_risk': 24, 'medium_risk': 48, 'safe_window': 72},
            'portfolio': {
                'overweight_threshold': 0.15, 'underweight_threshold': 0.02,
                'profit_target': 0.20, 'stop_loss': -0.10
            },
            'rules': {
                'buy': {
                    'conditions': [
                        'trend_regime: positive',
                        'momentum_1m: positive',
                        'news_consensus: >= 0.2',
                        'earnings_risk: safe'
                    ],
                    'confidence_base': 0.8,
                    'horizon_days': 30
                },
                'strong_buy': {
                    'conditions': [
                        'trend_regime: positive',
                        'momentum_1m: strong_positive',
                        'momentum_3m: positive',
                        'news_consensus: >= 0.3'
                    ],
                    'confidence_base': 0.9,
                    'horizon_days': 45
                },
                'hold': {
                    'conditions': ['trend_regime: mixed'],
                    'confidence_base': 0.6,
                    'horizon_days': 14
                },
                'hold_earnings': {
                    'conditions': [
                        'earnings_risk: high',
                        'news_consensus: >= -0.1'
                    ],
                    'confidence_base': 0.5,
                    'horizon_days': 7
                },
                'reduce': {
                    'conditions': [
                        'trend_regime: negative',
                        'news_consensus: < 0.0'
                    ],
                    'confidence_base': 0.7,
                    'horizon_days': 14
                },
                'exit': {
                    'conditions': [
                        'trend_regime: broken',
                        'news_consensus: <= -0.3'
                    ],
                    'confidence_base': 0.9,
                    'horizon_days': 3
                },
                'exit_stop_loss': {
                    'conditions': ['position_pnl: <= -0.10'],
                    'confidence_base': 0.95,
                    'horizon_days': 1
                }
            }
        }
        
        with open(self.rules_path, 'w') as f:
            yaml.dump(test_rules, f)
        
        self.engine = RecommendationEngine(str(self.rules_path))
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_base_inputs(self):
        """Create base input data for testing"""
        technical = TechnicalIndicators(
            price_current=100.0,
            dma_20=98.0,
            dma_50=97.0,
            dma_200=95.0,
            rsi=50.0,
            macd_histogram=0.15
        )
        
        momentum = MomentumData(
            momentum_1m=0.08,
            momentum_3m=0.12,
            momentum_6m=0.18
        )
        
        news = NewsConsensus(
            consensus_14d=0.25,
            trend_direction="rising",
            confidence=0.8
        )
        
        earnings = EarningsContext(
            next_earnings_date=datetime.now() + timedelta(days=10),
            hours_until_earnings=240.0,
            confirmed=True
        )
        
        portfolio = PortfolioContext(
            quantity=100.0,
            cost_basis=90.0,
            current_value=10000.0,
            position_pct=0.05,
            unrealized_pnl_pct=0.11
        )
        
        return technical, momentum, news, earnings, portfolio
    
    def test_buy_recommendation_scenario(self):
        """Test scenario that should generate BUY recommendation"""
        technical, momentum, news, earnings, portfolio = self._create_base_inputs()
        
        # Adjust for buy conditions: positive trend, positive momentum, positive news, safe earnings
        technical.price_current = 102.0  # Above all DMAs
        momentum.momentum_1m = 0.08      # Positive momentum
        news.consensus_14d = 0.25        # Above 0.2 threshold
        earnings.hours_until_earnings = 240.0  # Safe window (> 72 hours)
        
        recommendation = self.engine.generate_recommendation(
            "AAPL", technical, momentum, news, earnings, portfolio
        )
        
        self.assertEqual(recommendation.action, "buy")
        self.assertEqual(recommendation.confidence, 0.8)
        self.assertEqual(recommendation.horizon_days, 30)
        self.assertIn("Positive trend", recommendation.why)
    
    def test_strong_buy_recommendation_scenario(self):
        """Test scenario that should generate STRONG_BUY recommendation"""
        technical, momentum, news, earnings, portfolio = self._create_base_inputs()
        
        # Adjust for strong buy conditions
        technical.price_current = 103.0  # Strong uptrend
        momentum.momentum_1m = 0.18      # Strong positive momentum
        momentum.momentum_3m = 0.15      # Strong 3m momentum too
        news.consensus_14d = 0.35        # Very positive news
        
        recommendation = self.engine.generate_recommendation(
            "NVDA", technical, momentum, news, earnings, portfolio
        )
        
        self.assertEqual(recommendation.action, "strong")  # Should be "strong_buy" -> "strong"
        self.assertEqual(recommendation.confidence, 0.9)
        self.assertEqual(recommendation.horizon_days, 45)
    
    def test_hold_recommendation_scenario(self):
        """Test scenario that should generate HOLD recommendation"""
        technical, momentum, news, earnings, portfolio = self._create_base_inputs()
        
        # Mixed trend scenario
        technical.price_current = 98.5   # Between some DMAs
        technical.dma_20 = 99.0
        technical.dma_50 = 98.0
        technical.dma_200 = 97.0
        
        recommendation = self.engine.generate_recommendation(
            "MSFT", technical, momentum, news, earnings, portfolio
        )
        
        self.assertEqual(recommendation.action, "hold")
        self.assertEqual(recommendation.confidence, 0.6)
        self.assertEqual(recommendation.horizon_days, 14)
    
    def test_hold_earnings_scenario(self):
        """Test HOLD due to earnings risk"""
        technical, momentum, news, earnings, portfolio = self._create_base_inputs()
        
        # Close to earnings
        earnings.next_earnings_date = datetime.now() + timedelta(hours=20)
        earnings.hours_until_earnings = 20.0  # High risk window
        news.consensus_14d = 0.1  # Not negative
        
        recommendation = self.engine.generate_recommendation(
            "TSLA", technical, momentum, news, earnings, portfolio
        )
        
        self.assertEqual(recommendation.action, "hold")
        self.assertEqual(recommendation.confidence, 0.5)
        self.assertEqual(recommendation.horizon_days, 7)
    
    def test_reduce_recommendation_scenario(self):
        """Test scenario that should generate REDUCE recommendation"""
        technical, momentum, news, earnings, portfolio = self._create_base_inputs()
        
        # Negative trend with poor news
        technical.price_current = 93.0   # Below all DMAs
        technical.dma_20 = 95.0
        technical.dma_50 = 96.0
        technical.dma_200 = 97.0
        news.consensus_14d = -0.15       # Negative news
        
        recommendation = self.engine.generate_recommendation(
            "META", technical, momentum, news, earnings, portfolio
        )
        
        self.assertEqual(recommendation.action, "reduce")
        self.assertEqual(recommendation.confidence, 0.7)
        self.assertEqual(recommendation.horizon_days, 14)
    
    def test_exit_recommendation_scenario(self):
        """Test scenario that should generate EXIT recommendation"""
        technical, momentum, news, earnings, portfolio = self._create_base_inputs()
        
        # Broken trend with very negative news
        technical.price_current = 92.0   # Well below 20/50 DMAs
        technical.dma_20 = 95.0
        technical.dma_50 = 96.0
        news.consensus_14d = -0.35       # Strongly negative news
        
        recommendation = self.engine.generate_recommendation(
            "NFLX", technical, momentum, news, earnings, portfolio
        )
        
        self.assertEqual(recommendation.action, "exit")
        self.assertEqual(recommendation.confidence, 0.9)
        self.assertEqual(recommendation.horizon_days, 3)
    
    def test_exit_stop_loss_scenario(self):
        """Test EXIT due to stop loss trigger"""
        technical, momentum, news, earnings, portfolio = self._create_base_inputs()
        
        # Position with significant loss
        portfolio.cost_basis = 100.0
        portfolio.current_value = 8500.0  # 15% loss
        portfolio.unrealized_pnl_pct = -0.15  # Below stop loss threshold
        
        recommendation = self.engine.generate_recommendation(
            "AMD", technical, momentum, news, earnings, portfolio
        )
        
        self.assertEqual(recommendation.action, "exit")
        self.assertEqual(recommendation.confidence, 0.95)
        self.assertEqual(recommendation.horizon_days, 1)
        self.assertIn("Stop loss", recommendation.why)
    
    def test_trend_regime_evaluation(self):
        """Test trend regime evaluation logic"""
        technical = TechnicalIndicators(
            price_current=100.0, dma_20=97.0, dma_50=95.0, dma_200=92.0,
            rsi=50.0, macd_histogram=0.1
        )
        
        # All DMAs below current price by >2%
        trend = self.engine._evaluate_trend_regime(technical)
        self.assertEqual(trend, "positive")
        
        # Price below all DMAs
        technical.price_current = 90.0
        trend = self.engine._evaluate_trend_regime(technical)
        self.assertEqual(trend, "negative")
        
        # Mixed scenario
        technical.price_current = 96.0  # Between 20 and 50 DMA
        trend = self.engine._evaluate_trend_regime(technical)
        self.assertEqual(trend, "mixed")
    
    def test_rsi_band_evaluation(self):
        """Test RSI band classification"""
        # Oversold
        band = self.engine._evaluate_rsi_band(25.0)
        self.assertEqual(band, "oversold")
        
        # Overbought
        band = self.engine._evaluate_rsi_band(75.0)
        self.assertEqual(band, "overbought")
        
        # Neutral
        band = self.engine._evaluate_rsi_band(50.0)
        self.assertEqual(band, "neutral")
    
    def test_momentum_classification(self):
        """Test momentum signal classification"""
        momentum = MomentumData(momentum_1m=0.18, momentum_3m=0.08, momentum_6m=-0.02)
        
        signals = self.engine._evaluate_momentum_signals(momentum)
        
        self.assertEqual(signals['1m'], "strong_positive")  # >15%
        self.assertEqual(signals['3m'], "positive")         # 5-15%
        self.assertEqual(signals['6m'], "neutral")          # -5% to 5%
    
    def test_earnings_risk_evaluation(self):
        """Test earnings risk window classification"""
        # High risk (within 24 hours)
        earnings = EarningsContext(
            next_earnings_date=datetime.now() + timedelta(hours=20),
            hours_until_earnings=20.0
        )
        risk = self.engine._evaluate_earnings_risk(earnings)
        self.assertEqual(risk, "high")
        
        # Medium risk (24-48 hours)
        earnings.hours_until_earnings = 36.0
        risk = self.engine._evaluate_earnings_risk(earnings)
        self.assertEqual(risk, "medium")
        
        # Safe (>72 hours)
        earnings.hours_until_earnings = 100.0
        risk = self.engine._evaluate_earnings_risk(earnings)
        self.assertEqual(risk, "safe")
    
    def test_portfolio_context_evaluation(self):
        """Test portfolio position context evaluation"""
        # Overweight position
        portfolio = PortfolioContext(
            quantity=1000, cost_basis=50.0, current_value=60000.0,
            position_pct=0.20, unrealized_pnl_pct=0.20
        )
        
        context = self.engine._evaluate_portfolio_context(portfolio)
        self.assertEqual(context['size'], "overweight")
        self.assertEqual(context['pnl'], "take_profit")
        
        # Underweight position with loss
        portfolio.position_pct = 0.01
        portfolio.unrealized_pnl_pct = -0.12
        
        context = self.engine._evaluate_portfolio_context(portfolio)
        self.assertEqual(context['size'], "underweight")
        self.assertEqual(context['pnl'], "stop_loss")
    
    def test_condition_evaluation(self):
        """Test individual condition evaluation logic"""
        context = {
            'trend_regime': 'positive',
            'news_consensus': 0.25,
            'position_pnl': -0.05
        }
        
        # String equality
        self.assertTrue(self.engine._evaluate_condition(context, 'trend_regime: positive'))
        self.assertFalse(self.engine._evaluate_condition(context, 'trend_regime: negative'))
        
        # Numeric comparisons
        self.assertTrue(self.engine._evaluate_condition(context, 'news_consensus: >= 0.2'))
        self.assertFalse(self.engine._evaluate_condition(context, 'news_consensus: >= 0.3'))
        self.assertTrue(self.engine._evaluate_condition(context, 'position_pnl: <= -0.04'))
    
    def test_convenience_function(self):
        """Test the convenience function for external integration"""
        technical_data = {
            'price_current': 150.0, 'dma_20': 145.0, 'dma_50': 140.0, 'dma_200': 135.0,
            'rsi': 45.0, 'macd_histogram': 0.2
        }
        
        momentum_data = {
            'momentum_1m': 0.12, 'momentum_3m': 0.08, 'momentum_6m': 0.15
        }
        
        news_data = {
            'consensus_14d': 0.3, 'trend_direction': 'rising', 'confidence': 0.85
        }
        
        earnings_data = {
            'next_earnings_date': datetime.now() + timedelta(days=5),
            'hours_until_earnings': 120.0, 'confirmed': True
        }
        
        portfolio_data = {
            'quantity': 50.0, 'cost_basis': 140.0, 'current_value': 7500.0,
            'position_pct': 0.08, 'unrealized_pnl_pct': 0.07
        }
        
        result = generate_symbol_recommendation(
            "AAPL", technical_data, momentum_data, news_data, earnings_data, portfolio_data
        )
        
        # Should be a dictionary with all required fields
        self.assertIsInstance(result, dict)
        self.assertIn('symbol', result)
        self.assertIn('action', result)
        self.assertIn('confidence', result)
        self.assertIn('why', result)
        self.assertIn('horizon_days', result)
        self.assertIn('next_check_date', result)
        
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertIn(result['action'], ['buy', 'hold', 'reduce', 'exit', 'strong'])
        self.assertIsInstance(result['confidence'], float)
        self.assertGreater(len(result['why']), 10)  # Should have meaningful explanation

class TestRulesParsing(unittest.TestCase):
    """Test YAML rules parsing and validation"""
    
    def test_rules_file_loading(self):
        """Test that rules file loads correctly"""
        # Create temporary rules file
        temp_dir = tempfile.mkdtemp()
        rules_path = Path(temp_dir) / "test_rules.yaml"
        
        test_rules = {
            'technical': {'trend': {'bullish_threshold': 0.03}},
            'rules': {'hold': {'confidence_base': 0.7, 'horizon_days': 10}}
        }
        
        with open(rules_path, 'w') as f:
            yaml.dump(test_rules, f)
        
        engine = RecommendationEngine(str(rules_path))
        
        # Should load custom threshold
        self.assertEqual(engine.rules['technical']['trend']['bullish_threshold'], 0.03)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_fallback_to_defaults(self):
        """Test fallback to default rules when file is missing"""
        engine = RecommendationEngine("nonexistent_file.yaml")
        
        # Should have some default rules loaded
        self.assertIn('technical', engine.rules)
        self.assertIn('rules', engine.rules)

if __name__ == '__main__':
    unittest.main()