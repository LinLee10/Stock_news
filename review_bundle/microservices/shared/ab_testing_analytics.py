"""
A/B Testing Analytics and Statistical Analysis
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import redis
import numpy as np
import scipy.stats as stats
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class StatisticalSignificance(Enum):
    """Statistical significance levels"""
    NOT_SIGNIFICANT = "not_significant"
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # p < 0.1
    SIGNIFICANT = "significant"  # p < 0.05
    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01


@dataclass
class ABTestMetrics:
    """A/B test metrics for a single variation"""
    variation_name: str
    unique_users: int
    conversions: float
    conversion_rate: float
    confidence_interval: Tuple[float, float]
    total_events: int
    avg_session_duration: float
    bounce_rate: float


@dataclass
class ABTestResults:
    """Complete A/B test results with statistical analysis"""
    flag_name: str
    test_duration_days: int
    variations: List[ABTestMetrics]
    winner: Optional[str]
    confidence_level: float
    p_value: float
    statistical_significance: StatisticalSignificance
    effect_size: float
    minimum_detectable_effect: float
    sample_size_recommendation: int
    test_power: float
    recommendations: List[str]


class ABTestingAnalytics:
    """A/B Testing Analytics Engine"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = structlog.get_logger(__name__)
        
        # Statistical parameters
        self.default_confidence_level = 0.95
        self.default_power = 0.8
        self.minimum_sample_size = 100
        self.minimum_test_duration = 7  # days
    
    async def analyze_experiment(self, flag_name: str, days: int = 30) -> ABTestResults:
        """Perform comprehensive A/B test analysis"""
        try:
            # Collect raw data
            raw_data = await self._collect_experiment_data(flag_name, days)
            
            if not raw_data or len(raw_data) < 2:
                return self._create_insufficient_data_result(flag_name, days)
            
            # Calculate metrics for each variation
            variation_metrics = []
            for variation_name, data in raw_data.items():
                metrics = await self._calculate_variation_metrics(variation_name, data)
                variation_metrics.append(metrics)
            
            # Perform statistical tests
            statistical_results = await self._perform_statistical_tests(variation_metrics)
            
            # Determine winner and recommendations
            winner = await self._determine_winner(variation_metrics, statistical_results)
            recommendations = await self._generate_recommendations(variation_metrics, statistical_results)
            
            return ABTestResults(
                flag_name=flag_name,
                test_duration_days=days,
                variations=variation_metrics,
                winner=winner,
                confidence_level=statistical_results['confidence_level'],
                p_value=statistical_results['p_value'],
                statistical_significance=statistical_results['significance'],
                effect_size=statistical_results['effect_size'],
                minimum_detectable_effect=statistical_results['mde'],
                sample_size_recommendation=statistical_results['recommended_sample_size'],
                test_power=statistical_results['power'],
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error("Error analyzing experiment", flag_name=flag_name, error=str(e))
            return self._create_error_result(flag_name, days)
    
    async def _collect_experiment_data(self, flag_name: str, days: int) -> Dict[str, Dict[str, Any]]:
        """Collect raw experiment data from Redis"""
        variations_data = {}
        
        for day_offset in range(days):
            date = (datetime.utcnow() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
            event_key = f"ab_test_events:{flag_name}:{date}"
            
            events = await self.redis.lrange(event_key, 0, -1)
            
            for event_data in events:
                try:
                    event = json.loads(event_data.decode('utf-8'))
                    variation = event['variation']
                    
                    if variation not in variations_data:
                        variations_data[variation] = {
                            'users': set(),
                            'conversions': [],
                            'events': [],
                            'session_durations': [],
                            'user_segments': {}
                        }
                    
                    user_id = event['user_id']
                    variations_data[variation]['users'].add(user_id)
                    variations_data[variation]['events'].append(event)
                    
                    # Track conversions
                    if event['event'] == 'conversion':
                        variations_data[variation]['conversions'].append(event.get('value', 1.0))
                    
                    # Track session durations
                    if event['event'] == 'feature_duration':
                        variations_data[variation]['session_durations'].append(event.get('value', 0))
                    
                    # Track user segments
                    segment = event.get('user_segment', 'unknown')
                    if segment not in variations_data[variation]['user_segments']:
                        variations_data[variation]['user_segments'][segment] = 0
                    variations_data[variation]['user_segments'][segment] += 1
                
                except json.JSONDecodeError:
                    continue
        
        return variations_data
    
    async def _calculate_variation_metrics(self, variation_name: str, data: Dict[str, Any]) -> ABTestMetrics:
        """Calculate metrics for a single variation"""
        unique_users = len(data['users'])
        total_conversions = sum(data['conversions'])
        conversion_rate = total_conversions / unique_users if unique_users > 0 else 0
        
        # Calculate confidence interval for conversion rate
        confidence_interval = self._calculate_confidence_interval(
            conversion_rate, unique_users, self.default_confidence_level
        )
        
        # Calculate session metrics
        avg_session_duration = np.mean(data['session_durations']) if data['session_durations'] else 0
        
        # Calculate bounce rate (users with only one event)
        user_event_counts = {}
        for event in data['events']:
            user_id = event['user_id']
            user_event_counts[user_id] = user_event_counts.get(user_id, 0) + 1
        
        single_event_users = sum(1 for count in user_event_counts.values() if count == 1)
        bounce_rate = single_event_users / unique_users if unique_users > 0 else 0
        
        return ABTestMetrics(
            variation_name=variation_name,
            unique_users=unique_users,
            conversions=total_conversions,
            conversion_rate=conversion_rate,
            confidence_interval=confidence_interval,
            total_events=len(data['events']),
            avg_session_duration=avg_session_duration,
            bounce_rate=bounce_rate
        )
    
    async def _perform_statistical_tests(self, variations: List[ABTestMetrics]) -> Dict[str, Any]:
        """Perform statistical tests on A/B test variations"""
        if len(variations) < 2:
            return self._default_statistical_results()
        
        # Find control and treatment groups
        control = variations[0]  # Assume first is control
        treatment = variations[1] if len(variations) > 1 else variations[0]
        
        # Two-proportion z-test
        control_successes = int(control.conversions)
        control_trials = control.unique_users
        treatment_successes = int(treatment.conversions)
        treatment_trials = treatment.unique_users
        
        if control_trials == 0 or treatment_trials == 0:
            return self._default_statistical_results()
        
        # Calculate pooled proportion
        pooled_p = (control_successes + treatment_successes) / (control_trials + treatment_trials)
        
        # Calculate standard error
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_trials + 1/treatment_trials))
        
        if se == 0:
            return self._default_statistical_results()
        
        # Calculate z-score and p-value
        z_score = (treatment.conversion_rate - control.conversion_rate) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        
        # Determine statistical significance
        if p_value < 0.01:
            significance = StatisticalSignificance.HIGHLY_SIGNIFICANT
        elif p_value < 0.05:
            significance = StatisticalSignificance.SIGNIFICANT
        elif p_value < 0.1:
            significance = StatisticalSignificance.MARGINALLY_SIGNIFICANT
        else:
            significance = StatisticalSignificance.NOT_SIGNIFICANT
        
        # Calculate effect size (Cohen's h for proportions)
        effect_size = self._cohens_h(control.conversion_rate, treatment.conversion_rate)
        
        # Calculate minimum detectable effect
        mde = self._calculate_mde(control_trials, treatment_trials)
        
        # Calculate statistical power
        power = self._calculate_power(effect_size, control_trials, treatment_trials)
        
        # Recommend sample size for future tests
        recommended_sample_size = self._recommend_sample_size(control.conversion_rate, mde)
        
        return {
            'confidence_level': self.default_confidence_level,
            'p_value': p_value,
            'z_score': z_score,
            'significance': significance,
            'effect_size': effect_size,
            'mde': mde,
            'power': power,
            'recommended_sample_size': recommended_sample_size
        }
    
    def _calculate_confidence_interval(self, p: float, n: int, confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for proportion"""
        if n == 0:
            return (0, 0)
        
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        se = np.sqrt(p * (1 - p) / n)
        margin_error = z * se
        
        lower = max(0, p - margin_error)
        upper = min(1, p + margin_error)
        
        return (lower, upper)
    
    def _cohens_h(self, p1: float, p2: float) -> float:
        """Calculate Cohen's h effect size for proportions"""
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    def _calculate_mde(self, n1: int, n2: int) -> float:
        """Calculate minimum detectable effect"""
        if n1 == 0 or n2 == 0:
            return 0
        
        # Simplified MDE calculation
        z_alpha = stats.norm.ppf(1 - 0.05/2)  # 95% confidence
        z_beta = stats.norm.ppf(self.default_power)  # 80% power
        
        harmonic_mean = 2 / (1/n1 + 1/n2)
        mde = (z_alpha + z_beta) * np.sqrt(2 * 0.1 * 0.9 / harmonic_mean)  # Assuming baseline rate of 10%
        
        return mde
    
    def _calculate_power(self, effect_size: float, n1: int, n2: int) -> float:
        """Calculate statistical power"""
        if n1 == 0 or n2 == 0 or effect_size == 0:
            return 0
        
        # Simplified power calculation
        harmonic_mean = 2 / (1/n1 + 1/n2)
        z_alpha = stats.norm.ppf(1 - 0.05/2)
        z_beta = effect_size * np.sqrt(harmonic_mean / 2) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return max(0, min(1, power))
    
    def _recommend_sample_size(self, baseline_rate: float, mde: float) -> int:
        """Recommend sample size for desired effect size"""
        if baseline_rate == 0 or mde == 0:
            return self.minimum_sample_size
        
        z_alpha = stats.norm.ppf(1 - 0.05/2)
        z_beta = stats.norm.ppf(self.default_power)
        
        # Sample size calculation for two proportions
        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_avg = (p1 + p2) / 2
        
        n = 2 * (z_alpha + z_beta)**2 * p_avg * (1 - p_avg) / (p2 - p1)**2
        
        return max(self.minimum_sample_size, int(n))
    
    async def _determine_winner(self, variations: List[ABTestMetrics], statistical_results: Dict[str, Any]) -> Optional[str]:
        """Determine the winning variation"""
        if len(variations) < 2:
            return None
        
        # Only declare winner if statistically significant
        if statistical_results['significance'] in [StatisticalSignificance.SIGNIFICANT, 
                                                 StatisticalSignificance.HIGHLY_SIGNIFICANT]:
            # Return variation with highest conversion rate
            best_variation = max(variations, key=lambda v: v.conversion_rate)
            return best_variation.variation_name
        
        return None
    
    async def _generate_recommendations(self, variations: List[ABTestMetrics], 
                                      statistical_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # Sample size recommendations
        min_users = min(v.unique_users for v in variations)
        if min_users < self.minimum_sample_size:
            recommendations.append(
                f"Increase sample size: Current minimum is {min_users} users, "
                f"recommended minimum is {statistical_results['recommended_sample_size']} users"
            )
        
        # Test duration recommendations
        if statistical_results['significance'] == StatisticalSignificance.NOT_SIGNIFICANT:
            if statistical_results['power'] < 0.8:
                recommendations.append(
                    f"Low statistical power ({statistical_results['power']:.2f}). "
                    "Consider running test longer or increasing sample size"
                )
        
        # Effect size recommendations
        if abs(statistical_results['effect_size']) < 0.2:
            recommendations.append("Small effect size detected. Consider testing larger changes")
        elif abs(statistical_results['effect_size']) > 0.8:
            recommendations.append("Large effect size detected. Results may be practically significant")
        
        # Conversion rate recommendations
        if len(variations) >= 2:
            control = variations[0]
            treatment = variations[1]
            relative_lift = ((treatment.conversion_rate - control.conversion_rate) / 
                           control.conversion_rate * 100) if control.conversion_rate > 0 else 0
            
            if relative_lift > 10:
                recommendations.append(f"Strong positive lift detected ({relative_lift:.1f}%)")
            elif relative_lift < -5:
                recommendations.append(f"Negative impact detected ({relative_lift:.1f}%)")
        
        # Statistical significance recommendations
        if statistical_results['significance'] == StatisticalSignificance.HIGHLY_SIGNIFICANT:
            recommendations.append("Results are highly statistically significant. Safe to implement changes")
        elif statistical_results['significance'] == StatisticalSignificance.SIGNIFICANT:
            recommendations.append("Results are statistically significant. Consider implementing changes")
        else:
            recommendations.append("Results not statistically significant. Continue testing or abandon hypothesis")
        
        return recommendations
    
    def _default_statistical_results(self) -> Dict[str, Any]:
        """Return default statistical results when calculation fails"""
        return {
            'confidence_level': self.default_confidence_level,
            'p_value': 1.0,
            'z_score': 0.0,
            'significance': StatisticalSignificance.NOT_SIGNIFICANT,
            'effect_size': 0.0,
            'mde': 0.05,
            'power': 0.0,
            'recommended_sample_size': self.minimum_sample_size
        }
    
    def _create_insufficient_data_result(self, flag_name: str, days: int) -> ABTestResults:
        """Create result for insufficient data"""
        return ABTestResults(
            flag_name=flag_name,
            test_duration_days=days,
            variations=[],
            winner=None,
            confidence_level=self.default_confidence_level,
            p_value=1.0,
            statistical_significance=StatisticalSignificance.NOT_SIGNIFICANT,
            effect_size=0.0,
            minimum_detectable_effect=0.05,
            sample_size_recommendation=self.minimum_sample_size,
            test_power=0.0,
            recommendations=["Insufficient data for analysis. Collect more data and retry."]
        )
    
    def _create_error_result(self, flag_name: str, days: int) -> ABTestResults:
        """Create result for analysis errors"""
        return ABTestResults(
            flag_name=flag_name,
            test_duration_days=days,
            variations=[],
            winner=None,
            confidence_level=self.default_confidence_level,
            p_value=1.0,
            statistical_significance=StatisticalSignificance.NOT_SIGNIFICANT,
            effect_size=0.0,
            minimum_detectable_effect=0.05,
            sample_size_recommendation=self.minimum_sample_size,
            test_power=0.0,
            recommendations=["Error occurred during analysis. Check logs and retry."]
        )
    
    async def generate_experiment_report(self, results: ABTestResults) -> str:
        """Generate a comprehensive experiment report"""
        report_lines = [
            f"# A/B Test Results: {results.flag_name}",
            f"**Test Duration:** {results.test_duration_days} days",
            f"**Statistical Significance:** {results.statistical_significance.value.replace('_', ' ').title()}",
            f"**P-Value:** {results.p_value:.4f}",
            f"**Effect Size:** {results.effect_size:.4f}",
            f"**Test Power:** {results.test_power:.2f}",
            "",
            "## Variation Results:",
        ]
        
        for variation in results.variations:
            report_lines.extend([
                f"### {variation.variation_name}",
                f"- **Users:** {variation.unique_users:,}",
                f"- **Conversions:** {variation.conversions:.1f}",
                f"- **Conversion Rate:** {variation.conversion_rate:.2%}",
                f"- **95% CI:** [{variation.confidence_interval[0]:.3f}, {variation.confidence_interval[1]:.3f}]",
                f"- **Avg Session Duration:** {variation.avg_session_duration:.1f}s",
                f"- **Bounce Rate:** {variation.bounce_rate:.2%}",
                ""
            ])
        
        if results.winner:
            report_lines.extend([
                f"## Winner: {results.winner}",
                ""
            ])
        
        if results.recommendations:
            report_lines.extend([
                "## Recommendations:",
                *[f"- {rec}" for rec in results.recommendations],
                ""
            ])
        
        return "\n".join(report_lines)