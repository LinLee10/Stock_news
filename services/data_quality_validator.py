"""
Data Quality Validation and Verification System

Cross-validates data between sources, detects anomalies, and ensures data integrity.
Implements quality scoring, anomaly detection, and cross-source verification.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics
from multi_source_data_manager import DataSource, DataQuality
import math

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationCategory(Enum):
    COMPLETENESS = "completeness"      # Missing data checks
    ACCURACY = "accuracy"              # Cross-source validation
    CONSISTENCY = "consistency"        # Internal consistency
    TIMELINESS = "timeliness"          # Data freshness
    PLAUSIBILITY = "plausibility"      # Reasonable value ranges
    UNIQUENESS = "uniqueness"          # Duplicate detection

@dataclass
class ValidationIssue:
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationResult:
    ticker: str
    data_type: str
    source: DataSource
    quality_score: float  # 0-100
    issues: List[ValidationIssue] = field(default_factory=list)
    data_points: int = 0
    validation_timestamp: datetime = field(default_factory=datetime.now)
    cross_validated: bool = False
    reference_sources: List[DataSource] = field(default_factory=list)

@dataclass
class QualityMetrics:
    completeness_score: float = 0.0      # % of expected fields present
    accuracy_score: float = 0.0          # Cross-source agreement %
    consistency_score: float = 0.0       # Internal consistency %
    timeliness_score: float = 0.0        # Data freshness score
    plausibility_score: float = 0.0      # Reasonable values %
    uniqueness_score: float = 0.0        # Duplicate detection %

class DataQualityValidator:
    """Comprehensive data quality validation system"""
    
    def __init__(self):
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 90.0,
            'good': 75.0,
            'fair': 60.0,
            'poor': 40.0
        }
        
        # Expected fields for different data types
        self.expected_fields = {
            'daily': ['open', 'high', 'low', 'close', 'volume'],
            'intraday': ['open', 'high', 'low', 'close', 'volume', 'timestamp'],
            'overview': ['symbol', 'name', 'sector', 'market_cap'],
            'news': ['title', 'content', 'published_at', 'source']
        }
        
        # Plausibility ranges (relative to previous close)
        self.plausibility_ranges = {
            'daily_change_limit': 0.50,    # 50% max daily change
            'volume_spike_limit': 10.0,    # 10x volume spike limit  
            'price_gap_limit': 0.20,       # 20% max gap limit
            'min_price': 0.01,             # Minimum stock price
            'max_price': 10000.0           # Maximum stock price
        }
        
        # Cross-validation tolerances
        self.cross_validation_tolerance = {
            'price': 0.02,      # 2% price difference tolerance
            'volume': 0.10,     # 10% volume difference tolerance
            'percentage': 0.05   # 5% percentage difference tolerance
        }

    async def validate_data(self, ticker: str, data_type: str, data: Dict[str, Any], 
                          source: DataSource, reference_data: Optional[Dict[DataSource, Dict[str, Any]]] = None) -> ValidationResult:
        """Comprehensive data validation"""
        
        result = ValidationResult(
            ticker=ticker,
            data_type=data_type,
            source=source,
            quality_score=0.0
        )
        
        try:
            # Calculate individual quality metrics
            metrics = QualityMetrics()
            
            # 1. Completeness validation
            metrics.completeness_score = await self._validate_completeness(data, data_type, result)
            
            # 2. Consistency validation
            metrics.consistency_score = await self._validate_consistency(data, data_type, result)
            
            # 3. Plausibility validation
            metrics.plausibility_score = await self._validate_plausibility(ticker, data, data_type, result)
            
            # 4. Timeliness validation
            metrics.timeliness_score = await self._validate_timeliness(data, data_type, result)
            
            # 5. Uniqueness validation
            metrics.uniqueness_score = await self._validate_uniqueness(data, data_type, result)
            
            # 6. Cross-source accuracy validation (if reference data available)
            if reference_data:
                metrics.accuracy_score = await self._validate_accuracy(data, reference_data, data_type, result)
                result.cross_validated = True
                result.reference_sources = list(reference_data.keys())
            else:
                metrics.accuracy_score = 80.0  # Assume good accuracy without cross-validation
            
            # Calculate overall quality score (weighted average)
            weights = {
                'completeness': 0.25,
                'accuracy': 0.25,
                'consistency': 0.20,
                'timeliness': 0.15,
                'plausibility': 0.10,
                'uniqueness': 0.05
            }
            
            result.quality_score = (
                metrics.completeness_score * weights['completeness'] +
                metrics.accuracy_score * weights['accuracy'] +
                metrics.consistency_score * weights['consistency'] +
                metrics.timeliness_score * weights['timeliness'] +
                metrics.plausibility_score * weights['plausibility'] +
                metrics.uniqueness_score * weights['uniqueness']
            )
            
            # Count data points
            result.data_points = self._count_data_points(data, data_type)
            
            logger.info(f"Validation complete for {ticker} {data_type}: {result.quality_score:.1f}% quality")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error for {ticker}: {e}")
            result.issues.append(ValidationIssue(
                category=ValidationCategory.ACCURACY,
                severity=ValidationSeverity.ERROR,
                message=f"Validation failed: {e}"
            ))
            result.quality_score = 0.0
            return result

    async def _validate_completeness(self, data: Dict[str, Any], data_type: str, result: ValidationResult) -> float:
        """Validate data completeness"""
        expected_fields = self.expected_fields.get(data_type, [])
        if not expected_fields:
            return 100.0  # No expectations defined
        
        score = 100.0
        present_fields = 0
        
        if data_type == 'daily':
            # Check time series data
            time_series = data.get('Time Series (Daily)', {})
            
            if not time_series:
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=ValidationSeverity.CRITICAL,
                    message="No time series data found"
                ))
                return 0.0
            
            # Check each day's data
            total_expected = len(time_series) * len(expected_fields)
            total_present = 0
            
            for date, day_data in time_series.items():
                for field_num, expected_field in enumerate(expected_fields, 1):
                    field_key = f"{field_num}. {expected_field}"
                    if field_key in day_data and day_data[field_key] not in [None, '', 'N/A']:
                        total_present += 1
                    else:
                        if total_expected - total_present < 10:  # Only log first 10 missing
                            result.issues.append(ValidationIssue(
                                category=ValidationCategory.COMPLETENESS,
                                severity=ValidationSeverity.WARNING,
                                message=f"Missing {expected_field} for {date}",
                                field=f"{date}.{expected_field}"
                            ))
            
            score = (total_present / total_expected) * 100 if total_expected > 0 else 0
            
        elif data_type == 'overview':
            # Check company overview fields
            for field in expected_fields:
                if field.lower() in [k.lower() for k in data.keys()]:
                    value = next((v for k, v in data.items() if k.lower() == field.lower()), None)
                    if value and str(value).lower() not in ['n/a', 'none', '', '0']:
                        present_fields += 1
                else:
                    result.issues.append(ValidationIssue(
                        category=ValidationCategory.COMPLETENESS,
                        severity=ValidationSeverity.WARNING,
                        message=f"Missing expected field: {field}",
                        field=field
                    ))
            
            score = (present_fields / len(expected_fields)) * 100 if expected_fields else 100
        
        return score

    async def _validate_consistency(self, data: Dict[str, Any], data_type: str, result: ValidationResult) -> float:
        """Validate internal data consistency"""
        score = 100.0
        issues_found = 0
        
        if data_type == 'daily':
            time_series = data.get('Time Series (Daily)', {})
            
            for date, day_data in time_series.items():
                try:
                    # Extract OHLCV values
                    open_val = float(day_data.get('1. open', 0))
                    high_val = float(day_data.get('2. high', 0))
                    low_val = float(day_data.get('3. low', 0))
                    close_val = float(day_data.get('4. close', 0))
                    volume_val = float(day_data.get('5. volume', 0))
                    
                    # Consistency checks
                    if high_val < max(open_val, close_val, low_val):
                        result.issues.append(ValidationIssue(
                            category=ValidationCategory.CONSISTENCY,
                            severity=ValidationSeverity.ERROR,
                            message=f"High price ({high_val}) is lower than open/close/low on {date}",
                            field=f"{date}.high"
                        ))
                        issues_found += 1
                    
                    if low_val > min(open_val, close_val, high_val):
                        result.issues.append(ValidationIssue(
                            category=ValidationCategory.CONSISTENCY,
                            severity=ValidationSeverity.ERROR,
                            message=f"Low price ({low_val}) is higher than open/close/high on {date}",
                            field=f"{date}.low"
                        ))
                        issues_found += 1
                    
                    if volume_val < 0:
                        result.issues.append(ValidationIssue(
                            category=ValidationCategory.CONSISTENCY,
                            severity=ValidationSeverity.ERROR,
                            message=f"Negative volume ({volume_val}) on {date}",
                            field=f"{date}.volume"
                        ))
                        issues_found += 1
                    
                    if any(val <= 0 for val in [open_val, high_val, low_val, close_val]):
                        result.issues.append(ValidationIssue(
                            category=ValidationCategory.CONSISTENCY,
                            severity=ValidationSeverity.ERROR,
                            message=f"Non-positive price values on {date}",
                            field=f"{date}.prices"
                        ))
                        issues_found += 1
                        
                except (ValueError, TypeError):
                    result.issues.append(ValidationIssue(
                        category=ValidationCategory.CONSISTENCY,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid numeric values on {date}",
                        field=f"{date}.numeric"
                    ))
                    issues_found += 1
            
            # Calculate score based on error rate
            total_days = len(time_series)
            if total_days > 0:
                error_rate = issues_found / total_days
                score = max(0, 100 - (error_rate * 100))
        
        elif data_type == 'overview':
            # Check overview consistency
            try:
                market_cap = data.get('MarketCapitalization', '0')
                shares_outstanding = data.get('SharesOutstanding', '0')
                
                if market_cap and shares_outstanding:
                    market_cap_val = float(market_cap)
                    shares_val = float(shares_outstanding)
                    
                    if market_cap_val > 0 and shares_val > 0:
                        implied_price = market_cap_val / shares_val
                        
                        # Check if implied price is reasonable (basic sanity check)
                        if implied_price < 0.01 or implied_price > 10000:
                            result.issues.append(ValidationIssue(
                                category=ValidationCategory.CONSISTENCY,
                                severity=ValidationSeverity.WARNING,
                                message=f"Implied stock price ({implied_price:.2f}) seems unrealistic",
                                field="implied_price"
                            ))
                            score -= 10
                            
            except (ValueError, TypeError):
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.CONSISTENCY,
                    severity=ValidationSeverity.WARNING,
                    message="Cannot validate market cap consistency due to invalid values",
                    field="market_cap_validation"
                ))
                score -= 5
        
        return score

    async def _validate_plausibility(self, ticker: str, data: Dict[str, Any], data_type: str, result: ValidationResult) -> float:
        """Validate data plausibility and detect anomalies"""
        score = 100.0
        
        if data_type == 'daily':
            time_series = data.get('Time Series (Daily)', {})
            dates = sorted(time_series.keys())
            
            if len(dates) < 2:
                return score  # Need at least 2 days for comparison
            
            previous_close = None
            daily_changes = []
            volumes = []
            
            for date in dates:
                day_data = time_series[date]
                
                try:
                    open_val = float(day_data.get('1. open', 0))
                    high_val = float(day_data.get('2. high', 0))
                    low_val = float(day_data.get('3. low', 0))
                    close_val = float(day_data.get('4. close', 0))
                    volume_val = float(day_data.get('5. volume', 0))
                    
                    volumes.append(volume_val)
                    
                    # Price range checks
                    if not (self.plausibility_ranges['min_price'] <= close_val <= self.plausibility_ranges['max_price']):
                        result.issues.append(ValidationIssue(
                            category=ValidationCategory.PLAUSIBILITY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Price ({close_val}) outside reasonable range on {date}",
                            field=f"{date}.close"
                        ))
                        score -= 5
                    
                    # Daily change analysis
                    if previous_close:
                        daily_change = abs(close_val - previous_close) / previous_close
                        daily_changes.append(daily_change)
                        
                        if daily_change > self.plausibility_ranges['daily_change_limit']:
                            result.issues.append(ValidationIssue(
                                category=ValidationCategory.PLAUSIBILITY,
                                severity=ValidationSeverity.WARNING,
                                message=f"Large daily change ({daily_change:.1%}) on {date}",
                                field=f"{date}.daily_change",
                                actual_value=daily_change
                            ))
                            score -= 3
                        
                        # Gap analysis
                        gap = abs(open_val - previous_close) / previous_close
                        if gap > self.plausibility_ranges['price_gap_limit']:
                            result.issues.append(ValidationIssue(
                                category=ValidationCategory.PLAUSIBILITY,
                                severity=ValidationSeverity.WARNING,
                                message=f"Large price gap ({gap:.1%}) on {date}",
                                field=f"{date}.gap",
                                actual_value=gap
                            ))
                            score -= 2
                    
                    previous_close = close_val
                    
                except (ValueError, TypeError):
                    score -= 10
                    continue
            
            # Volume anomaly detection
            if len(volumes) > 5:
                avg_volume = statistics.mean(volumes)
                for i, (date, volume) in enumerate(zip(dates, volumes)):
                    if avg_volume > 0 and volume > avg_volume * self.plausibility_ranges['volume_spike_limit']:
                        result.issues.append(ValidationIssue(
                            category=ValidationCategory.PLAUSIBILITY,
                            severity=ValidationSeverity.INFO,
                            message=f"Volume spike ({volume/avg_volume:.1f}x average) on {date}",
                            field=f"{date}.volume_spike"
                        ))
            
            # Statistical anomaly detection for daily changes
            if len(daily_changes) > 10:
                try:
                    mean_change = statistics.mean(daily_changes)
                    std_change = statistics.stdev(daily_changes)
                    
                    for i, change in enumerate(daily_changes):
                        if std_change > 0:
                            z_score = abs(change - mean_change) / std_change
                            if z_score > 3:  # 3-sigma rule
                                date = dates[i + 1]  # +1 because we skip first day
                                result.issues.append(ValidationIssue(
                                    category=ValidationCategory.PLAUSIBILITY,
                                    severity=ValidationSeverity.INFO,
                                    message=f"Statistical outlier in daily change (z-score: {z_score:.1f}) on {date}",
                                    field=f"{date}.statistical_outlier"
                                ))
                                
                except statistics.StatisticsError:
                    pass  # Not enough data for statistical analysis
        
        return max(0, score)

    async def _validate_timeliness(self, data: Dict[str, Any], data_type: str, result: ValidationResult) -> float:
        """Validate data timeliness and freshness"""
        score = 100.0
        now = datetime.now()
        
        if data_type == 'daily':
            # Check last refreshed date
            meta_data = data.get('Meta Data', {})
            last_refreshed = meta_data.get('3. Last Refreshed')
            
            if last_refreshed:
                try:
                    last_date = datetime.strptime(last_refreshed, '%Y-%m-%d')
                    days_old = (now - last_date).days
                    
                    # Penalize old data
                    if days_old > 7:
                        result.issues.append(ValidationIssue(
                            category=ValidationCategory.TIMELINESS,
                            severity=ValidationSeverity.WARNING,
                            message=f"Data is {days_old} days old",
                            field="last_refreshed",
                            actual_value=days_old
                        ))
                        score = max(0, 100 - (days_old - 7) * 5)
                    elif days_old > 3:
                        score = max(80, 100 - days_old * 3)
                        
                except ValueError:
                    result.issues.append(ValidationIssue(
                        category=ValidationCategory.TIMELINESS,
                        severity=ValidationSeverity.WARNING,
                        message="Cannot parse last refreshed date",
                        field="last_refreshed"
                    ))
                    score = 70
            else:
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.TIMELINESS,
                    severity=ValidationSeverity.WARNING,
                    message="No last refreshed date available",
                    field="last_refreshed"
                ))
                score = 80
            
            # Check if we have recent trading days
            time_series = data.get('Time Series (Daily)', {})
            if time_series:
                latest_date_str = max(time_series.keys())
                try:
                    latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
                    days_since_latest = (now - latest_date).days
                    
                    # Account for weekends and potential holidays
                    if days_since_latest > 5:
                        result.issues.append(ValidationIssue(
                            category=ValidationCategory.TIMELINESS,
                            severity=ValidationSeverity.WARNING,
                            message=f"Latest data point is {days_since_latest} days old",
                            field="latest_data_point"
                        ))
                        score = min(score, 100 - days_since_latest * 2)
                        
                except ValueError:
                    score = min(score, 70)
        
        elif data_type == 'overview':
            # Company overview data can be older, but check for basic timeliness
            # Most overview fields should be relatively current
            pass  # Less strict requirements for overview data
        
        return max(0, score)

    async def _validate_uniqueness(self, data: Dict[str, Any], data_type: str, result: ValidationResult) -> float:
        """Validate data uniqueness (detect duplicates)"""
        score = 100.0
        
        if data_type == 'daily':
            time_series = data.get('Time Series (Daily)', {})
            
            # Check for duplicate dates (should not happen but possible with bad data)
            dates = list(time_series.keys())
            unique_dates = set(dates)
            
            if len(dates) != len(unique_dates):
                duplicates = len(dates) - len(unique_dates)
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.UNIQUENESS,
                    severity=ValidationSeverity.ERROR,
                    message=f"Found {duplicates} duplicate dates",
                    field="duplicate_dates",
                    actual_value=duplicates
                ))
                score = max(0, 100 - duplicates * 10)
            
            # Check for duplicate values (potentially indicating data errors)
            values_seen = {}
            for date, day_data in time_series.items():
                try:
                    ohlcv = (
                        day_data.get('1. open'),
                        day_data.get('2. high'),
                        day_data.get('3. low'),
                        day_data.get('4. close'),
                        day_data.get('5. volume')
                    )
                    
                    if ohlcv in values_seen:
                        result.issues.append(ValidationIssue(
                            category=ValidationCategory.UNIQUENESS,
                            severity=ValidationSeverity.WARNING,
                            message=f"Identical OHLCV values on {date} and {values_seen[ohlcv]}",
                            field=f"duplicate_values_{date}"
                        ))
                        score -= 5
                    else:
                        values_seen[ohlcv] = date
                        
                except Exception:
                    continue
        
        return max(0, score)

    async def _validate_accuracy(self, data: Dict[str, Any], reference_data: Dict[DataSource, Dict[str, Any]], 
                               data_type: str, result: ValidationResult) -> float:
        """Cross-validate data accuracy against reference sources"""
        if not reference_data:
            return 80.0  # Default score without reference
        
        score = 100.0
        comparisons_made = 0
        agreements = 0
        
        if data_type == 'daily':
            primary_series = data.get('Time Series (Daily)', {})
            
            for ref_source, ref_data in reference_data.items():
                ref_series = ref_data.get('Time Series (Daily)', {})
                
                # Find common dates
                common_dates = set(primary_series.keys()) & set(ref_series.keys())
                
                for date in list(common_dates)[:10]:  # Limit to 10 dates for performance
                    primary_day = primary_series[date]
                    ref_day = ref_series[date]
                    
                    try:
                        # Compare closing prices
                        primary_close = float(primary_day.get('4. close', 0))
                        ref_close = float(ref_day.get('4. close', 0))
                        
                        if primary_close > 0 and ref_close > 0:
                            price_diff = abs(primary_close - ref_close) / ref_close
                            comparisons_made += 1
                            
                            if price_diff <= self.cross_validation_tolerance['price']:
                                agreements += 1
                            else:
                                result.issues.append(ValidationIssue(
                                    category=ValidationCategory.ACCURACY,
                                    severity=ValidationSeverity.WARNING,
                                    message=f"Price discrepancy on {date}: {primary_close} vs {ref_close} ({ref_source.value})",
                                    field=f"{date}.close_comparison_{ref_source.value}",
                                    expected_value=ref_close,
                                    actual_value=primary_close,
                                    confidence=1.0 - price_diff
                                ))
                        
                        # Compare volumes
                        primary_volume = float(primary_day.get('5. volume', 0))
                        ref_volume = float(ref_day.get('5. volume', 0))
                        
                        if primary_volume > 0 and ref_volume > 0:
                            volume_diff = abs(primary_volume - ref_volume) / ref_volume
                            comparisons_made += 1
                            
                            if volume_diff <= self.cross_validation_tolerance['volume']:
                                agreements += 1
                            else:
                                result.issues.append(ValidationIssue(
                                    category=ValidationCategory.ACCURACY,
                                    severity=ValidationSeverity.INFO,
                                    message=f"Volume discrepancy on {date}: {primary_volume} vs {ref_volume} ({ref_source.value})",
                                    field=f"{date}.volume_comparison_{ref_source.value}",
                                    confidence=1.0 - min(volume_diff, 1.0)
                                ))
                                
                    except (ValueError, TypeError):
                        continue
            
            # Calculate accuracy score
            if comparisons_made > 0:
                agreement_rate = agreements / comparisons_made
                score = agreement_rate * 100
            else:
                score = 50  # No valid comparisons possible
        
        return score

    def _count_data_points(self, data: Dict[str, Any], data_type: str) -> int:
        """Count the number of data points"""
        if data_type == 'daily':
            return len(data.get('Time Series (Daily)', {}))
        elif data_type == 'intraday':
            time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
            return len(data.get(time_series_key, {})) if time_series_key else 0
        elif data_type == 'overview':
            return len([v for v in data.values() if v and str(v).lower() not in ['n/a', 'none', '', '0']])
        else:
            return len(data)

    def get_quality_rating(self, score: float) -> str:
        """Get quality rating based on score"""
        if score >= self.quality_thresholds['excellent']:
            return 'Excellent'
        elif score >= self.quality_thresholds['good']:
            return 'Good'
        elif score >= self.quality_thresholds['fair']:
            return 'Fair'
        elif score >= self.quality_thresholds['poor']:
            return 'Poor'
        else:
            return 'Very Poor'

    async def batch_validate(self, validation_requests: List[Tuple[str, str, Dict[str, Any], DataSource]]) -> List[ValidationResult]:
        """Validate multiple data sets in parallel"""
        tasks = []
        
        for ticker, data_type, data, source in validation_requests:
            task = self.validate_data(ticker, data_type, data, source)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch validation error: {result}")
            else:
                valid_results.append(result)
        
        return valid_results

    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not results:
            return {'error': 'No validation results provided'}
        
        # Aggregate statistics
        total_results = len(results)
        avg_quality_score = sum(r.quality_score for r in results) / total_results
        
        quality_distribution = {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0,
            'very_poor': 0
        }
        
        severity_counts = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }
        
        category_counts = {
            'completeness': 0,
            'accuracy': 0,
            'consistency': 0,
            'timeliness': 0,
            'plausibility': 0,
            'uniqueness': 0
        }
        
        for result in results:
            # Quality distribution
            rating = self.get_quality_rating(result.quality_score).lower().replace(' ', '_')
            quality_distribution[rating] += 1
            
            # Issue categorization
            for issue in result.issues:
                severity_counts[issue.severity.value] += 1
                category_counts[issue.category.value] += 1
        
        # Top issues
        all_issues = [issue for result in results for issue in result.issues]
        critical_issues = [i for i in all_issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in all_issues if i.severity == ValidationSeverity.ERROR]
        
        return {
            'summary': {
                'total_validations': total_results,
                'average_quality_score': round(avg_quality_score, 2),
                'overall_rating': self.get_quality_rating(avg_quality_score),
                'cross_validated_count': sum(1 for r in results if r.cross_validated)
            },
            'quality_distribution': quality_distribution,
            'issue_breakdown': {
                'by_severity': severity_counts,
                'by_category': category_counts,
                'total_issues': sum(severity_counts.values())
            },
            'critical_issues': len(critical_issues),
            'error_issues': len(error_issues),
            'top_issues': [
                {
                    'message': issue.message,
                    'severity': issue.severity.value,
                    'category': issue.category.value,
                    'field': issue.field
                }
                for issue in sorted(all_issues, key=lambda x: (x.severity.value, x.category.value))[:10]
            ],
            'source_performance': {
                source.value: {
                    'count': sum(1 for r in results if r.source == source),
                    'avg_quality': sum(r.quality_score for r in results if r.source == source) / max(1, sum(1 for r in results if r.source == source))
                }
                for source in set(r.source for r in results)
            }
        }

# Utility functions
async def validate_ticker_data(ticker: str, data_type: str, primary_data: Dict[str, Any], 
                             primary_source: DataSource, reference_data: Optional[Dict[DataSource, Dict[str, Any]]] = None) -> ValidationResult:
    """Convenience function to validate ticker data"""
    validator = DataQualityValidator()
    return await validator.validate_data(ticker, data_type, primary_data, primary_source, reference_data)

def is_data_quality_acceptable(result: ValidationResult, min_score: float = 60.0) -> bool:
    """Check if data quality meets minimum requirements"""
    return result.quality_score >= min_score and not any(
        issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] 
        for issue in result.issues
    )