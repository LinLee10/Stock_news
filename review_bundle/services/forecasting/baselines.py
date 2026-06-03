#!/usr/bin/env python3
"""
Baseline forecasting models (ARIMA, Prophet)
Optional implementations for forecasting comparison and evaluation
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optional imports for baseline models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from config.feature_flags import is_alt_forecasts_enabled

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result from baseline forecasting model"""
    model_name: str
    symbol: str
    predictions: List[float]
    forecast_dates: List[datetime]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    runtime_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None


@dataclass
class BaselineMetrics:
    """Metrics for baseline model evaluation"""
    symbol: str
    arima_mae: Optional[float] = None
    arima_rmse: Optional[float] = None
    arima_runtime_ms: Optional[int] = None
    prophet_mae: Optional[float] = None
    prophet_rmse: Optional[float] = None  
    prophet_runtime_ms: Optional[int] = None
    arima_success: bool = False
    prophet_success: bool = False


class BaselineForecaster:
    """Base class for baseline forecasting models"""
    
    def __init__(self, forecast_horizon: int = 7):
        self.forecast_horizon = forecast_horizon
        self.min_data_points = 30  # Minimum required for reliable forecasting
        
    def prepare_data(self, prices: List[Dict], symbol: str) -> Optional[pd.DataFrame]:
        """
        Prepare price data for forecasting
        
        Args:
            prices: List of price dictionaries with 'date' and 'close' keys
            symbol: Stock symbol for logging
            
        Returns:
            DataFrame with date and price columns, or None if insufficient data
        """
        if not prices or len(prices) < self.min_data_points:
            logger.warning(f"Insufficient data for {symbol}: {len(prices)} points, need {self.min_data_points}")
            return None
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(prices)
            
            # Ensure we have the required columns
            if 'date' not in df.columns or 'close' not in df.columns:
                logger.error(f"Missing required columns for {symbol}: {df.columns.tolist()}")
                return None
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date and remove duplicates
            df = df.sort_values('date').drop_duplicates(subset=['date'])
            
            # Remove any NaN values
            df = df.dropna(subset=['close'])
            
            # Final check for sufficient data
            if len(df) < self.min_data_points:
                logger.warning(f"Insufficient clean data for {symbol}: {len(df)} points")
                return None
            
            return df[['date', 'close']].reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {e}")
            return None
    
    def calculate_metrics(self, actual: List[float], predicted: List[float]) -> Tuple[float, float]:
        """
        Calculate MAE and RMSE metrics
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Tuple of (MAE, RMSE)
        """
        if not actual or not predicted or len(actual) != len(predicted):
            return float('inf'), float('inf')
            
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual_arr - predicted_arr))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((actual_arr - predicted_arr) ** 2))
        
        return float(mae), float(rmse)


class ARIMAForecaster(BaselineForecaster):
    """ARIMA (AutoRegressive Integrated Moving Average) baseline forecaster"""
    
    def __init__(self, forecast_horizon: int = 7):
        super().__init__(forecast_horizon)
        self.default_order = (1, 1, 1)  # (p, d, q) parameters
        self.max_runtime_seconds = 30  # Timeout for model fitting
        
    def auto_arima_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Simple automatic ARIMA order selection
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (p, d, q) parameters
        """
        # Start with default
        best_order = self.default_order
        best_aic = float('inf')
        
        # Try a few common configurations
        orders_to_try = [
            (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2),
            (0, 1, 1), (1, 1, 0), (0, 1, 0)
        ]
        
        for order in orders_to_try:
            try:
                model = ARIMA(series, order=order)
                fitted = model.fit()
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = order
                    
            except Exception:
                continue
        
        return best_order
    
    def forecast(self, prices: List[Dict], symbol: str) -> ForecastResult:
        """
        Generate ARIMA forecast
        
        Args:
            prices: List of price dictionaries
            symbol: Stock symbol
            
        Returns:
            ForecastResult with predictions and metrics
        """
        start_time = time.time()
        
        try:
            # Check if ARIMA is available
            if not STATSMODELS_AVAILABLE:
                return ForecastResult(
                    model_name="ARIMA",
                    symbol=symbol,
                    predictions=[],
                    forecast_dates=[],
                    success=False,
                    error_message="statsmodels not available",
                    runtime_ms=int((time.time() - start_time) * 1000)
                )
            
            # Prepare data
            df = self.prepare_data(prices, symbol)
            if df is None:
                return ForecastResult(
                    model_name="ARIMA",
                    symbol=symbol,
                    predictions=[],
                    forecast_dates=[],
                    success=False,
                    error_message="Insufficient or invalid data",
                    runtime_ms=int((time.time() - start_time) * 1000)
                )
            
            # Extract time series
            series = df['close']
            dates = df['date']
            
            # Automatic order selection (with timeout protection)
            try:
                order = self.auto_arima_order(series)
            except Exception:
                order = self.default_order
            
            # Fit ARIMA model
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_result = fitted_model.forecast(steps=self.forecast_horizon)
            confidence_intervals = fitted_model.get_forecast(steps=self.forecast_horizon).conf_int()
            
            # Generate forecast dates
            last_date = dates.iloc[-1]
            forecast_dates = [
                last_date + timedelta(days=i+1) 
                for i in range(self.forecast_horizon)
            ]
            
            # Convert results to lists
            predictions = forecast_result.tolist() if hasattr(forecast_result, 'tolist') else list(forecast_result)
            
            # Extract confidence intervals
            ci_tuples = None
            if confidence_intervals is not None:
                ci_tuples = [
                    (float(row[0]), float(row[1])) 
                    for _, row in confidence_intervals.iterrows()
                ]
            
            runtime_ms = int((time.time() - start_time) * 1000)
            
            # Calculate in-sample metrics (using last 20% for validation)
            validation_size = max(1, len(series) // 5)
            train_series = series[:-validation_size]
            validation_actual = series[-validation_size:].tolist()
            
            # Fit model on training data for validation
            val_model = ARIMA(train_series, order=order)
            val_fitted = val_model.fit()
            val_predictions = val_fitted.forecast(steps=validation_size).tolist()
            
            mae, rmse = self.calculate_metrics(validation_actual, val_predictions)
            
            return ForecastResult(
                model_name="ARIMA",
                symbol=symbol,
                predictions=predictions,
                forecast_dates=forecast_dates,
                confidence_intervals=ci_tuples,
                mae=mae,
                rmse=rmse,
                runtime_ms=runtime_ms,
                success=True,
                model_params={'order': order, 'aic': fitted_model.aic}
            )
            
        except Exception as e:
            runtime_ms = int((time.time() - start_time) * 1000)
            logger.error(f"ARIMA forecast failed for {symbol}: {e}")
            
            return ForecastResult(
                model_name="ARIMA",
                symbol=symbol,
                predictions=[],
                forecast_dates=[],
                success=False,
                error_message=str(e),
                runtime_ms=runtime_ms
            )


class ProphetForecaster(BaselineForecaster):
    """Facebook Prophet baseline forecaster"""
    
    def __init__(self, forecast_horizon: int = 7):
        super().__init__(forecast_horizon)
        self.max_runtime_seconds = 60  # Prophet can be slower
        
    def forecast(self, prices: List[Dict], symbol: str) -> ForecastResult:
        """
        Generate Prophet forecast
        
        Args:
            prices: List of price dictionaries
            symbol: Stock symbol
            
        Returns:
            ForecastResult with predictions and metrics
        """
        start_time = time.time()
        
        try:
            # Check if Prophet is available
            if not PROPHET_AVAILABLE:
                return ForecastResult(
                    model_name="Prophet",
                    symbol=symbol,
                    predictions=[],
                    forecast_dates=[],
                    success=False,
                    error_message="prophet not available",
                    runtime_ms=int((time.time() - start_time) * 1000)
                )
            
            # Prepare data
            df = self.prepare_data(prices, symbol)
            if df is None:
                return ForecastResult(
                    model_name="Prophet",
                    symbol=symbol,
                    predictions=[],
                    forecast_dates=[],
                    success=False,
                    error_message="Insufficient or invalid data",
                    runtime_ms=int((time.time() - start_time) * 1000)
                )
            
            # Prepare Prophet data format
            prophet_df = pd.DataFrame({
                'ds': df['date'],  # Prophet expects 'ds' for dates
                'y': df['close']   # Prophet expects 'y' for values
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05,
                seasonality_mode='additive'
            )
            
            # Suppress Prophet's verbose output
            import logging as prophet_logging
            prophet_logging.getLogger('prophet').setLevel(prophet_logging.WARNING)
            
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=self.forecast_horizon, freq='D')
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract forecast for future periods only
            future_forecast = forecast.tail(self.forecast_horizon)
            
            predictions = future_forecast['yhat'].tolist()
            forecast_dates = future_forecast['ds'].tolist()
            
            # Extract confidence intervals
            ci_tuples = [
                (float(row['yhat_lower']), float(row['yhat_upper']))
                for _, row in future_forecast.iterrows()
            ]
            
            runtime_ms = int((time.time() - start_time) * 1000)
            
            # Calculate cross-validation metrics (using last 20% for validation)
            validation_size = max(1, len(prophet_df) // 5)
            train_df = prophet_df[:-validation_size]
            validation_actual = prophet_df[-validation_size:]['y'].tolist()
            
            # Fit model on training data
            val_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            val_model.fit(train_df)
            
            # Predict validation period
            val_future = val_model.make_future_dataframe(periods=validation_size, freq='D')
            val_forecast = val_model.predict(val_future)
            val_predictions = val_forecast.tail(validation_size)['yhat'].tolist()
            
            mae, rmse = self.calculate_metrics(validation_actual, val_predictions)
            
            return ForecastResult(
                model_name="Prophet",
                symbol=symbol,
                predictions=predictions,
                forecast_dates=forecast_dates,
                confidence_intervals=ci_tuples,
                mae=mae,
                rmse=rmse,
                runtime_ms=runtime_ms,
                success=True,
                model_params={'changepoint_prior_scale': 0.05}
            )
            
        except Exception as e:
            runtime_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Prophet forecast failed for {symbol}: {e}")
            
            return ForecastResult(
                model_name="Prophet",
                symbol=symbol,
                predictions=[],
                forecast_dates=[],
                success=False,
                error_message=str(e),
                runtime_ms=runtime_ms
            )


class BaselineForecasterRegistry:
    """Registry for baseline forecasting models"""
    
    def __init__(self, forecast_horizon: int = 7):
        self.forecast_horizon = forecast_horizon
        self.forecasters = {
            'arima': ARIMAForecaster(forecast_horizon),
            'prophet': ProphetForecaster(forecast_horizon)
        }
        
    def run_baselines(self, prices_by_symbol: Dict[str, List[Dict]]) -> Dict[str, List[ForecastResult]]:
        """
        Run all baseline forecasters on provided price data
        
        Args:
            prices_by_symbol: Dict mapping symbol to list of price dictionaries
            
        Returns:
            Dict mapping symbol to list of ForecastResult objects
        """
        if not is_alt_forecasts_enabled():
            logger.debug("Alternative forecasts disabled by feature flag")
            return {}
            
        results = {}
        
        for symbol, prices in prices_by_symbol.items():
            symbol_results = []
            
            for forecaster_name, forecaster in self.forecasters.items():
                logger.info(f"Running {forecaster_name} forecast for {symbol}")
                
                try:
                    result = forecaster.forecast(prices, symbol)
                    symbol_results.append(result)
                    
                    if result.success:
                        logger.info(f"{forecaster_name} forecast for {symbol}: MAE={result.mae:.4f}, RMSE={result.rmse:.4f}, Runtime={result.runtime_ms}ms")
                    else:
                        logger.warning(f"{forecaster_name} forecast failed for {symbol}: {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"Error running {forecaster_name} for {symbol}: {e}")
                    
                    # Create failed result
                    symbol_results.append(ForecastResult(
                        model_name=forecaster_name.upper(),
                        symbol=symbol,
                        predictions=[],
                        forecast_dates=[],
                        success=False,
                        error_message=str(e)
                    ))
            
            results[symbol] = symbol_results
        
        return results
    
    def get_baseline_metrics(self, results: Dict[str, List[ForecastResult]]) -> Dict[str, BaselineMetrics]:
        """
        Extract baseline metrics from forecast results
        
        Args:
            results: Forecast results from run_baselines
            
        Returns:
            Dict mapping symbol to BaselineMetrics
        """
        metrics = {}
        
        for symbol, symbol_results in results.items():
            baseline_metrics = BaselineMetrics(symbol=symbol)
            
            for result in symbol_results:
                if result.model_name.lower() == 'arima':
                    baseline_metrics.arima_success = result.success
                    if result.success:
                        baseline_metrics.arima_mae = result.mae
                        baseline_metrics.arima_rmse = result.rmse
                        baseline_metrics.arima_runtime_ms = result.runtime_ms
                        
                elif result.model_name.lower() == 'prophet':
                    baseline_metrics.prophet_success = result.success
                    if result.success:
                        baseline_metrics.prophet_mae = result.mae
                        baseline_metrics.prophet_rmse = result.rmse
                        baseline_metrics.prophet_runtime_ms = result.runtime_ms
            
            metrics[symbol] = baseline_metrics
        
        return metrics


# Global registry instance
baseline_registry = BaselineForecasterRegistry()


# Convenience functions
def run_baseline_forecasts(prices_by_symbol: Dict[str, List[Dict]], forecast_horizon: int = 7) -> Dict[str, List[ForecastResult]]:
    """
    Run baseline forecasts on price data
    
    Args:
        prices_by_symbol: Dict mapping symbol to price data
        forecast_horizon: Number of days to forecast
        
    Returns:
        Forecast results by symbol
    """
    registry = BaselineForecasterRegistry(forecast_horizon)
    return registry.run_baselines(prices_by_symbol)


def get_baseline_summary(results: Dict[str, List[ForecastResult]]) -> Dict[str, Any]:
    """
    Get summary statistics from baseline forecast results
    
    Args:
        results: Results from run_baseline_forecasts
        
    Returns:
        Summary statistics
    """
    summary = {
        'total_symbols': len(results),
        'successful_forecasts': 0,
        'failed_forecasts': 0,
        'models_run': set(),
        'avg_runtime_ms': {},
        'avg_mae': {},
        'avg_rmse': {}
    }
    
    for symbol_results in results.values():
        for result in symbol_results:
            summary['models_run'].add(result.model_name)
            
            if result.success:
                summary['successful_forecasts'] += 1
                
                # Update averages
                model = result.model_name
                if model not in summary['avg_runtime_ms']:
                    summary['avg_runtime_ms'][model] = []
                    summary['avg_mae'][model] = []
                    summary['avg_rmse'][model] = []
                
                summary['avg_runtime_ms'][model].append(result.runtime_ms)
                if result.mae is not None:
                    summary['avg_mae'][model].append(result.mae)
                if result.rmse is not None:
                    summary['avg_rmse'][model].append(result.rmse)
            else:
                summary['failed_forecasts'] += 1
    
    # Calculate final averages
    for model in summary['models_run']:
        if summary['avg_runtime_ms'][model]:
            summary['avg_runtime_ms'][model] = np.mean(summary['avg_runtime_ms'][model])
        if summary['avg_mae'][model]:
            summary['avg_mae'][model] = np.mean(summary['avg_mae'][model])
        if summary['avg_rmse'][model]:
            summary['avg_rmse'][model] = np.mean(summary['avg_rmse'][model])
    
    summary['models_run'] = list(summary['models_run'])
    
    return summary