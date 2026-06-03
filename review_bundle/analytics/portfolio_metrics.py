#!/usr/bin/env python3
"""
F05 - Portfolio Analytics & Benchmarks

Comprehensive portfolio analysis including:
- Sector/industry allocation computation
- Beta calculation via rolling OLS regression
- Benchmark comparison vs ^GSPC, ^NDX, ^DJI
- Earnings/dividends aggregation
- Risk metrics and performance attribution
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yfinance as yf
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# BEGIN F05 - Sector mapping for portfolio allocation
SECTOR_MAPPING = {
    # Technology
    'AAPL': {'sector': 'Technology', 'industry': 'Consumer Electronics'},
    'MSFT': {'sector': 'Technology', 'industry': 'Software'},
    'GOOGL': {'sector': 'Communication Services', 'industry': 'Internet Content & Information'},
    'AMZN': {'sector': 'Consumer Discretionary', 'industry': 'Internet & Direct Marketing Retail'},
    'META': {'sector': 'Communication Services', 'industry': 'Social Media'},
    'NVDA': {'sector': 'Technology', 'industry': 'Semiconductors'},
    'TSLA': {'sector': 'Consumer Discretionary', 'industry': 'Auto Manufacturers'},
    'NFLX': {'sector': 'Communication Services', 'industry': 'Entertainment'},
    'AMD': {'sector': 'Technology', 'industry': 'Semiconductors'},
    'INTC': {'sector': 'Technology', 'industry': 'Semiconductors'},
    'QCOM': {'sector': 'Technology', 'industry': 'Semiconductors'},
    'AVGO': {'sector': 'Technology', 'industry': 'Semiconductors'},
    'PYPL': {'sector': 'Financial Services', 'industry': 'Credit Services'},
    'ADBE': {'sector': 'Technology', 'industry': 'Software'},
    'CRM': {'sector': 'Technology', 'industry': 'Software'},
    
    # Financial Services
    'JPM': {'sector': 'Financial Services', 'industry': 'Banks'},
    'BAC': {'sector': 'Financial Services', 'industry': 'Banks'},
    'WFC': {'sector': 'Financial Services', 'industry': 'Banks'},
    'GS': {'sector': 'Financial Services', 'industry': 'Capital Markets'},
    'MS': {'sector': 'Financial Services', 'industry': 'Capital Markets'},
    'C': {'sector': 'Financial Services', 'industry': 'Banks'},
    'V': {'sector': 'Financial Services', 'industry': 'Credit Services'},
    'MA': {'sector': 'Financial Services', 'industry': 'Credit Services'},
    
    # Healthcare
    'JNJ': {'sector': 'Healthcare', 'industry': 'Drug Manufacturers'},
    'PFE': {'sector': 'Healthcare', 'industry': 'Drug Manufacturers'},
    'UNH': {'sector': 'Healthcare', 'industry': 'Healthcare Plans'},
    'ABBV': {'sector': 'Healthcare', 'industry': 'Drug Manufacturers'},
    'MRK': {'sector': 'Healthcare', 'industry': 'Drug Manufacturers'},
    'LLY': {'sector': 'Healthcare', 'industry': 'Drug Manufacturers'},
    'BMY': {'sector': 'Healthcare', 'industry': 'Drug Manufacturers'},
    'ABT': {'sector': 'Healthcare', 'industry': 'Medical Devices'},
    
    # Consumer & Retail
    'WMT': {'sector': 'Consumer Staples', 'industry': 'Discount Stores'},
    'HD': {'sector': 'Consumer Discretionary', 'industry': 'Home Improvement Retail'},
    'PG': {'sector': 'Consumer Staples', 'industry': 'Household & Personal Products'},
    'KO': {'sector': 'Consumer Staples', 'industry': 'Beverages'},
    'PEP': {'sector': 'Consumer Staples', 'industry': 'Beverages'},
    'MCD': {'sector': 'Consumer Discretionary', 'industry': 'Restaurants'},
    'DIS': {'sector': 'Communication Services', 'industry': 'Entertainment'},
    'NKE': {'sector': 'Consumer Discretionary', 'industry': 'Footwear & Accessories'},
    
    # Energy
    'XOM': {'sector': 'Energy', 'industry': 'Oil & Gas Integrated'},
    'CVX': {'sector': 'Energy', 'industry': 'Oil & Gas Integrated'},
    'COP': {'sector': 'Energy', 'industry': 'Oil & Gas E&P'},
    
    # Industrial
    'BA': {'sector': 'Industrials', 'industry': 'Aerospace & Defense'},
    'CAT': {'sector': 'Industrials', 'industry': 'Farm & Heavy Construction Machinery'},
    'GE': {'sector': 'Industrials', 'industry': 'Specialty Industrial Machinery'},
    'RTX': {'sector': 'Industrials', 'industry': 'Aerospace & Defense'},
    'LMT': {'sector': 'Industrials', 'industry': 'Aerospace & Defense'},
    'MMM': {'sector': 'Industrials', 'industry': 'Conglomerates'},
    'HON': {'sector': 'Industrials', 'industry': 'Specialty Industrial Machinery'},
    
    # Other stocks from config
    'UUUU': {'sector': 'Basic Materials', 'industry': 'Uranium'},
    'SRAD': {'sector': 'Communication Services', 'industry': 'Sports Betting'},
    'MRVL': {'sector': 'Technology', 'industry': 'Semiconductors'},
    'ADI': {'sector': 'Technology', 'industry': 'Semiconductors'},
    'PLTR': {'sector': 'Technology', 'industry': 'Software'},
    'RIVN': {'sector': 'Consumer Discretionary', 'industry': 'Auto Manufacturers'},
    'SPX': {'sector': 'Index', 'industry': 'S&P 500'},  # Special case for index
}

BENCHMARK_SYMBOLS = ['^GSPC', '^IXIC', '^DJI']  # S&P 500, NASDAQ, Dow Jones
# END F05

class PortfolioAnalyzer:
    """
    F05: Comprehensive portfolio analytics with sector allocation, beta calculation,
    and benchmark comparison
    """
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.metrics = {
            'processing_time_ms': 0,
            'tickers_processed': 0,
            'benchmarks_processed': 0,
            'sector_allocations': 0,
            'betas_calculated': 0
        }
        
        logger.info("F05: PortfolioAnalyzer initialized")
    
    def _extract_price_data(self, data: pd.DataFrame, symbol: str = None) -> pd.Series:
        """
        Extract price data from yfinance DataFrame, handling multi-level columns
        """
        if data.empty:
            return pd.Series()
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            if symbol and ('Adj Close', symbol) in data.columns:
                return data[('Adj Close', symbol)]
            elif 'Adj Close' in [col[0] for col in data.columns]:
                # Take the first Adj Close column
                adj_close_cols = [col for col in data.columns if col[0] == 'Adj Close']
                return data[adj_close_cols[0]]
            else:
                # Fallback to Close
                close_cols = [col for col in data.columns if col[0] == 'Close']
                if close_cols:
                    return data[close_cols[0]]
        else:
            # Single-level columns
            if 'Adj Close' in data.columns:
                return data['Adj Close']
            elif 'Close' in data.columns:
                return data['Close']
        
        # Last resort - return the last column
        return data.iloc[:, -1]
    
    def compute_allocations(self, portfolio_tickers: List[str], 
                          portfolio_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        F05: Compute sector and industry allocations for portfolio
        
        Args:
            portfolio_tickers: List of portfolio symbols
            portfolio_weights: Optional dict of symbol -> weight (defaults to equal weight)
            
        Returns:
            Dict with sector/industry allocation data
        """
        start_time = datetime.now()
        
        if portfolio_weights is None:
            # Equal weighting if no weights provided
            weight = 1.0 / len(portfolio_tickers)
            portfolio_weights = {ticker: weight for ticker in portfolio_tickers}
        
        # Normalize weights to sum to 1
        total_weight = sum(portfolio_weights.values())
        if total_weight > 0:
            portfolio_weights = {k: v/total_weight for k, v in portfolio_weights.items()}
        
        sector_allocation = {}
        industry_allocation = {}
        unmapped_tickers = []
        
        # Aggregate by sector and industry
        for ticker in portfolio_tickers:
            weight = portfolio_weights.get(ticker, 0)
            
            if ticker in SECTOR_MAPPING:
                sector_info = SECTOR_MAPPING[ticker]
                sector = sector_info['sector']
                industry = sector_info['industry']
                
                # Aggregate sector weights
                if sector in sector_allocation:
                    sector_allocation[sector] += weight
                else:
                    sector_allocation[sector] = weight
                
                # Aggregate industry weights
                industry_key = f"{sector}::{industry}"
                if industry_key in industry_allocation:
                    industry_allocation[industry_key] += weight
                else:
                    industry_allocation[industry_key] = weight
            else:
                unmapped_tickers.append(ticker)
                logger.warning(f"F05: No sector mapping for {ticker}")
        
        # Sort allocations by weight
        sector_allocation = dict(sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True))
        industry_allocation = dict(sorted(industry_allocation.items(), key=lambda x: x[1], reverse=True))
        
        self.metrics['sector_allocations'] += 1
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"F05: Computed allocations for {len(portfolio_tickers)} tickers in {processing_time:.1f}ms")
        
        return {
            'sector_allocation': sector_allocation,
            'industry_allocation': industry_allocation,
            'portfolio_weights': portfolio_weights,
            'unmapped_tickers': unmapped_tickers,
            'total_mapped_weight': sum(sector_allocation.values()),
            'processing_time_ms': processing_time
        }
    
    def compute_betas(self, portfolio_tickers: List[str], 
                     benchmark_symbol: str = '^GSPC',
                     lookback_days: int = 252) -> Dict[str, Any]:
        """
        F05: Compute beta coefficients via rolling OLS regression vs benchmark
        
        Args:
            portfolio_tickers: List of symbols to compute betas for
            benchmark_symbol: Benchmark index symbol (default: S&P 500)
            lookback_days: Rolling window for beta calculation (default: 252 trading days)
            
        Returns:
            Dict with beta calculations and statistics
        """
        start_time = datetime.now()
        
        betas = {}
        beta_stats = {}
        failed_tickers = []
        
        try:
            # Download benchmark data
            logger.info(f"F05: Downloading benchmark data for {benchmark_symbol}")
            benchmark_data = yf.download(benchmark_symbol, period=f"{int(lookback_days*1.5)}d", progress=False)
            
            if benchmark_data.empty:
                logger.error(f"F05: Failed to download benchmark {benchmark_symbol}")
                return {'error': f'Failed to download benchmark {benchmark_symbol}'}
            
            benchmark_prices = self._extract_price_data(benchmark_data, benchmark_symbol)
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Process each ticker
            for ticker in portfolio_tickers:
                try:
                    logger.debug(f"F05: Computing beta for {ticker}")
                    
                    # Download ticker data
                    ticker_data = yf.download(ticker, period=f"{int(lookback_days*1.5)}d", progress=False)
                    
                    if ticker_data.empty:
                        logger.warning(f"F05: No data for {ticker}")
                        failed_tickers.append(ticker)
                        continue
                    
                    ticker_prices = self._extract_price_data(ticker_data, ticker)
                    ticker_returns = ticker_prices.pct_change().dropna()
                    
                    # Align dates between ticker and benchmark
                    aligned_data = pd.concat([ticker_returns, benchmark_returns], axis=1, join='inner')
                    aligned_data.columns = ['ticker_returns', 'benchmark_returns']
                    aligned_data = aligned_data.dropna()
                    
                    if len(aligned_data) < 30:  # Minimum data requirement
                        logger.warning(f"F05: Insufficient data for {ticker} ({len(aligned_data)} observations)")
                        failed_tickers.append(ticker)
                        continue
                    
                    # Use most recent lookback_days observations
                    recent_data = aligned_data.tail(lookback_days)
                    
                    # Calculate beta using OLS regression
                    X = recent_data['benchmark_returns'].values.reshape(-1, 1)
                    y = recent_data['ticker_returns'].values
                    
                    # Sklearn linear regression
                    model = LinearRegression().fit(X, y)
                    beta = float(model.coef_[0])
                    alpha = float(model.intercept_)
                    
                    # Calculate R-squared and other statistics
                    y_pred = model.predict(X)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Additional statistics
                    correlation = np.corrcoef(recent_data['ticker_returns'], recent_data['benchmark_returns'])[0, 1]
                    ticker_volatility = recent_data['ticker_returns'].std() * np.sqrt(252)  # Annualized
                    benchmark_volatility = recent_data['benchmark_returns'].std() * np.sqrt(252)
                    
                    # Tracking error
                    tracking_error = (recent_data['ticker_returns'] - recent_data['benchmark_returns']).std() * np.sqrt(252)
                    
                    betas[ticker] = beta
                    beta_stats[ticker] = {
                        'beta': beta,
                        'alpha': alpha,
                        'r_squared': r_squared,
                        'correlation': correlation,
                        'ticker_volatility': ticker_volatility,
                        'benchmark_volatility': benchmark_volatility,
                        'tracking_error': tracking_error,
                        'observations': len(recent_data)
                    }
                    
                    self.metrics['betas_calculated'] += 1
                    
                except Exception as e:
                    logger.error(f"F05: Beta calculation failed for {ticker}: {e}")
                    failed_tickers.append(ticker)
                    continue
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics['processing_time_ms'] += processing_time
            
            logger.info(f"F05: Computed {len(betas)} betas vs {benchmark_symbol} in {processing_time:.1f}ms")
            
            return {
                'betas': betas,
                'beta_stats': beta_stats,
                'benchmark_symbol': benchmark_symbol,
                'failed_tickers': failed_tickers,
                'lookback_days': lookback_days,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"F05: Beta computation failed: {e}")
            return {'error': str(e)}
    
    def compute_benchmark_table(self, portfolio_tickers: List[str],
                               benchmark_symbols: List[str] = None) -> Dict[str, Any]:
        """
        F05: Compute comprehensive benchmark comparison table
        
        Args:
            portfolio_tickers: List of portfolio symbols
            benchmark_symbols: List of benchmark symbols (defaults to S&P 500, NASDAQ, Dow)
            
        Returns:
            Dict with benchmark performance comparison data
        """
        if benchmark_symbols is None:
            benchmark_symbols = BENCHMARK_SYMBOLS
            
        start_time = datetime.now()
        
        results = {
            'benchmark_performance': {},
            'portfolio_performance': {},
            'relative_performance': {},
            'correlations': {},
            'failed_symbols': []
        }
        
        periods = ['1M', '3M', '6M', '1Y']  # Performance periods
        
        try:
            # Download benchmark data
            benchmark_data = {}
            for benchmark in benchmark_symbols:
                try:
                    logger.info(f"F05: Downloading benchmark {benchmark}")
                    data = yf.download(benchmark, period='2y', progress=False)
                    
                    if not data.empty:
                        benchmark_prices = self._extract_price_data(data, benchmark)
                        benchmark_data[benchmark] = benchmark_prices
                        self.metrics['benchmarks_processed'] += 1
                    else:
                        logger.warning(f"F05: No data for benchmark {benchmark}")
                        results['failed_symbols'].append(benchmark)
                        
                except Exception as e:
                    logger.error(f"F05: Failed to download {benchmark}: {e}")
                    results['failed_symbols'].append(benchmark)
                    continue
            
            # Calculate benchmark returns for different periods
            for benchmark, prices in benchmark_data.items():
                perf = {}
                for period in periods:
                    try:
                        if period == '1M':
                            start_price = prices.iloc[-21] if len(prices) > 21 else prices.iloc[0]
                        elif period == '3M':
                            start_price = prices.iloc[-63] if len(prices) > 63 else prices.iloc[0]
                        elif period == '6M':
                            start_price = prices.iloc[-126] if len(prices) > 126 else prices.iloc[0]
                        elif period == '1Y':
                            start_price = prices.iloc[-252] if len(prices) > 252 else prices.iloc[0]
                        
                        end_price = prices.iloc[-1]
                        perf[period] = ((end_price - start_price) / start_price) * 100
                        
                    except Exception as e:
                        logger.warning(f"F05: Failed to calculate {period} performance for {benchmark}: {e}")
                        perf[period] = None
                
                results['benchmark_performance'][benchmark] = perf
            
            # Download and calculate portfolio performance
            for ticker in portfolio_tickers:
                try:
                    logger.debug(f"F05: Downloading performance data for {ticker}")
                    data = yf.download(ticker, period='2y', progress=False)
                    
                    if data.empty:
                        logger.warning(f"F05: No data for {ticker}")
                        results['failed_symbols'].append(ticker)
                        continue
                    
                    prices = self._extract_price_data(data, ticker)
                    perf = {}
                    
                    # Calculate returns for each period
                    for period in periods:
                        try:
                            if period == '1M':
                                start_price = prices.iloc[-21] if len(prices) > 21 else prices.iloc[0]
                            elif period == '3M':
                                start_price = prices.iloc[-63] if len(prices) > 63 else prices.iloc[0]
                            elif period == '6M':
                                start_price = prices.iloc[-126] if len(prices) > 126 else prices.iloc[0]
                            elif period == '1Y':
                                start_price = prices.iloc[-252] if len(prices) > 252 else prices.iloc[0]
                            
                            end_price = prices.iloc[-1]
                            perf[period] = ((end_price - start_price) / start_price) * 100
                            
                        except Exception as e:
                            logger.warning(f"F05: Failed to calculate {period} performance for {ticker}: {e}")
                            perf[period] = None
                    
                    results['portfolio_performance'][ticker] = perf
                    self.metrics['tickers_processed'] += 1
                    
                    # Calculate correlations with benchmarks
                    correlations = {}
                    ticker_returns = prices.pct_change().dropna()
                    
                    for benchmark, benchmark_prices in benchmark_data.items():
                        try:
                            benchmark_returns = benchmark_prices.pct_change().dropna()
                            
                            # Align dates
                            aligned = pd.concat([ticker_returns, benchmark_returns], axis=1, join='inner')
                            if len(aligned) > 30:
                                correlation = aligned.corr().iloc[0, 1]
                                correlations[benchmark] = correlation
                            
                        except Exception as e:
                            logger.warning(f"F05: Correlation calculation failed for {ticker} vs {benchmark}: {e}")
                            continue
                    
                    results['correlations'][ticker] = correlations
                    
                except Exception as e:
                    logger.error(f"F05: Performance calculation failed for {ticker}: {e}")
                    results['failed_symbols'].append(ticker)
                    continue
            
            # Calculate relative performance (portfolio vs benchmarks)
            for ticker, ticker_perf in results['portfolio_performance'].items():
                relative_perf = {}
                for benchmark, benchmark_perf in results['benchmark_performance'].items():
                    rel_perf = {}
                    for period in periods:
                        ticker_ret = ticker_perf.get(period)
                        benchmark_ret = benchmark_perf.get(period)
                        
                        if ticker_ret is not None and benchmark_ret is not None:
                            rel_perf[period] = ticker_ret - benchmark_ret  # Excess return
                        else:
                            rel_perf[period] = None
                    
                    relative_perf[benchmark] = rel_perf
                
                results['relative_performance'][ticker] = relative_perf
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics['processing_time_ms'] += processing_time
            
            logger.info(f"F05: Computed benchmark table for {len(portfolio_tickers)} tickers vs {len(benchmark_symbols)} benchmarks in {processing_time:.1f}ms")
            
            results['processing_time_ms'] = processing_time
            results['periods'] = periods
            results['benchmark_symbols'] = list(benchmark_data.keys())
            
            return results
            
        except Exception as e:
            logger.error(f"F05: Benchmark table computation failed: {e}")
            return {'error': str(e)}
    
    async def get_earnings_dividends_data(self, portfolio_tickers: List[str]) -> Dict[str, Any]:
        """
        F05: Get earnings and dividends data aggregation
        
        Args:
            portfolio_tickers: List of portfolio symbols
            
        Returns:
            Dict with earnings/dividends roll-up data
        """
        try:
            # Try to use existing earnings service if available
            try:
                from services.earnings_service import EarningsService
                earnings_service = EarningsService()
                
                earnings_data = {}
                for ticker in portfolio_tickers:
                    try:
                        # Get recent earnings data
                        ticker_earnings = await earnings_service.get_earnings_data(ticker)
                        if ticker_earnings:
                            earnings_data[ticker] = ticker_earnings
                    except Exception as e:
                        logger.warning(f"F05: Earnings data failed for {ticker}: {e}")
                        continue
                
                return {
                    'earnings_data': earnings_data,
                    'source': 'earnings_service'
                }
                
            except ImportError:
                # Fallback to yfinance dividend data
                logger.info("F05: Using yfinance fallback for dividends data")
                
                dividends_data = {}
                for ticker in portfolio_tickers:
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        
                        # Get dividend history
                        dividends = ticker_obj.dividends
                        if not dividends.empty:
                            # Get last 12 months of dividends
                            recent_dividends = dividends.last('1Y')
                            annual_dividend = recent_dividends.sum()
                            
                            dividends_data[ticker] = {
                                'annual_dividend': float(annual_dividend),
                                'dividend_count': len(recent_dividends),
                                'last_dividend_date': recent_dividends.index[-1].strftime('%Y-%m-%d') if len(recent_dividends) > 0 else None,
                                'last_dividend_amount': float(recent_dividends.iloc[-1]) if len(recent_dividends) > 0 else 0
                            }
                        
                    except Exception as e:
                        logger.warning(f"F05: Dividend data failed for {ticker}: {e}")
                        continue
                
                return {
                    'dividends_data': dividends_data,
                    'source': 'yfinance_fallback'
                }
                
        except Exception as e:
            logger.error(f"F05: Earnings/dividends aggregation failed: {e}")
            return {'error': str(e)}
    
    def save_portfolio_metrics(self, portfolio_data: Dict[str, Any], 
                              filename: str = "portfolio_metrics.csv") -> bool:
        """
        F05: Save portfolio metrics to CSV file
        
        Args:
            portfolio_data: Portfolio analysis results
            filename: Output CSV filename
            
        Returns:
            True if saved successfully
        """
        try:
            output_path = self.data_dir / filename
            
            # Flatten portfolio data for CSV format
            rows = []
            
            # Add sector allocations
            if 'sector_allocation' in portfolio_data:
                for sector, weight in portfolio_data['sector_allocation'].items():
                    rows.append({
                        'metric_type': 'sector_allocation',
                        'symbol': sector,
                        'value': weight,
                        'description': f'Sector allocation weight'
                    })
            
            # Add beta statistics
            if 'beta_stats' in portfolio_data:
                for ticker, stats in portfolio_data['beta_stats'].items():
                    for stat_name, stat_value in stats.items():
                        rows.append({
                            'metric_type': f'beta_{stat_name}',
                            'symbol': ticker,
                            'value': stat_value,
                            'description': f'Beta statistic: {stat_name}'
                        })
            
            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df['timestamp'] = datetime.now().isoformat()
            
            # Atomic save
            temp_path = str(output_path) + '.tmp'
            df.to_csv(temp_path, index=False)
            Path(temp_path).replace(output_path)
            
            logger.info(f"F05: Portfolio metrics saved to {output_path} ({len(rows)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"F05: Failed to save portfolio metrics: {e}")
            return False
    
    def save_benchmarks(self, benchmark_data: Dict[str, Any],
                       filename: str = "benchmarks.csv") -> bool:
        """
        F05: Save benchmark comparison data to CSV file
        
        Args:
            benchmark_data: Benchmark comparison results
            filename: Output CSV filename
            
        Returns:
            True if saved successfully
        """
        try:
            output_path = self.data_dir / filename
            
            # Flatten benchmark data for CSV format
            rows = []
            
            # Add benchmark performance
            if 'benchmark_performance' in benchmark_data:
                for benchmark, perf in benchmark_data['benchmark_performance'].items():
                    for period, return_pct in perf.items():
                        rows.append({
                            'benchmark': benchmark,
                            'symbol': benchmark,
                            'period': period,
                            'return_pct': return_pct,
                            'metric_type': 'benchmark_performance'
                        })
            
            # Add portfolio performance
            if 'portfolio_performance' in benchmark_data:
                for ticker, perf in benchmark_data['portfolio_performance'].items():
                    for period, return_pct in perf.items():
                        rows.append({
                            'benchmark': 'PORTFOLIO',
                            'symbol': ticker,
                            'period': period,
                            'return_pct': return_pct,
                            'metric_type': 'portfolio_performance'
                        })
            
            # Add relative performance
            if 'relative_performance' in benchmark_data:
                for ticker, benchmarks in benchmark_data['relative_performance'].items():
                    for benchmark, perf in benchmarks.items():
                        for period, excess_return in perf.items():
                            rows.append({
                                'benchmark': benchmark,
                                'symbol': ticker,
                                'period': period,
                                'return_pct': excess_return,
                                'metric_type': 'relative_performance'
                            })
            
            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df['timestamp'] = datetime.now().isoformat()
            
            # Atomic save
            temp_path = str(output_path) + '.tmp'
            df.to_csv(temp_path, index=False)
            Path(temp_path).replace(output_path)
            
            logger.info(f"F05: Benchmarks saved to {output_path} ({len(rows)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"F05: Failed to save benchmarks: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get F05 performance and telemetry metrics"""
        return self.metrics.copy()

# F05: Public API functions
def analyze_portfolio(portfolio_tickers: List[str], 
                     portfolio_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    F05: Complete portfolio analysis including allocations, betas, and benchmarks
    
    Args:
        portfolio_tickers: List of portfolio symbols
        portfolio_weights: Optional symbol -> weight mapping
        
    Returns:
        Complete portfolio analysis results
    """
    analyzer = PortfolioAnalyzer()
    
    results = {}
    
    try:
        # Compute sector allocations
        logger.info("F05: Computing portfolio allocations")
        allocations = analyzer.compute_allocations(portfolio_tickers, portfolio_weights)
        results.update(allocations)
        
        # Compute betas vs S&P 500
        logger.info("F05: Computing portfolio betas")
        betas = analyzer.compute_betas(portfolio_tickers)
        results.update(betas)
        
        # Compute benchmark comparison
        logger.info("F05: Computing benchmark comparison")
        benchmarks = analyzer.compute_benchmark_table(portfolio_tickers)
        results.update(benchmarks)
        
        # Save results to CSV
        analyzer.save_portfolio_metrics(results)
        analyzer.save_benchmarks(results)
        
        # Add performance metrics
        results['f05_metrics'] = analyzer.get_performance_metrics()
        
        logger.info(f"F05: Portfolio analysis completed for {len(portfolio_tickers)} tickers")
        
        return results
        
    except Exception as e:
        logger.error(f"F05: Portfolio analysis failed: {e}")
        return {'error': str(e)}

# END F05