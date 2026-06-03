#!/usr/bin/env python3
"""
Correlation analysis for portfolio and watchlist stocks
Computes Pearson correlation matrix across aligned returns windows with optional heatmap visualization
"""

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from config.feature_flags import is_correlation_enabled

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """
    Portfolio correlation analysis with heatmap visualization
    
    Computes Pearson correlation coefficients across aligned returns windows
    and generates publication-ready heatmaps for correlation matrices.
    """
    
    def __init__(self, lookback_days: int = 252, min_data_points: int = 30):
        """
        Initialize correlation analyzer
        
        Args:
            lookback_days: Number of days to look back for correlation analysis
            min_data_points: Minimum number of overlapping data points required
        """
        self.lookback_days = lookback_days
        self.min_data_points = min_data_points
        self.correlation_matrix = None
        self.symbols = []
        self.returns_data = None
        
    def compute_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Compute Pearson correlation matrix from price data
        
        Args:
            price_data: Dict mapping symbol to DataFrame with 'Date' and 'Close' columns
            
        Returns:
            Correlation matrix DataFrame or None if insufficient data
        """
        if not is_correlation_enabled():
            logger.debug("Correlation analysis disabled by feature flag")
            return None
            
        if not price_data:
            logger.warning("No price data provided for correlation analysis")
            return None
            
        start_time = time.time()
        
        try:
            # Prepare returns data with stable symbol ordering
            returns_dict = {}
            valid_symbols = []
            
            # Sort symbols for stable ordering (idempotency requirement)
            sorted_symbols = sorted(price_data.keys())
            
            for symbol in sorted_symbols:
                df = price_data[symbol]
                
                if df.empty or len(df) < self.min_data_points:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} points")
                    continue
                
                # Ensure required columns exist
                if 'Date' not in df.columns or 'Close' not in df.columns:
                    logger.warning(f"Missing required columns for {symbol}")
                    continue
                
                # Calculate returns
                df_clean = df.copy()
                df_clean['Date'] = pd.to_datetime(df_clean['Date'])
                df_clean = df_clean.sort_values('Date').drop_duplicates(subset=['Date'])
                
                # Filter to lookback period
                if self.lookback_days > 0:
                    cutoff_date = df_clean['Date'].max() - timedelta(days=self.lookback_days)
                    df_clean = df_clean[df_clean['Date'] >= cutoff_date]
                
                if len(df_clean) < self.min_data_points:
                    logger.warning(f"Insufficient recent data for {symbol}: {len(df_clean)} points")
                    continue
                
                # Calculate daily returns
                df_clean['Returns'] = df_clean['Close'].pct_change()
                df_clean = df_clean.dropna(subset=['Returns'])
                
                if len(df_clean) < self.min_data_points:
                    logger.warning(f"Insufficient return data for {symbol} after processing")
                    continue
                
                # Store returns with date as index
                returns_series = df_clean.set_index('Date')['Returns']
                returns_dict[symbol] = returns_series
                valid_symbols.append(symbol)
            
            if len(valid_symbols) < 2:
                logger.warning(f"Need at least 2 valid symbols for correlation, got {len(valid_symbols)}")
                return None
            
            # Create aligned returns DataFrame
            returns_df = pd.DataFrame(returns_dict)
            
            # Remove rows where any symbol has NaN (for proper correlation calculation)
            returns_df_clean = returns_df.dropna()
            
            if len(returns_df_clean) < self.min_data_points:
                logger.warning(f"Insufficient overlapping data points: {len(returns_df_clean)}")
                return None
            
            # Compute Pearson correlation matrix
            correlation_matrix = returns_df_clean.corr(method='pearson')
            
            # Store results
            self.correlation_matrix = correlation_matrix
            self.symbols = valid_symbols
            self.returns_data = returns_df_clean
            
            # Save correlation CSV (AC1)
            self._save_correlation_csv(correlation_matrix)
            
            processing_time = time.time() - start_time
            logger.info(f"Correlation matrix computed for {len(valid_symbols)} symbols, "
                       f"{len(returns_df_clean)} overlapping periods, "
                       f"processing time: {processing_time:.2f}s")
            
            return correlation_matrix
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error computing correlation matrix: {e}, processing time: {processing_time:.2f}s")
            return None
    
    def render_heatmap(self, 
                      correlation_matrix: Optional[pd.DataFrame] = None,
                      title: str = "Portfolio Correlation Heatmap",
                      figsize: Tuple[int, int] = (12, 10),
                      save_path: str = "charts/corr_heatmap.png") -> bool:
        """
        Render correlation heatmap and save to file (AC1)
        
        Args:
            correlation_matrix: Correlation matrix DataFrame (uses stored if None)
            title: Plot title
            figsize: Figure size tuple
            save_path: Path to save the heatmap image
            
        Returns:
            True if heatmap rendered successfully, False otherwise
        """
        if not is_correlation_enabled():
            logger.debug("Correlation heatmap disabled by feature flag")
            return False
            
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for heatmap rendering")
            return False
        
        # Use provided matrix or stored matrix
        matrix = correlation_matrix if correlation_matrix is not None else self.correlation_matrix
        
        if matrix is None or matrix.empty:
            logger.warning("No correlation matrix available for heatmap rendering")
            return False
        
        start_time = time.time()
        
        try:
            # Create charts directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Create figure and axis
            plt.figure(figsize=figsize)
            
            # Create heatmap using seaborn if available, otherwise matplotlib
            if 'sns' in globals():
                # Use seaborn for better-looking heatmap
                mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)  # Mask upper triangle
                
                sns.heatmap(
                    matrix,
                    annot=True,
                    cmap='RdBu_r',  # Red-Blue diverging colormap
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    fmt='.2f',
                    mask=mask,
                    vmin=-1,
                    vmax=1
                )
            else:
                # Use matplotlib imshow
                im = plt.imshow(matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
                
                # Add colorbar
                plt.colorbar(im, shrink=0.8)
                
                # Add correlation values as text
                for i in range(len(matrix)):
                    for j in range(len(matrix.columns)):
                        if i != j:  # Don't show diagonal (always 1.0)
                            text = plt.text(j, i, f'{matrix.iloc[i, j]:.2f}',
                                          ha="center", va="center", color="black", fontsize=8)
                
                # Set ticks and labels
                plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha='right')
                plt.yticks(range(len(matrix.index)), matrix.index)
            
            # Set title and layout
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            plt.figtext(0.99, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', 
                       fontsize=8, alpha=0.7)
            
            # Save the figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            processing_time = time.time() - start_time
            logger.info(f"Correlation heatmap saved to {save_path}, processing time: {processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error rendering correlation heatmap: {e}, processing time: {processing_time:.2f}s")
            plt.close('all')  # Clean up any open figures
            return False
    
    def _save_correlation_csv(self, correlation_matrix: pd.DataFrame):
        """
        Save correlation matrix to CSV file (AC1)
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            csv_path = 'data/correlation.csv'
            
            # Add metadata header
            with open(csv_path, 'w') as f:
                f.write(f"# Correlation Matrix\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Symbols: {len(correlation_matrix)}\n")
                f.write(f"# Lookback Days: {self.lookback_days}\n")
                f.write(f"# Method: Pearson\n")
                f.write("#\n")
            
            # Append correlation matrix
            correlation_matrix.to_csv(csv_path, mode='a')
            
            logger.info(f"Correlation matrix saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving correlation CSV: {e}")
    
    def get_correlation_stats(self) -> Dict[str, Any]:
        """
        Get correlation statistics and summary
        
        Returns:
            Dictionary with correlation statistics
        """
        if self.correlation_matrix is None:
            return {"error": "No correlation matrix available"}
        
        try:
            matrix = self.correlation_matrix.copy()
            
            # Remove diagonal (perfect self-correlations)
            np.fill_diagonal(matrix.values, np.nan)
            
            # Flatten to get all correlation values
            all_corr = matrix.values.flatten()
            all_corr = all_corr[~np.isnan(all_corr)]
            
            if len(all_corr) == 0:
                return {"error": "No valid correlations found"}
            
            stats = {
                'symbol_count': len(self.symbols),
                'symbols': sorted(self.symbols),
                'data_points': len(self.returns_data) if self.returns_data is not None else 0,
                'correlation_stats': {
                    'mean': float(np.mean(all_corr)),
                    'median': float(np.median(all_corr)),
                    'std': float(np.std(all_corr)),
                    'min': float(np.min(all_corr)),
                    'max': float(np.max(all_corr)),
                    'count': len(all_corr)
                }
            }
            
            # Find highest/lowest correlations
            # Get upper triangle indices (avoid duplicates)
            triu_indices = np.triu_indices_from(matrix, k=1)
            triu_values = matrix.values[triu_indices]
            
            if len(triu_values) > 0:
                valid_mask = ~np.isnan(triu_values)
                if np.any(valid_mask):
                    valid_indices = [(triu_indices[0][i], triu_indices[1][i]) 
                                   for i in range(len(triu_values)) if valid_mask[i]]
                    valid_values = triu_values[valid_mask]
                    
                    # Highest correlation
                    max_idx = np.argmax(valid_values)
                    max_i, max_j = valid_indices[max_idx]
                    stats['highest_correlation'] = {
                        'symbols': [matrix.index[max_i], matrix.columns[max_j]],
                        'value': float(valid_values[max_idx])
                    }
                    
                    # Lowest correlation  
                    min_idx = np.argmin(valid_values)
                    min_i, min_j = valid_indices[min_idx]
                    stats['lowest_correlation'] = {
                        'symbols': [matrix.index[min_i], matrix.columns[min_j]],
                        'value': float(valid_values[min_idx])
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing correlation stats: {e}")
            return {"error": str(e)}
    
    def get_symbol_correlations(self, target_symbol: str) -> Dict[str, float]:
        """
        Get correlations for a specific symbol with all others
        
        Args:
            target_symbol: Symbol to get correlations for
            
        Returns:
            Dict mapping other symbols to correlation values
        """
        if self.correlation_matrix is None or target_symbol not in self.correlation_matrix.index:
            return {}
        
        try:
            correlations = {}
            target_row = self.correlation_matrix.loc[target_symbol]
            
            for symbol in target_row.index:
                if symbol != target_symbol:
                    correlations[symbol] = float(target_row[symbol])
            
            # Sort by correlation value (descending)
            return dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting correlations for {target_symbol}: {e}")
            return {}


# Global instance
correlation_analyzer = CorrelationAnalyzer()


# Convenience functions
def compute_portfolio_correlations(price_data: Dict[str, pd.DataFrame], 
                                 lookback_days: int = 252) -> Optional[pd.DataFrame]:
    """
    Compute correlation matrix for portfolio price data
    
    Args:
        price_data: Dict mapping symbol to price DataFrame
        lookback_days: Number of days to look back
        
    Returns:
        Correlation matrix DataFrame or None
    """
    analyzer = CorrelationAnalyzer(lookback_days=lookback_days)
    return analyzer.compute_correlation_matrix(price_data)


def generate_correlation_heatmap(price_data: Dict[str, pd.DataFrame],
                                title: str = "Portfolio Correlation Heatmap",
                                save_path: str = "charts/corr_heatmap.png") -> bool:
    """
    Generate and save correlation heatmap
    
    Args:
        price_data: Dict mapping symbol to price DataFrame
        title: Heatmap title
        save_path: Path to save heatmap
        
    Returns:
        True if successful
    """
    analyzer = CorrelationAnalyzer()
    matrix = analyzer.compute_correlation_matrix(price_data)
    
    if matrix is not None:
        return analyzer.render_heatmap(matrix, title=title, save_path=save_path)
    
    return False


def get_correlation_summary(price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Get correlation analysis summary
    
    Args:
        price_data: Dict mapping symbol to price DataFrame
        
    Returns:
        Summary statistics dictionary
    """
    analyzer = CorrelationAnalyzer()
    matrix = analyzer.compute_correlation_matrix(price_data)
    
    if matrix is not None:
        return analyzer.get_correlation_stats()
    
    return {"error": "Failed to compute correlations"}


def load_correlation_from_csv(csv_path: str = "data/correlation.csv") -> Optional[pd.DataFrame]:
    """
    Load correlation matrix from saved CSV file
    
    Args:
        csv_path: Path to correlation CSV file
        
    Returns:
        Correlation matrix DataFrame or None
    """
    try:
        if not os.path.exists(csv_path):
            logger.warning(f"Correlation CSV not found: {csv_path}")
            return None
        
        # Read CSV, skipping comment lines
        df = pd.read_csv(csv_path, comment='#', index_col=0)
        
        logger.info(f"Loaded correlation matrix from {csv_path}: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading correlation CSV: {e}")
        return None