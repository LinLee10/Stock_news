import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_stock_forecast(ax, ticker: str,
                        hist_df: pd.DataFrame,
                        forecast_df: pd.DataFrame):
    """
    Plot the last 30 trading days of historical prices with 3-day forecast.
    Includes financial styling with price trends and better visualization.
    """
    # Ensure Date columns are datetime
    hist = hist_df.copy()
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist = hist.sort_values('Date').tail(30)  # Show 30 days instead of 10

    fc = forecast_df.copy()
    fc['Date'] = pd.to_datetime(fc['Date'])

    # Plot historical close with trend coloring
    if not hist.empty and len(hist) > 0:
        # Calculate trend for color coding
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        trend_color = '#2E8B57' if end_price >= start_price else '#DC143C'  # Green if up, red if down
        
        ax.plot(
            hist['Date'], hist['Close'],
            label=f'Historical ({len(hist)}d)',
            color=trend_color, linewidth=2, alpha=0.8
        )
        
        # Add price range shading
        ax.fill_between(hist['Date'], hist['Close'], alpha=0.1, color=trend_color)
        
        # Add current price annotation
        if len(hist) > 0:
            current_price = hist['Close'].iloc[-1]
            current_date = hist['Date'].iloc[-1]
            ax.annotate(f'${current_price:.2f}', 
                       xy=(current_date, current_price),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Plot 3-day forecast
    if not fc.empty and len(fc) > 0:
        ax.plot(
            fc['Date'], fc['Forecast_Close'],
            '--o', label='Forecast (3d)',
            color='#FF8C00', linewidth=2.5, markersize=5,
            markerfacecolor='white', markeredgecolor='#FF8C00', markeredgewidth=2
        )
        
        # Add forecast price annotations
        for i, (date, price) in enumerate(zip(fc['Date'], fc['Forecast_Close'])):
            ax.annotate(f'${price:.2f}', 
                       xy=(date, price),
                       xytext=(0, 10 if i % 2 == 0 else -15), textcoords='offset points',
                       fontsize=7, ha='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFE4B5', alpha=0.8))

    # Enhanced formatting
    ax.set_title(f'{ticker} - Price & Forecast', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # Format y-axis for currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))

    # Enhanced legend
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9,
                 fancybox=True, shadow=True)


def create_collage(tickers: list[str],
                   price_data: dict[str, pd.DataFrame],
                   forecast_data: dict[str, pd.DataFrame],
                   title: str,
                   save_path: str):
    """
    Create a professional financial chart collage with enhanced styling.
    Shows historical prices and forecasts for multiple tickers.
    """
    n = len(tickers)
    cols = 2
    rows = math.ceil(n / cols)

    # Enhanced figure size and styling
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4.5), squeeze=False)
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    for ax, tic in zip(axes_flat, tickers):
        hist_df = price_data.get(tic, pd.DataFrame(columns=['Date', 'Close']))
        fc_df = forecast_data.get(tic, pd.DataFrame(columns=['Date', 'Forecast_Close']))
        
        # Set background color for each subplot
        ax.set_facecolor('#FAFAFA')
        
        plot_stock_forecast(ax, tic, hist_df, fc_df)
        
        # Add border around each subplot
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1)

    # Remove any extra axes
    for ax in axes_flat[n:]:
        fig.delaxes(ax)

    # Enhanced title with timestamp and styling
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f'{title}\nGenerated: {timestamp}', 
                fontsize=16, fontweight='bold', y=0.98)

    # Improved layout
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    # Save with high quality
    fig.savefig(save_path, dpi=200, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Log creation details
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Created {title} chart with {n} tickers: {', '.join(tickers)}")
    
    return save_path


# BEGIN F07 - Chart upgrades for analytics/alerts/benchmarks

def plot_benchmark_comparison(ax, benchmark_data: dict, portfolio_performance: dict = None):
    """
    Plot benchmark performance comparison with portfolio overlay.
    
    Args:
        ax: Matplotlib axis
        benchmark_data: Dict with benchmark symbols -> performance data
        portfolio_performance: Optional portfolio performance data
    """
    if not benchmark_data:
        ax.text(0.5, 0.5, 'No benchmark data available', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10, alpha=0.6)
        return
    
    # Prepare data for plotting
    periods = ['1M', '3M', '1Y']
    benchmark_names = {'^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones'}
    
    x_pos = range(len(periods))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot each benchmark
    for i, (symbol, data) in enumerate(benchmark_data.items()):
        if i >= len(colors):
            break
            
        label = benchmark_names.get(symbol, symbol)
        values = [data.get(period, 0) for period in periods]
        
        ax.bar([x + width * i for x in x_pos], values, width, 
              label=label, color=colors[i], alpha=0.8)
    
    # Add portfolio performance if available
    if portfolio_performance:
        portfolio_values = [portfolio_performance.get(period, 0) for period in periods]
        ax.bar([x + width * len(benchmark_data) for x in x_pos], portfolio_values, 
              width, label='Portfolio', color='#ff6b6b', alpha=0.9, 
              edgecolor='darkred', linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Period', fontsize=9)
    ax.set_ylabel('Returns (%)', fontsize=9)
    ax.set_title('Benchmark Performance Comparison', fontsize=11, fontweight='bold')
    ax.set_xticks([x + width * (len(benchmark_data) - 1) / 2 for x in x_pos])
    ax.set_xticklabels(periods)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    
    # Color-code positive/negative returns
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))


def plot_portfolio_metrics(ax, portfolio_analytics: dict):
    """
    Plot portfolio sector allocation and key metrics.
    
    Args:
        ax: Matplotlib axis  
        portfolio_analytics: Dict with sector allocation and beta stats
    """
    sector_allocation = portfolio_analytics.get('sector_allocation', {})
    
    if not sector_allocation:
        ax.text(0.5, 0.5, 'No sector data available', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10, alpha=0.6)
        return
    
    # Create pie chart for sector allocation
    sectors = list(sector_allocation.keys())
    sizes = [allocation * 100 for allocation in sector_allocation.values()]
    
    # Color palette for sectors
    colors = plt.cm.Set3(range(len(sectors)))
    
    wedges, texts, autotexts = ax.pie(sizes, labels=sectors, colors=colors,
                                     autopct='%1.1f%%', startangle=90,
                                     textprops={'fontsize': 8})
    
    # Enhance text formatting
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(7)
    
    ax.set_title('Portfolio Sector Allocation', fontsize=11, fontweight='bold')
    
    # Add metrics summary as text
    beta_stats = portfolio_analytics.get('beta_stats', {})
    if beta_stats:
        # Calculate average portfolio beta
        betas = [stats.get('beta', 1.0) for stats in beta_stats.values()]
        avg_beta = sum(betas) / len(betas) if betas else 1.0
        
        metrics_text = f'Avg Portfolio Beta: {avg_beta:.2f}\nPositions: {len(beta_stats)}'
        ax.text(1.3, 0.5, metrics_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))


def plot_alert_summary(ax, smart_alerts: list):
    """
    Plot smart alerts summary visualization.
    
    Args:
        ax: Matplotlib axis
        smart_alerts: List of SmartAlert objects
    """
    if not smart_alerts:
        ax.text(0.5, 0.5, 'No alerts triggered today', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10, alpha=0.6, style='italic')
        return
    
    # Count alerts by severity and type
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    type_counts = {'price_move': 0, 'sentiment_swing': 0, 'earnings_proximity': 0}
    
    for alert in smart_alerts:
        severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
    
    # Create stacked bar chart
    severities = list(severity_counts.keys())
    counts = list(severity_counts.values())
    
    # Color mapping for severities
    severity_colors = {
        'CRITICAL': '#8B0000',  # Dark red
        'HIGH': '#FF4500',      # Orange red  
        'MEDIUM': '#FF8C00',    # Dark orange
        'LOW': '#32CD32'        # Lime green
    }
    
    colors = [severity_colors[sev] for sev in severities]
    
    bars = ax.bar(severities, counts, color=colors, alpha=0.8)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_title(f'Smart Alerts Summary ({len(smart_alerts)} total)', 
                fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=9)
    ax.set_xlabel('Severity Level', fontsize=9)
    
    # Add alert type breakdown as text
    type_text = []
    for alert_type, count in type_counts.items():
        if count > 0:
            type_label = alert_type.replace('_', ' ').title()
            type_text.append(f'{type_label}: {count}')
    
    if type_text:
        breakdown_text = 'Alert Types:\n' + '\n'.join(type_text)
        ax.text(0.98, 0.98, breakdown_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))


def create_enhanced_collage(tickers: list[str],
                           price_data: dict[str, pd.DataFrame],
                           forecast_data: dict[str, pd.DataFrame],
                           title: str,
                           save_path: str,
                           portfolio_analytics: dict = None,
                           benchmark_data: dict = None,
                           smart_alerts: list = None,
                           finbert_results: dict = None):
    """
    Create an enhanced financial chart collage with optional analytics panes.
    
    Args:
        tickers: List of stock symbols
        price_data: Dict mapping symbol -> price DataFrame
        forecast_data: Dict mapping symbol -> forecast DataFrame  
        title: Chart title
        save_path: Path to save the chart
        portfolio_analytics: Optional F05 portfolio analytics data
        benchmark_data: Optional benchmark performance data
        smart_alerts: Optional F06 smart alerts list
        finbert_results: Optional F04 FinBERT analysis results
    """
    from config.feature_flags import (is_portfolio_analytics_enabled, 
                                     is_smart_alerts_enabled, 
                                     is_finbert_pipeline_enabled)
    
    # Determine layout based on enabled features
    has_portfolio = portfolio_analytics and is_portfolio_analytics_enabled()
    has_benchmarks = benchmark_data and is_portfolio_analytics_enabled()
    has_alerts = smart_alerts and is_smart_alerts_enabled() 
    has_finbert = finbert_results and is_finbert_pipeline_enabled()
    
    # Calculate enhanced layout
    n_stocks = len(tickers)
    stock_cols = 2
    stock_rows = math.ceil(n_stocks / stock_cols)
    
    # Add extra rows for optional panes
    extra_panes = sum([bool(has_portfolio), bool(has_benchmarks), bool(has_alerts)])
    if extra_panes > 0:
        extra_rows = 1  # Add one row for optional panes
        total_cols = max(stock_cols, extra_panes)
        total_rows = stock_rows + extra_rows
    else:
        total_cols = stock_cols
        total_rows = stock_rows
    
    # Create figure with enhanced layout
    fig = plt.figure(figsize=(7 * total_cols, 4.5 * total_rows))
    gs = fig.add_gridspec(total_rows, total_cols, hspace=0.3, wspace=0.3)
    fig.patch.set_facecolor('white')
    
    # Plot stock charts in standard positions
    for i, ticker in enumerate(tickers):
        row = i // stock_cols
        col = i % stock_cols
        ax = fig.add_subplot(gs[row, col])
        
        hist_df = price_data.get(ticker, pd.DataFrame(columns=['Date', 'Close']))
        fc_df = forecast_data.get(ticker, pd.DataFrame(columns=['Date', 'Forecast_Close']))
        
        ax.set_facecolor('#FAFAFA')
        plot_stock_forecast(ax, ticker, hist_df, fc_df)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1)
    
    # Add optional panes in the extra row
    if extra_panes > 0:
        pane_col = 0
        
        # Portfolio analytics pane
        if has_portfolio:
            ax = fig.add_subplot(gs[stock_rows, pane_col])
            ax.set_facecolor('#F8F8FF')
            plot_portfolio_metrics(ax, portfolio_analytics)
            pane_col += 1
        
        # Benchmark comparison pane  
        if has_benchmarks:
            ax = fig.add_subplot(gs[stock_rows, pane_col])
            ax.set_facecolor('#FFF8F8')
            portfolio_perf = portfolio_analytics.get('portfolio_performance') if portfolio_analytics else None
            plot_benchmark_comparison(ax, benchmark_data, portfolio_perf)
            pane_col += 1
            
        # Smart alerts pane
        if has_alerts:
            ax = fig.add_subplot(gs[stock_rows, pane_col])
            ax.set_facecolor('#F8FFF8')
            plot_alert_summary(ax, smart_alerts)
            pane_col += 1
    
    # Enhanced title with feature indicators
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    features = []
    if has_portfolio:
        features.append("Portfolio Analytics")
    if has_benchmarks:
        features.append("Benchmarks") 
    if has_alerts:
        features.append("Alerts")
    if has_finbert:
        features.append("FinBERT")
    
    subtitle = f"Features: {', '.join(features)}" if features else "Standard View"
    full_title = f'{title}\n{subtitle} | Generated: {timestamp}'
    
    fig.suptitle(full_title, fontsize=14, fontweight='bold', y=0.98)
    
    # Save with high quality
    fig.savefig(save_path, dpi=200, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Enhanced logging
    import logging
    logger = logging.getLogger(__name__)
    feature_count = len(features)
    logger.info(f"F07: Created enhanced {title} with {n_stocks} stocks + {feature_count} feature panes")
    
    return save_path

# END F07
