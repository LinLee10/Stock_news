import os
import logging
import smtplib
import pandas as pd

from email.message import EmailMessage
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger("email_report")
logging.basicConfig(level=logging.INFO)


# BEGIN F07 - Renderer helper functions

def render_enhanced_benchmark_section(benchmark_performance: dict, portfolio_analytics: dict) -> str:
    """Render enhanced benchmark comparison section with visual improvements."""
    html = "<h3 style='margin-top: 25px; color: #2c3e50;'>📊 Enhanced Benchmark Analysis</h3>"
    html += "<p style='font-size: 12px; color: #7f8c8d; margin-bottom: 15px;'>Advanced comparison with portfolio performance overlay</p>"
    
    # Create enhanced benchmark table
    html += (
        "<div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 15px; border-radius: 8px; margin-bottom: 15px;'>"
        "<table style='border-collapse: collapse; width: 100%; background: white; border-radius: 5px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
        "<thead>"
        "<tr style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>"
        "<th style='padding: 10px; text-align: left; font-size: 11px; font-weight: bold;'>Benchmark</th>"
        "<th style='padding: 10px; text-align: center; font-size: 11px; font-weight: bold;'>1M</th>"
        "<th style='padding: 10px; text-align: center; font-size: 11px; font-weight: bold;'>3M</th>"
        "<th style='padding: 10px; text-align: center; font-size: 11px; font-weight: bold;'>1Y</th>"
        "<th style='padding: 10px; text-align: center; font-size: 11px; font-weight: bold;'>Trend</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
    )
    
    benchmark_names = {'^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones'}
    
    for i, (benchmark, perf) in enumerate(benchmark_performance.items()):
        display_name = benchmark_names.get(benchmark, benchmark)
        row_bg = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        
        html += f"<tr style='background-color: {row_bg};'>"
        html += f"<td style='padding: 8px; font-size: 11px; font-weight: bold;'>{display_name}</td>"
        
        for period in ['1M', '3M', '1Y']:
            ret = perf.get(period)
            if ret is not None:
                color = '#28a745' if ret > 0 else '#dc3545' if ret < 0 else '#6c757d'
                arrow = '📈' if ret > 0 else '📉' if ret < 0 else '➡️'
                html += f"<td style='padding: 8px; text-align: center; color: {color}; font-weight: bold; font-size: 11px;'>{ret:+.1f}%</td>"
            else:
                html += "<td style='padding: 8px; text-align: center; color: #adb5bd; font-size: 11px;'>N/A</td>"
        
        # Add trend indicator
        returns_list = [perf.get(p) for p in ['1M', '3M', '1Y'] if perf.get(p) is not None]
        if len(returns_list) >= 2:
            trend = '🔥' if all(r > 0 for r in returns_list) else '❄️' if all(r < 0 for r in returns_list) else '⚖️'
        else:
            trend = '❓'
        html += f"<td style='padding: 8px; text-align: center; font-size: 14px;'>{trend}</td>"
        html += "</tr>"
    
    # Add portfolio row if available
    portfolio_performance = portfolio_analytics.get('portfolio_performance', {})
    if portfolio_performance:
        html += "<tr style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-top: 2px solid #fd7e14;'>"
        html += "<td style='padding: 8px; font-size: 11px; font-weight: bold; color: #d63384;'>🎯 Your Portfolio</td>"
        
        for period in ['1M', '3M', '1Y']:
            ret = portfolio_performance.get(period)
            if ret is not None:
                color = '#28a745' if ret > 0 else '#dc3545' if ret < 0 else '#6c757d'
                html += f"<td style='padding: 8px; text-align: center; color: {color}; font-weight: bold; font-size: 11px;'>{ret:+.1f}%</td>"
            else:
                html += "<td style='padding: 8px; text-align: center; color: #adb5bd; font-size: 11px;'>N/A</td>"
        
        # Portfolio trend
        portfolio_returns = [portfolio_performance.get(p) for p in ['1M', '3M', '1Y'] if portfolio_performance.get(p) is not None]
        if len(portfolio_returns) >= 2:
            portfolio_trend = '🚀' if all(r > 0 for r in portfolio_returns) else '🔻' if all(r < 0 for r in portfolio_returns) else '🎯'
        else:
            portfolio_trend = '❓'
        html += f"<td style='padding: 8px; text-align: center; font-size: 14px;'>{portfolio_trend}</td>"
        html += "</tr>"
    
    html += "</tbody></table></div>"
    
    return html


def render_enhanced_alerts_section(smart_alerts: list) -> str:
    """Render enhanced smart alerts section with priority grouping."""
    if not smart_alerts:
        return ""
    
    html = "<h3 style='margin-top: 25px; color: #e74c3c;'>⚠️ Enhanced Alert Dashboard</h3>"
    html += f"<p style='font-size: 12px; color: #7f8c8d; margin-bottom: 15px;'>Comprehensive alert analysis • {len(smart_alerts)} alerts triggered today</p>"
    
    # Group alerts by severity
    alerts_by_severity = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for alert in smart_alerts:
        alerts_by_severity[alert.severity].append(alert)
    
    # Alert summary dashboard
    html += "<div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%); padding: 15px; border-radius: 8px; margin-bottom: 15px;'>"
    html += "<div style='display: flex; justify-content: space-around; text-align: center;'>"
    
    severity_config = {
        'CRITICAL': {'color': '#721c24', 'bg': '#f8d7da', 'icon': '🚨'},
        'HIGH': {'color': '#856404', 'bg': '#fff3cd', 'icon': '⚠️'},
        'MEDIUM': {'color': '#0c5460', 'bg': '#d1ecf1', 'icon': '📈'},
        'LOW': {'color': '#155724', 'bg': '#d4edda', 'icon': '💡'}
    }
    
    for severity, config in severity_config.items():
        count = len(alerts_by_severity[severity])
        html += f"""
        <div style='background: {config['bg']}; padding: 10px; border-radius: 5px; min-width: 80px; margin: 0 5px;'>
            <div style='font-size: 20px;'>{config['icon']}</div>
            <div style='font-size: 16px; font-weight: bold; color: {config['color']};'>{count}</div>
            <div style='font-size: 10px; color: {config['color']}; text-transform: uppercase;'>{severity}</div>
        </div>
        """
    html += "</div></div>"
    
    # Top priority alerts detail
    priority_alerts = alerts_by_severity['CRITICAL'] + alerts_by_severity['HIGH']
    if priority_alerts:
        html += "<h4 style='color: #c0392b; margin-top: 15px; margin-bottom: 10px;'>🎯 Priority Alerts Requiring Attention</h4>"
        
        for alert in priority_alerts[:5]:  # Show top 5 priority alerts
            severity_color = severity_config[alert.severity]['color']
            severity_bg = severity_config[alert.severity]['bg']
            severity_icon = severity_config[alert.severity]['icon']
            
            html += f"""
            <div style='background: {severity_bg}; border-left: 4px solid {severity_color}; padding: 10px; margin: 8px 0; border-radius: 0 5px 5px 0;'>
                <div style='display: flex; align-items: center; margin-bottom: 5px;'>
                    <span style='font-size: 16px; margin-right: 8px;'>{severity_icon}</span>
                    <strong style='color: {severity_color}; font-size: 12px;'>{alert.symbol}</strong>
                    <span style='background: {severity_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 9px; margin-left: 8px; font-weight: bold;'>{alert.severity}</span>
                </div>
                <div style='font-size: 11px; color: #2c3e50; margin-bottom: 3px;'>{alert.description}</div>
                <div style='font-size: 10px; color: #7f8c8d; font-style: italic;'>💡 {alert.guidance}</div>
            </div>
            """
    
    return html


def render_enhanced_finbert_section(finbert_results: dict) -> str:
    """Render enhanced FinBERT section with sentiment visualization."""
    if not finbert_results:
        return ""
    
    html = "<h3 style='margin-top: 25px; color: #8e44ad;'>🧠 Enhanced FinBERT AI Analysis</h3>"
    html += "<p style='font-size: 12px; color: #7f8c8d; margin-bottom: 15px;'>Advanced sentiment analysis with confidence scoring and trend analysis</p>"
    
    # Sentiment overview
    html += "<div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 15px; border-radius: 8px; margin-bottom: 15px;'>"
    
    # Calculate aggregate sentiment metrics
    total_analyzed = len(finbert_results)
    action_counts = {'strong_buy': 0, 'buy': 0, 'hold': 0, 'sell': 0, 'strong_sell': 0}
    total_confidence = 0
    total_conviction = 0
    
    for ticker, data in finbert_results.items():
        recommendation = data.get('recommendation')
        if recommendation:
            action_counts[recommendation.action.value] += 1
            total_confidence += recommendation.confidence
            total_conviction += recommendation.conviction_score
    
    avg_confidence = (total_confidence / total_analyzed) if total_analyzed > 0 else 0
    avg_conviction = (total_conviction / total_analyzed) if total_analyzed > 0 else 0
    
    html += f"""
    <div style='text-align: center; color: #2c3e50;'>
        <div style='font-size: 14px; font-weight: bold; margin-bottom: 8px;'>Portfolio Sentiment Overview</div>
        <div style='display: flex; justify-content: center; gap: 20px; margin-bottom: 10px;'>
            <div style='text-align: center;'>
                <div style='font-size: 20px; font-weight: bold; color: #e74c3c;'>{total_analyzed}</div>
                <div style='font-size: 10px; text-transform: uppercase;'>Analyzed</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 20px; font-weight: bold; color: #3498db;'>{avg_confidence:.0f}%</div>
                <div style='font-size: 10px; text-transform: uppercase;'>Avg Confidence</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 20px; font-weight: bold; color: #9b59b6;'>{avg_conviction:.1f}/10</div>
                <div style='font-size: 10px; text-transform: uppercase;'>Avg Conviction</div>
            </div>
        </div>
    </div>
    """
    html += "</div>"
    
    # Action distribution
    action_colors = {
        'strong_buy': '#27ae60',
        'buy': '#2ecc71', 
        'hold': '#f39c12',
        'sell': '#e67e22',
        'strong_sell': '#e74c3c'
    }
    
    html += "<h4 style='color: #8e44ad; margin-bottom: 10px;'>📊 Recommendation Distribution</h4>"
    html += "<div style='display: flex; justify-content: space-around; margin-bottom: 15px;'>"
    
    for action, count in action_counts.items():
        if count > 0:
            color = action_colors.get(action, '#95a5a6')
            action_label = action.replace('_', ' ').title()
            percentage = (count / total_analyzed * 100) if total_analyzed > 0 else 0
            
            html += f"""
            <div style='text-align: center; padding: 8px;'>
                <div style='width: 40px; height: 40px; border-radius: 50%; background: {color}; 
                           display: flex; align-items: center; justify-content: center; margin: 0 auto 5px;
                           color: white; font-weight: bold; font-size: 12px;'>{count}</div>
                <div style='font-size: 10px; color: #2c3e50;'>{action_label}</div>
                <div style='font-size: 9px; color: #7f8c8d;'>{percentage:.0f}%</div>
            </div>
            """
    
    html += "</div>"
    
    return html


def render_performance_footer() -> str:
    """Render performance metrics footer with generation timestamp."""
    from datetime import datetime
    import time
    
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    html = f"""
    <div style='margin-top: 30px; padding: 15px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
               border-radius: 8px; border-top: 3px solid #667eea;'>
        <h4 style='color: #2c3e50; margin-bottom: 10px; font-size: 13px;'>📈 Report Generation Metrics</h4>
        <div style='display: flex; justify-content: space-between; font-size: 11px; color: #34495e;'>
            <div>
                <strong>Generated:</strong> {generation_time}
            </div>
            <div>
                <strong>Features:</strong> F04 FinBERT • F05 Analytics • F06 Alerts • F07 Enhanced
            </div>
            <div>
                <strong>Version:</strong> Stonk News v2.0
            </div>
        </div>
        <div style='margin-top: 8px; font-size: 10px; color: #7f8c8d; text-align: center;'>
            🤖 Enhanced with F07 Report Upgrades • Professional Financial Analysis Pipeline
        </div>
    </div>
    """
    
    return html

# END F07

# ─── Load SMTP credentials ────────────────────────────────
load_dotenv("config/secrets.env")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_TO  = os.getenv("EMAIL_TO")

def send_report(
    watchlist:         list[str],
    portfolio:         list[str],
    head7:             dict,      # 7-day mention data
    head30:            dict,      # 30-day sentiment data
    preds:             dict,
    portfolio_collage: str = None,
    watchlist_collage: str = None,
    out_path:          str = "report.html",
    top_mentioned:     list[str] = None,
    finbert_results:   dict = None,  # F04: FinBERT analysis results
    portfolio_analytics: dict = None,  # F05: Portfolio analytics results
    smart_alerts:      list = None,   # F06: Smart alerts results
    earnings_schedule: pd.DataFrame = None  # F16: Earnings schedule
):
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    # BEGIN F15_DISCLAIMER
    # Get disclaimer text from environment or use default
    disclaimer_text = os.getenv('DISCLAIMER_TEXT', 'This report is for research and educational purposes only. Not financial advice.')
    company_name = os.getenv('COMPANY_NAME', 'Stonk News')
    
    disclaimer_html = f"""
    <div style='background-color: #f8f9fa; border-left: 4px solid #ffc107; padding: 12px; margin: 16px 0; border-radius: 4px;'>
        <div style='font-size: 12px; color: #6c757d; font-weight: bold;'>⚠️ DISCLAIMER</div>
        <div style='font-size: 11px; color: #495057; margin-top: 4px;'>{disclaimer_text}</div>
    </div>
    """
    # END F15_DISCLAIMER
    
    html = f"<html><body><div style='font-family: Arial, sans-serif; max-width:600px; margin:auto;'>"
    html += disclaimer_html
    html += f"<h2>Stock News Forecast Report — {now}</h2>"

    #
    # 30-Day Sentiment — Portfolio
    #
    html += "<h3>30-Day Sentiment — Portfolio</h3>"
    html += (
        "<table style='border-collapse: collapse; margin-bottom: 1em;'>"
        "<tr>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:left;'>Ticker</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Avg_Sentiment</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Count_Positive</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Count_Negative</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Count_Neutral</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Total_Headlines</th>"
        "</tr>"
    )
    for t in portfolio:
        info       = head30.get(t, {})
        total      = info.get('count', 0)
        pos        = info.get('count_positive', 0)
        neg        = info.get('count_negative', 0)
        neu        = info.get('count_neutral', 0)
        dsent_dict = info.get('daily_sentiment', {})
        avg_sent   = (sum(dsent_dict.values()) / len(dsent_dict)) if dsent_dict else 0.0

        html += (
            f"<tr>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:left;'>{t}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{avg_sent:.2f}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{pos}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{neg}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{neu}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{total}</td>"
            f"</tr>"
        )
    html += "</table>"

    #
    # 30-Day Sentiment — Watchlist
    #
    html += "<h3>30-Day Sentiment — Watchlist</h3>"
    html += (
        "<table style='border-collapse: collapse; margin-bottom: 1em;'>"
        "<tr>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:left;'>Ticker</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Avg_Sentiment</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Count_Positive</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Count_Negative</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Count_Neutral</th>"
        "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Total_Headlines</th>"
        "</tr>"
    )
    for t in watchlist:
        info       = head30.get(t, {})
        total      = info.get('count', 0)
        pos        = info.get('count_positive', 0)
        neg        = info.get('count_negative', 0)
        neu        = info.get('count_neutral', 0)
        dsent_dict = info.get('daily_sentiment', {})
        avg_sent   = (sum(dsent_dict.values()) / len(dsent_dict)) if dsent_dict else 0.0

        html += (
            f"<tr>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:left;'>{t}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{avg_sent:.2f}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{pos}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{neg}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{neu}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{total}</td>"
            f"</tr>"
        )
    html += "</table>"

    #
    # Watchlist Forecasts
    #
    html += "<h3>Watchlist Forecasts</h3>"
    html += (
      "<table style='border-collapse: collapse; margin-bottom: 1em;'>"
      "<tr>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:left;'>Ticker</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Confidence</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:center;'>RedFlag</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>10-Day %</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:left;'>3-Day Forecast</th>"
      "</tr>"
    )
    for t in watchlist:
        p = preds[t]
        perf_val = p.get('perf_10d')
        perf_str = f"{perf_val:.2f}%" if perf_val is not None else "N/A"
        color = "black"
        if perf_val is not None:
            color = "green" if perf_val >= 0 else "red"
        preds_str = ", ".join(str(x) for x in p['predictions'])
        html += (
          f"<tr>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:left;'>{t}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{p['confidence']:.2f}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:center;'>{p['red_flag']}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right; color:{color};'>{perf_str}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:left;'>{preds_str}</td>"
          f"</tr>"
        )
    html += "</table>"
    if watchlist_collage and os.path.exists(watchlist_collage):
        html += "<p><strong>📈 Watchlist Charts:</strong> See attached watchlist_collage.png for detailed price charts and forecasts.</p>"

    #
    # Portfolio Forecasts
    #
    html += "<h3>Portfolio Forecasts</h3>"
    html += (
      "<table style='border-collapse: collapse; margin-bottom: 1em;'>"
      "<tr>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:left;'>Ticker</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Confidence</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:center;'>RedFlag</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>10-Day %</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:left;'>3-Day Forecast</th>"
      "</tr>"
    )
    for t in portfolio:
        p = preds[t]
        perf_val = p.get('perf_10d')
        perf_str = f"{perf_val:.2f}%" if perf_val is not None else "N/A"
        color = "black"
        if perf_val is not None:
            color = "green" if perf_val >= 0 else "red"
        preds_str = ", ".join(str(x) for x in p['predictions'])
        html += (
          f"<tr>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:left;'>{t}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{p['confidence']:.2f}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:center;'>{p['red_flag']}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right; color:{color};'>{perf_str}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:left;'>{preds_str}</td>"
          f"</tr>"
        )
    html += "</table>"
    if portfolio_collage and os.path.exists(portfolio_collage):
        html += "<p><strong>📊 Portfolio Charts:</strong> See attached portfolio_collage.png for detailed price charts and forecasts.</p>"
    
    # BEGIN F13 optional image
    corr_heatmap_path = "charts/corr_heatmap.png"
    if os.path.exists(corr_heatmap_path):
        html += "<p><strong>📈 Correlation Heatmap:</strong> See attached corr_heatmap.png for portfolio correlation analysis.</p>"

    # BEGIN F16_SECTION
    # Upcoming Earnings section
    if earnings_schedule is not None and not earnings_schedule.empty:
        from config.feature_flags import is_earnings_reads_enabled
        if is_earnings_reads_enabled():
            html += "<h3>📅 Upcoming Earnings</h3>"
            html += (
                "<table style='border-collapse: collapse; margin-bottom: 1em;'>"
                "<tr>"
                "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:left;'>Symbol</th>"
                "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:center;'>Date</th>"
                "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Implied Move</th>"
                "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:center;'>Direction</th>"
                "</tr>"
            )
            
            for _, row in earnings_schedule.iterrows():
                symbol = row['symbol']
                earnings_date = row['earnings_date']
                implied_move = row.get('implied_move_pct')
                direction = row.get('direction', 'Neutral')
                
                # Format earnings date
                if pd.notna(earnings_date):
                    if isinstance(earnings_date, str):
                        date_str = earnings_date[:10]  # Take first 10 chars (YYYY-MM-DD)
                    else:
                        date_str = earnings_date.strftime("%Y-%m-%d")
                else:
                    date_str = "TBD"
                
                # Format implied move
                if pd.notna(implied_move) and implied_move is not None:
                    move_str = f"{implied_move:.1f}%"
                    move_color = "green" if implied_move > 0 else "black"
                else:
                    move_str = "N/A"
                    move_color = "#adb5bd"
                
                # Direction color coding
                direction_color = {
                    'Bullish': '#28a745',
                    'Bearish': '#dc3545', 
                    'Neutral': '#6c757d'
                }.get(direction, '#6c757d')
                
                html += (
                    f"<tr>"
                    f"<td style='padding:6px; border:1px solid #ddd; text-align:left; font-weight:bold;'>{symbol}</td>"
                    f"<td style='padding:6px; border:1px solid #ddd; text-align:center;'>{date_str}</td>"
                    f"<td style='padding:6px; border:1px solid #ddd; text-align:right; color:{move_color};'>{move_str}</td>"
                    f"<td style='padding:6px; border:1px solid #ddd; text-align:center; color:{direction_color};'>{direction}</td>"
                    f"</tr>"
                )
            
            html += "</table>"
            html += "<p style='font-size:11px; color:#6c757d; margin-top:8px;'>💡 Implied moves are based on options pricing. Past performance does not guarantee future results.</p>"
    # END F16_SECTION

    #
    # 7-Day Mention Leaders & Headlines
    #
    html += "<h3>7-Day Mention Leaders & Headlines</h3>"
    html += (
      "<table style='border-collapse: collapse; margin-bottom: 1em;'>"
      "<tr>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:left;'>Ticker</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Avg_Sentiment</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Count_Positive</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Count_Negative</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Count_Neutral</th>"
      "<th style='padding:6px; border:1px solid #ddd; background-color:#f0f0f0; text-align:right;'>Total_Headlines</th>"
      "</tr>"
    )
    recs = []
    for t, info in head7.items():
        total   = info.get('count', 0)
        pos     = info.get('count_positive', 0)
        neg     = info.get('count_negative', 0)
        neu     = info.get('count_neutral', 0)
        dsent   = info.get('daily_sentiment', {})
        avg_s   = (sum(dsent.values()) / len(dsent)) if dsent else 0.0
        headlines = info.get('headlines', [])
        recs.append((t, avg_s, pos, neg, neu, total, headlines))

    recs.sort(key=lambda x: x[5], reverse=True)
    top10 = recs[:10]

    for t, avg_s, pos, neg, neu, total, headlines in top10:
        html += (
          f"<tr>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:left;'>{t}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{avg_s:.2f}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{pos}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{neg}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{neu}</td>"
            f"<td style='padding:6px; border:1px solid #ddd; text-align:right;'>{total}</td>"
          f"</tr>"
        )
    html += "</table>"

    # Top 10 Most Mentioned Stocks
    if top_mentioned:
        html += "<h3>Top 10 Most Mentioned Stocks (Past 7 Days)</h3>"
        html += "<p style='font-size: 14px; color: #666;'>"
        html += "<strong>Most Active:</strong> " + ", ".join(top_mentioned[:5]) + "<br>"
        html += "<strong>Also Trending:</strong> " + ", ".join(top_mentioned[5:]) if len(top_mentioned) > 5 else ""
        html += "</p>"

    # Top 3 headlines per ticker
    for t, avg_s, pos, neg, neu, total, headlines in top10:
        html += f"<p><strong>{t} Headlines (Top 3):</strong></p><ul>"
        for title, link, date in headlines[:3]:
            safe = title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            html += f"<li><a href='{link}' target='_blank'>{safe}</a> ({date})</li>"
        html += "</ul>"

    # BEGIN F04 - FinBERT Summary section
    if finbert_results:
        from config.feature_flags import is_finbert_pipeline_enabled, is_finbert_backtest_enabled
        
        html += "<h3>FinBERT AI Sentiment Analysis</h3>"
        html += "<p style='font-size: 12px; color: #888;'>Advanced AI-powered financial sentiment analysis with investment recommendations.</p>"
        
        # Create FinBERT results table
        html += (
            "<table style='border-collapse: collapse; margin-bottom: 1em; width: 100%;'>"
            "<tr>"
            "<th style='padding:4px; border:1px solid #ddd; background-color:#e6f3ff; text-align:left; font-size:12px;'>Ticker</th>"
            "<th style='padding:4px; border:1px solid #ddd; background-color:#e6f3ff; text-align:left; font-size:12px;'>Action</th>"
            "<th style='padding:4px; border:1px solid #ddd; background-color:#e6f3ff; text-align:right; font-size:12px;'>Confidence</th>"
            "<th style='padding:4px; border:1px solid #ddd; background-color:#e6f3ff; text-align:right; font-size:12px;'>Conviction</th>"
            "<th style='padding:4px; border:1px solid #ddd; background-color:#e6f3ff; text-align:right; font-size:12px;'>Sentiment</th>"
        )
        
        # Add backtest columns if enabled
        if is_finbert_backtest_enabled():
            html += (
                "<th style='padding:4px; border:1px solid #ddd; background-color:#ffe6e6; text-align:center; font-size:12px;'>Backtest</th>"
            )
        
        html += "</tr>"
        
        # Add FinBERT data rows
        for ticker, data in finbert_results.items():
            recommendation = data.get('recommendation')
            backtest = data.get('backtest', {})
            article_count = data.get('article_count', 0)
            
            html += f"<tr>"
            html += f"<td style='padding:4px; border:1px solid #ddd; font-size:12px;'>{ticker}</td>"
            
            if recommendation:
                # Color code the action
                action = recommendation.action.value
                action_color = {
                    'strong_buy': '#006400',    # Dark green
                    'buy': '#228B22',           # Forest green  
                    'hold': '#666666',          # Gray
                    'sell': '#FF4500',          # Orange red
                    'strong_sell': '#8B0000'    # Dark red
                }.get(action, '#666666')
                
                html += f"<td style='padding:4px; border:1px solid #ddd; font-size:12px; color:{action_color}; font-weight:bold;'>{action.replace('_', ' ').title()}</td>"
                html += f"<td style='padding:4px; border:1px solid #ddd; font-size:12px; text-align:right;'>{recommendation.confidence:.1f}%</td>"
                html += f"<td style='padding:4px; border:1px solid #ddd; font-size:12px; text-align:right;'>{recommendation.conviction_score:.1f}/10</td>"
                html += f"<td style='padding:4px; border:1px solid #ddd; font-size:12px; text-align:right;'>{recommendation.sentiment_score:+.2f}</td>"
            else:
                html += "<td style='padding:4px; border:1px solid #ddd; font-size:12px; color:#999;'>N/A</td>"
                html += "<td style='padding:4px; border:1px solid #ddd; font-size:12px; color:#999;'>N/A</td>"
                html += "<td style='padding:4px; border:1px solid #ddd; font-size:12px; color:#999;'>N/A</td>"
                html += "<td style='padding:4px; border:1px solid #ddd; font-size:12px; color:#999;'>N/A</td>"
            
            # Add backtest summary if enabled
            if is_finbert_backtest_enabled():
                if backtest and 'deltas' in backtest:
                    deltas = backtest['deltas']
                    lambda_effect = deltas.get('lambda_effect_29d', {})
                    window_effect = deltas.get('window_effect_lambda_0.2', {})
                    
                    backtest_summary = []
                    if lambda_effect.get('action_difference'):
                        backtest_summary.append("λ affects action")
                    if window_effect.get('action_difference'):  
                        backtest_summary.append("window affects action")
                    
                    summary_text = ", ".join(backtest_summary) if backtest_summary else "consistent"
                    html += f"<td style='padding:4px; border:1px solid #ddd; font-size:11px; text-align:center;'>{summary_text}</td>"
                else:
                    html += "<td style='padding:4px; border:1px solid #ddd; font-size:11px; text-align:center; color:#999;'>N/A</td>"
            
            html += "</tr>"
        
        html += "</table>"
        
        # Add configuration info
        from config.config import FINBERT_LAMBDA, FINBERT_BARRIER_DAYS
        html += f"<p style='font-size: 11px; color: #888; margin-top: 10px;'>"
        html += f"Configuration: λ={FINBERT_LAMBDA}, Barrier Window={FINBERT_BARRIER_DAYS} days. "
        
        if is_finbert_backtest_enabled():
            html += "Backtest compares λ=0.2 vs 0.94 and 29-day vs 90-day windows."
        
        html += "</p>"
    # END F04
    
    # BEGIN F05 - Portfolio Analytics sections
    if portfolio_analytics and not portfolio_analytics.get('error'):
        from config.feature_flags import is_portfolio_analytics_enabled
        
        if is_portfolio_analytics_enabled():
            html += "<h3>Portfolio Analytics & Benchmarks</h3>"
            html += "<p style='font-size: 12px; color: #888;'>Sector allocation, beta analysis, and benchmark comparison.</p>"
            
            # Sector Allocation section
            sector_allocation = portfolio_analytics.get('sector_allocation', {})
            if sector_allocation:
                html += "<h4 style='margin-top: 15px; margin-bottom: 5px; font-size: 14px; color: #333;'>Sector Allocation</h4>"
                html += (
                    "<table style='border-collapse: collapse; margin-bottom: 10px; width: 100%;'>"
                    "<tr>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:left; font-size:11px;'>Sector</th>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:right; font-size:11px;'>Weight %</th>"
                    "</tr>"
                )
                
                for sector, weight in sector_allocation.items():
                    weight_pct = weight * 100
                    html += f"<tr>"
                    html += f"<td style='padding:3px; border:1px solid #ddd; font-size:11px;'>{sector}</td>"
                    html += f"<td style='padding:3px; border:1px solid #ddd; font-size:11px; text-align:right;'>{weight_pct:.1f}%</td>"
                    html += f"</tr>"
                
                html += "</table>"
            
            # Beta Analysis section
            beta_stats = portfolio_analytics.get('beta_stats', {})
            if beta_stats:
                html += "<h4 style='margin-top: 15px; margin-bottom: 5px; font-size: 14px; color: #333;'>Beta vs S&P 500</h4>"
                html += (
                    "<table style='border-collapse: collapse; margin-bottom: 10px; width: 100%;'>"
                    "<tr>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:left; font-size:11px;'>Symbol</th>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:right; font-size:11px;'>Beta</th>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:right; font-size:11px;'>R²</th>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:right; font-size:11px;'>Volatility</th>"
                    "</tr>"
                )
                
                # Show top holdings by beta
                sorted_betas = sorted(beta_stats.items(), key=lambda x: abs(x[1]['beta']), reverse=True)
                for ticker, stats in sorted_betas[:10]:  # Top 10
                    beta = stats['beta']
                    r_squared = stats['r_squared']
                    volatility = stats['ticker_volatility']
                    
                    # Color code beta
                    if beta > 1.2:
                        beta_color = '#cc0000'  # High beta (red)
                    elif beta < 0.8:
                        beta_color = '#006600'  # Low beta (green)  
                    else:
                        beta_color = '#333333'  # Normal beta
                    
                    html += f"<tr>"
                    html += f"<td style='padding:3px; border:1px solid #ddd; font-size:11px;'>{ticker}</td>"
                    html += f"<td style='padding:3px; border:1px solid #ddd; font-size:11px; text-align:right; color:{beta_color};'>{beta:.2f}</td>"
                    html += f"<td style='padding:3px; border:1px solid #ddd; font-size:11px; text-align:right;'>{r_squared:.2f}</td>"
                    html += f"<td style='padding:3px; border:1px solid #ddd; font-size:11px; text-align:right;'>{volatility:.1f}%</td>"
                    html += f"</tr>"
                
                html += "</table>"
            
            # Benchmark Performance section
            benchmark_performance = portfolio_analytics.get('benchmark_performance', {})
            portfolio_performance = portfolio_analytics.get('portfolio_performance', {})
            
            if benchmark_performance:
                html += "<h4 style='margin-top: 15px; margin-bottom: 5px; font-size: 14px; color: #333;'>Benchmark Performance</h4>"
                html += (
                    "<table style='border-collapse: collapse; margin-bottom: 10px; width: 100%;'>"
                    "<tr>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:left; font-size:11px;'>Index</th>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:right; font-size:11px;'>1M</th>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:right; font-size:11px;'>3M</th>"
                    "<th style='padding:3px; border:1px solid #ddd; background-color:#f9f9f9; text-align:right; font-size:11px;'>1Y</th>"
                    "</tr>"
                )
                
                # Add benchmark rows
                benchmark_names = {'^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones'}
                for benchmark, perf in benchmark_performance.items():
                    display_name = benchmark_names.get(benchmark, benchmark)
                    
                    html += f"<tr>"
                    html += f"<td style='padding:3px; border:1px solid #ddd; font-size:11px;'>{display_name}</td>"
                    
                    for period in ['1M', '3M', '1Y']:
                        ret = perf.get(period)
                        if ret is not None:
                            color = '#006600' if ret > 0 else '#cc0000' if ret < 0 else '#666666'
                            html += f"<td style='padding:3px; border:1px solid #ddd; font-size:11px; text-align:right; color:{color};'>{ret:+.1f}%</td>"
                        else:
                            html += f"<td style='padding:3px; border:1px solid #ddd; font-size:11px; text-align:right; color:#999;'>N/A</td>"
                    
                    html += f"</tr>"
                
                html += "</table>"
            
            # Add summary statistics
            f05_metrics = portfolio_analytics.get('f05_metrics', {})
            if f05_metrics:
                tickers_processed = f05_metrics.get('tickers_processed', 0)
                processing_time = f05_metrics.get('processing_time_ms', 0)
                
                html += f"<p style='font-size: 11px; color: #888; margin-top: 10px;'>"
                html += f"Analysis: {tickers_processed} symbols processed in {processing_time:.0f}ms. "
                html += f"Data sources: Yahoo Finance, sector mappings. "
                html += f"Beta calculations use 252-day rolling OLS regression vs S&P 500."
                html += "</p>"
    # END F05

    # BEGIN F06 - Smart Alerts section
    if smart_alerts:
        from config.feature_flags import is_smart_alerts_enabled
        
        if is_smart_alerts_enabled():
            html += "<h3>Smart Alerts</h3>"
            html += "<p style='font-size: 12px; color: #888;'>Automated alerts for price moves, sentiment changes, and earnings proximity.</p>"
            
            if smart_alerts:
                # Group alerts by severity for better presentation
                alerts_by_severity = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}
                for alert in smart_alerts:
                    alerts_by_severity.get(alert.severity, alerts_by_severity['LOW']).append(alert)
                
                # Alert summary
                total_alerts = len(smart_alerts)
                alert_counts = {sev: len(alerts) for sev, alerts in alerts_by_severity.items() if alerts}
                
                if total_alerts > 0:
                    html += f"<p style='font-size: 13px; margin-bottom: 10px;'>"
                    html += f"<strong>Today's Alerts:</strong> {total_alerts} total"
                    if alert_counts:
                        severity_summary = ", ".join(f"{count} {sev.lower()}" for sev, count in alert_counts.items())
                        html += f" ({severity_summary})"
                    html += "</p>"
                
                # Create alerts table
                html += (
                    "<table style='border-collapse: collapse; margin-bottom: 1em; width: 100%;'>"
                    "<tr>"
                    "<th style='padding:4px; border:1px solid #ddd; background-color:#fff2e6; text-align:left; font-size:11px;'>Symbol</th>"
                    "<th style='padding:4px; border:1px solid #ddd; background-color:#fff2e6; text-align:left; font-size:11px;'>Type</th>"
                    "<th style='padding:4px; border:1px solid #ddd; background-color:#fff2e6; text-align:center; font-size:11px;'>Severity</th>"
                    "<th style='padding:4px; border:1px solid #ddd; background-color:#fff2e6; text-align:left; font-size:11px;'>Alert</th>"
                    "</tr>"
                )
                
                # Show alerts sorted by severity (CRITICAL first)
                severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
                alerts_to_show = []
                for severity in severity_order:
                    alerts_to_show.extend(alerts_by_severity[severity])
                
                # Limit to most important alerts (max 15 in email)
                max_alerts_in_email = 15
                displayed_alerts = alerts_to_show[:max_alerts_in_email]
                
                for alert in displayed_alerts:
                    # Severity color coding
                    severity_colors = {
                        'CRITICAL': '#8B0000',  # Dark red
                        'HIGH': '#FF4500',      # Orange red
                        'MEDIUM': '#FF8C00',    # Dark orange
                        'LOW': '#32CD32'        # Lime green
                    }
                    severity_color = severity_colors.get(alert.severity, '#666666')
                    
                    # Alert type icons/labels
                    type_labels = {
                        'price_move': '📈' if alert.change_percent and alert.change_percent > 0 else '📉',
                        'sentiment_swing': '💭',
                        'earnings_proximity': '📊'
                    }
                    type_label = type_labels.get(alert.alert_type, '⚠️')
                    
                    html += f"<tr>"
                    html += f"<td style='padding:4px; border:1px solid #ddd; font-size:11px; font-weight:bold;'>{alert.symbol}</td>"
                    html += f"<td style='padding:4px; border:1px solid #ddd; font-size:11px;'>{type_label} {alert.alert_type.replace('_', ' ').title()}</td>"
                    html += f"<td style='padding:4px; border:1px solid #ddd; font-size:11px; text-align:center; color:{severity_color}; font-weight:bold;'>{alert.severity}</td>"
                    html += f"<td style='padding:4px; border:1px solid #ddd; font-size:11px;'>{alert.description}</td>"
                    html += f"</tr>"
                
                html += "</table>"
                
                # Guidance section for high-priority alerts
                high_priority_alerts = [a for a in displayed_alerts if a.severity in ['CRITICAL', 'HIGH']]
                if high_priority_alerts:
                    html += "<h4 style='margin-top: 15px; margin-bottom: 8px; font-size: 13px; color: #d2691e;'>Recommended Actions</h4>"
                    
                    guidance_shown = set()  # Prevent duplicate guidance
                    for alert in high_priority_alerts[:5]:  # Top 5 high-priority alerts
                        if alert.guidance and alert.guidance not in guidance_shown:
                            html += f"<p style='font-size: 11px; margin: 5px 0; color: #555;'>"
                            html += f"<strong>{alert.symbol}:</strong> {alert.guidance}"
                            html += f"</p>"
                            guidance_shown.add(alert.guidance)
                
                # Show truncation notice if needed
                if len(smart_alerts) > max_alerts_in_email:
                    remaining = len(smart_alerts) - max_alerts_in_email
                    html += f"<p style='font-size: 11px; color: #888; font-style: italic;'>"
                    html += f"+ {remaining} more alerts available in full dashboard"
                    html += "</p>"
                
                # Alert disclaimer
                html += f"<p style='font-size: 10px; color: #999; margin-top: 15px; padding: 5px; background-color: #f9f9f9; border-left: 3px solid #ddd;'>"
                html += f"<strong>Disclaimer:</strong> Smart alerts are automated notifications based on price movements, sentiment analysis, and earnings schedules. "
                html += f"These are not investment recommendations. Always conduct your own research and consider your risk tolerance before making investment decisions. "
                html += f"Past performance does not guarantee future results."
                html += "</p>"
                
            else:
                html += "<p style='font-size: 12px; color: #666;'>No alerts triggered today.</p>"
    # END F06

    # BEGIN F07 - Enhanced report sections
    from config.feature_flags import (is_portfolio_analytics_enabled, 
                                     is_smart_alerts_enabled,
                                     is_finbert_pipeline_enabled)
    
    # F07: Enhanced Benchmarks section (extends F05)
    if portfolio_analytics and is_portfolio_analytics_enabled():
        benchmark_performance = portfolio_analytics.get('benchmark_performance', {})
        if benchmark_performance:
            html += render_enhanced_benchmark_section(benchmark_performance, portfolio_analytics)
    
    # F07: Enhanced Alerts section (extends F06) 
    if smart_alerts and is_smart_alerts_enabled():
        html += render_enhanced_alerts_section(smart_alerts)
    
    # F07: Enhanced FinBERT section (extends F04)
    if finbert_results and is_finbert_pipeline_enabled():
        html += render_enhanced_finbert_section(finbert_results)
    
    # F07: Performance metrics footer (only if any enhanced features are enabled)
    if any([portfolio_analytics and is_portfolio_analytics_enabled(),
            smart_alerts and is_smart_alerts_enabled(),
            finbert_results and is_finbert_pipeline_enabled()]):
        html += render_performance_footer()
    # END F07
    
    # BEGIN F15_DISCLAIMER
    # Add footer disclaimer
    footer_disclaimer = f"""
    <div style='margin-top: 24px; padding: 16px; background-color: #f8f9fa; border-radius: 4px; border-top: 2px solid #dee2e6;'>
        <div style='font-size: 10px; color: #6c757d; text-align: center; line-height: 1.4;'>
            <strong>{company_name}</strong> • {disclaimer_text}<br>
            Generated on {now} • This analysis is automated and should not be used as the sole basis for investment decisions.
        </div>
    </div>
    """
    html += footer_disclaimer
    # END F15_DISCLAIMER

    html += "</div></body></html>"

    # Write HTML to disk
    with open(out_path, "w") as f:
        f.write(html)
    logger.info(f"Report written to {out_path}")

    # Send via SMTP
    if SMTP_HOST and SMTP_USER and SMTP_PASS and EMAIL_TO:
        logger.info(f"Attempting to send email to {EMAIL_TO}")
        msg = EmailMessage()
        msg["Subject"] = f"Stock News Forecast — {now}"
        msg["From"]    = SMTP_USER
        msg["To"]      = EMAIL_TO
        msg.add_alternative(html, subtype='html')

        # Attach collages and correlation heatmap as regular attachments (not inline)
        attachment_paths = [portfolio_collage, watchlist_collage]
        
        # Add correlation heatmap if it exists (F13)
        corr_heatmap_path = "charts/corr_heatmap.png"
        if os.path.exists(corr_heatmap_path):
            attachment_paths.append(corr_heatmap_path)
        
        for path in attachment_paths:
            if path and os.path.exists(path):
                with open(path, "rb") as img:
                    data = img.read()
                msg.add_attachment(
                    data,
                    maintype="image",
                    subtype="png",
                    filename=os.path.basename(path)
                )
                logger.info(f"Attached chart: {os.path.basename(path)}")
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
                smtp.starttls()
                smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
                logger.info(f"Email sent to {EMAIL_TO}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            logger.error("Make sure you have:")
            logger.error("1. Enabled 2-Step Verification on your Google account")
            logger.error("2. Generated an App Password (not your regular password)")
            logger.error("3. Updated SMTP_PASS in config/secrets.env with the App Password")
