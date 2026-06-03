import logging
import pandas as pd

from news_scraper import scrape_headlines
from prediction import train_predict_stock
from charts import create_collage
from email_report import send_report
from config.tickers import TICKER_COMPANY, PORTFOLIO, WATCHLIST
from config.feature_flags import (feature_flags, is_symbol_intake_enabled, 
                                  is_news_corroboration_enabled, is_90_day_sentiment_enabled,
                                  is_finbert_pipeline_enabled, is_finbert_backtest_enabled,
                                  is_portfolio_analytics_enabled, is_smart_alerts_enabled)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_top_mentioned_stocks(headlines_data: dict, top_n: int = 10) -> list[str]:
    """
    Get the top N most mentioned stocks based on headline count.
    Returns list of ticker symbols sorted by mention count (descending).
    """
    # Count headlines for each ticker
    mention_counts = {}
    for ticker, data in headlines_data.items():
        count = data.get('count', 0)
        if count > 0:
            mention_counts[ticker] = count
    
    # Sort by count (descending) and return top N
    sorted_tickers = sorted(mention_counts.items(), key=lambda x: x[1], reverse=True)
    top_tickers = [ticker for ticker, count in sorted_tickers[:top_n]]
    
    return top_tickers

# BEGIN F04 - FinBERT analysis function
async def run_finbert_analysis(tickers: list, headlines_data: dict) -> dict:
    """
    Run FinBERT analysis for given tickers using configured parameters
    """
    from config.config import FINBERT_LAMBDA, FINBERT_BARRIER_DAYS
    from services.finbert_sentiment_analyzer import analyze_articles_for_symbol, run_finbert_backtest
    
    finbert_results = {}
    
    for ticker in tickers:
        try:
            logger.info(f"F04: Running FinBERT analysis for {ticker}")
            
            # Convert headlines to articles format
            ticker_data = headlines_data.get(ticker, {})
            headlines = ticker_data.get('headlines', [])
            
            if not headlines:
                logger.warning(f"F04: No headlines for {ticker}, skipping FinBERT analysis")
                continue
            
            # Convert headlines to article format expected by FinBERT
            articles = []
            for headline in headlines:
                if isinstance(headline, tuple) and len(headline) >= 2:
                    title, url = headline[0], headline[1]
                    articles.append({
                        'title': title,
                        'content': title,  # Use title as content for now
                        'url': url
                    })
            
            if not articles:
                continue
            
            # Create dummy price data and technical indicators for now
            # In production, these would come from actual price data
            import pandas as pd
            price_data = pd.DataFrame({
                'close': [100.0] * 30,  # Dummy price data
                'date': pd.date_range('2025-01-01', periods=30)
            })
            
            technical_indicators = {
                'rsi': 50.0,
                'macd_signal': 0.0,
                'bb_position': 0.5
            }
            
            # Run regular FinBERT analysis if pipeline enabled
            if is_finbert_pipeline_enabled():
                recommendation = await analyze_articles_for_symbol(
                    articles, ticker, price_data, technical_indicators,
                    lambda_ewma=FINBERT_LAMBDA,
                    barrier_window_days=FINBERT_BARRIER_DAYS
                )
                
                if recommendation:
                    finbert_results[ticker] = {
                        'recommendation': recommendation,
                        'article_count': len(articles)
                    }
                    logger.info(f"F04: FinBERT analysis completed for {ticker}")
            
            # Run backtest if enabled
            if is_finbert_backtest_enabled():
                backtest_results = await run_finbert_backtest(
                    articles, ticker, price_data, technical_indicators
                )
                
                if ticker in finbert_results:
                    finbert_results[ticker]['backtest'] = backtest_results
                else:
                    finbert_results[ticker] = {
                        'backtest': backtest_results,
                        'article_count': len(articles)
                    }
                
                logger.info(f"F04: FinBERT backtest completed for {ticker}")
                
        except Exception as e:
            logger.error(f"F04: FinBERT analysis failed for {ticker}: {e}")
            continue
    
    return finbert_results
# END F04

def scrape_all_stocks_for_mentions(days: int = 7) -> dict:
    """
    Scrape headlines for a broader set of stocks to get true top mentioned stocks.
    This includes major stocks beyond just the portfolio/watchlist.
    """
    # Extended list of major stocks to check for mentions
    major_stocks = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "CRM", "ADBE", "PYPL", "INTC", "AMD", "QCOM", "AVGO", "TXN", "MU",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "COF", "AXP",
        # Healthcare
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "DHR", "ABT", "LLY", "BMY",
        # Consumer
        "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS", "NKE", "SBUX", "TGT",
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "KMI", "PSX", "VLO", "MPC", "OXY",
        # Industrial
        "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX", "NOC",
        # Other major stocks
        "PLTR", "RIVN", "LCID", "NIO", "XPEV", "LI", "BYD", "BIDU", "JD", "PDD"
    ]
    
    logger.info(f"Scraping headlines for {len(major_stocks)} major stocks to find top mentioned...")
    
    # Scrape headlines for all major stocks
    all_headlines = scrape_headlines(major_stocks, days=days)
    
    return all_headlines

def main():
    # 1. Scrape headlines & sentiment for portfolio/watchlist
    logger.info(f"Starting main pipeline with flags: {feature_flags.get_all_flags()}")
    
    # Core functionality (always enabled)
    all_tickers = WATCHLIST + PORTFOLIO
    head7  = scrape_headlines(all_tickers, days=7)    # for Top-10 mentions
    
    # 90-day sentiment analysis (feature flagged)
    if is_90_day_sentiment_enabled():
        logger.info("90-day sentiment analysis enabled - using extended sentiment window")
        head30 = scrape_headlines(all_tickers, days=90)   # Extended 90-day window
        # Future: Could integrate with FinBERT sentiment pipeline here
        # from services.finbert_sentiment_analyzer import create_finbert_pipeline
        # sentiment_pipeline = await create_finbert_pipeline()
        # enhanced_sentiment = await sentiment_pipeline.analyze_articles_for_symbol(...)
    else:
        logger.debug("90-day sentiment analysis disabled - using standard 30-day window")
        head30 = scrape_headlines(all_tickers, days=30)   # Standard 30-day summaries
    
    # 2. Get top 10 most mentioned stocks in past 7 days (from broader market)
    all_stocks_headlines = scrape_all_stocks_for_mentions(days=7)
    top_mentioned = get_top_mentioned_stocks(all_stocks_headlines, top_n=10)
    
    # Enhanced news processing (feature flagged)
    if is_news_corroboration_enabled():
        logger.info("News corroboration enabled - enhancing sentiment analysis")
        # Future: enhanced_head30 = enhance_with_corroboration(head30)
    else:
        logger.debug("News corroboration disabled - using basic sentiment")
    
    logger.info(f"Top {len(top_mentioned)} most mentioned stocks: {', '.join(top_mentioned)}")

    # BEGIN F04 - FinBERT integration
    finbert_results = {}
    if is_finbert_pipeline_enabled() or is_finbert_backtest_enabled():
        logger.info("F04: FinBERT analysis enabled")
        import asyncio
        finbert_results = asyncio.run(run_finbert_analysis(all_tickers, head30))
    # END F04

    # BEGIN F05 - Portfolio analytics
    portfolio_analytics = {}
    if is_portfolio_analytics_enabled():
        logger.info("F05: Portfolio analytics enabled")
        try:
            from analytics.portfolio_metrics import analyze_portfolio
            portfolio_analytics = analyze_portfolio(PORTFOLIO)
            logger.info("F05: Portfolio analysis completed")
        except Exception as e:
            logger.error(f"F05: Portfolio analytics failed: {e}")
    # END F05

    # BEGIN F06 - Smart Alerts evaluation
    smart_alerts = []
    if is_smart_alerts_enabled():
        logger.info("F06: Smart alerts enabled")
        try:
            from services.monitoring_alerting import create_smart_alerts_engine
            
            # Create smart alerts engine
            alerts_engine = create_smart_alerts_engine()
            
            # Convert price data to proper format for alerts engine
            price_data_for_alerts = {}
            for symbol in all_tickers:
                # We'll get price data from the prediction pipeline results later
                # For now, create placeholder - this will be updated after price fetching
                price_data_for_alerts[symbol] = None
            
            # We'll evaluate alerts after price data is collected in the prediction loop
            logger.info("F06: Smart alerts engine initialized, will evaluate after price data collection")
        except Exception as e:
            logger.error(f"F06: Smart alerts initialization failed: {e}")
    # END F06

    # 2. Forecast & gather price history
    preds = {}
    price_data, forecast_data, sentiment_map = {}, {}, {}
    for t in all_tickers:
        daily_sent  = head30[t]["daily_sentiment"]
        sent_series = pd.Series(daily_sent)
        out = train_predict_stock(t, sent_series)

        preds[t] = {
            "predictions": out["predictions"],
            "confidence":  out["confidence"],
            "red_flag":    out["red_flag"]
        }
        price_data[t] = out["history"]
        hist_df = out['history']
        perf_10d = None
        if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
            closes = hist_df['Stock_Close'] if 'Stock_Close' in hist_df.columns else hist_df.iloc[:, -1]
            if len(closes) > 10:
                old_price = closes.iloc[-11]
            else:
                old_price = closes.iloc[0]
            last_price = closes.iloc[-1]
            if old_price != 0:
                perf_10d = (last_price - old_price) / old_price * 100
        preds[t]['perf_10d'] = perf_10d
        forecast_data[t] = pd.DataFrame({
            "Date":           [d.strftime("%Y-%m-%d") for d in out["dates"]],
            "Forecast_Close": out["predictions"],
        })
        sentiment_map[t] = daily_sent

    # BEGIN F06 - Evaluate smart alerts now that we have price and sentiment data
    if is_smart_alerts_enabled() and 'alerts_engine' in locals():
        logger.info("F06: Evaluating smart alerts with collected price and sentiment data")
        try:
            # Convert price data to format expected by alerts engine
            price_data_for_alerts = {}
            for symbol, hist_df in price_data.items():
                if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
                    # Convert to expected format with Date and Close columns
                    alerts_df = hist_df.copy()
                    if 'Stock_Close' in alerts_df.columns:
                        alerts_df['Close'] = alerts_df['Stock_Close']
                    elif 'Close' not in alerts_df.columns and len(alerts_df.columns) > 0:
                        # Use the last column as Close price
                        alerts_df['Close'] = alerts_df.iloc[:, -1]
                    
                    # Ensure Date column exists
                    if 'Date' not in alerts_df.columns:
                        if alerts_df.index.name == 'Date' or 'date' in str(alerts_df.index.dtype).lower():
                            alerts_df = alerts_df.reset_index()
                            if 'index' in alerts_df.columns:
                                alerts_df['Date'] = alerts_df['index']
                        else:
                            # Create date range if no date info available
                            alerts_df['Date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(alerts_df), freq='D')
                    
                    price_data_for_alerts[symbol] = alerts_df
            
            # Evaluate alerts
            smart_alerts = alerts_engine.evaluate_alerts(
                symbols=all_tickers,
                price_data=price_data_for_alerts,
                sentiment_data=head30,
                earnings_data=None  # TODO: Integrate earnings data when available
            )
            
            logger.info(f"F06: Evaluated smart alerts, found {len(smart_alerts)} triggered alerts")
            
            # Log alert summary
            if smart_alerts:
                alert_summary = {}
                for alert in smart_alerts:
                    alert_summary[alert.severity] = alert_summary.get(alert.severity, 0) + 1
                
                logger.info(f"F06: Alert breakdown: {dict(alert_summary)}")
            
        except Exception as e:
            logger.error(f"F06: Smart alerts evaluation failed: {e}")
            smart_alerts = []
    # END F06

    # 3. Create and save collages
    portfolio_collage = "portfolio_collage.png"
    create_collage(
        PORTFOLIO,
        price_data,
        forecast_data,
        "Portfolio Forecasts",
        portfolio_collage
    )
    logger.info(f"Saved collage to {portfolio_collage}")

    watchlist_collage = "watchlist_collage.png"
    create_collage(
        WATCHLIST,
        price_data,
        forecast_data,
        "Watchlist Forecasts",
        watchlist_collage
    )
    logger.info(f"Saved collage to {watchlist_collage}")

    # BEGIN F16_WIRE
    # Fetch earnings schedule if enabled
    earnings_schedule = None
    if feature_flags.is_enabled('enable_earnings_reads'):
        logger.info("F16: Fetching earnings schedule for portfolio and watchlist")
        try:
            from services.earnings_service import EarningsAnalysisService
            earnings_service = EarningsAnalysisService()
            
            # Get earnings for next 14 days (configurable via env)
            import os
            days_ahead = int(os.getenv('EARNINGS_WINDOW_DAYS', '14'))
            earnings_df = earnings_service.get_schedule(all_tickers, days=days_ahead)
            
            if not earnings_df.empty:
                # Save to CSV atomically
                earnings_csv_path = "data/earnings_schedule.csv"
                temp_path = f"{earnings_csv_path}.tmp"
                earnings_df.to_csv(temp_path, index=False)
                os.rename(temp_path, earnings_csv_path)  # Atomic replacement
                
                logger.info(f"F16: Saved {len(earnings_df)} earnings events to {earnings_csv_path}")
                earnings_schedule = earnings_df
            else:
                logger.info("F16: No upcoming earnings found in the specified window")
                
        except Exception as e:
            logger.error(f"F16: Failed to fetch earnings schedule: {e}")
            earnings_schedule = None
    # END F16_WIRE

    # 4. Generate & send report (HTML), passing both windows of headlines
    send_report(
        watchlist=WATCHLIST,
        portfolio=PORTFOLIO,
        head7=head7,
        head30=head30,
        preds=preds,
        portfolio_collage=portfolio_collage,
        watchlist_collage=watchlist_collage,
        out_path="report.html",
        top_mentioned=top_mentioned,
        finbert_results=finbert_results,  # F04: Add FinBERT results
        portfolio_analytics=portfolio_analytics,  # F05: Add portfolio analytics
        smart_alerts=smart_alerts if is_smart_alerts_enabled() else None,  # F06: Add smart alerts
        earnings_schedule=earnings_schedule  # F16: Add earnings schedule
    )

    # BEGIN F17_WIRE
    # Symbol intake processing if enabled
    if is_symbol_intake_enabled():
        logger.info("F17: Symbol intake enabled - processing candidate symbols")
        try:
            from services.symbol_intake import symbol_intake_service
            
            # Get candidate symbols from environment or CSV
            candidates = []
            
            # Option 1: From environment variable (comma-separated)
            env_candidates = os.getenv('SYMBOL_INTAKE_LIST', '')
            if env_candidates:
                candidates.extend([s.strip() for s in env_candidates.split(',') if s.strip()])
            
            # Option 2: From CSV file
            intake_csv_path = os.getenv('SYMBOL_INTAKE_CSV', 'data/symbol_candidates.csv')
            if os.path.exists(intake_csv_path):
                try:
                    candidates_df = pd.read_csv(intake_csv_path)
                    if 'symbol' in candidates_df.columns:
                        candidates.extend(candidates_df['symbol'].dropna().astype(str).tolist())
                    logger.info(f"F17: Loaded {len(candidates_df)} candidates from {intake_csv_path}")
                except Exception as e:
                    logger.warning(f"F17: Failed to read candidate CSV {intake_csv_path}: {e}")
            
            # Process candidates
            if candidates:
                intake_result = symbol_intake_service.intake_symbols(candidates)
                
                # Merge new symbols into current run if any were accepted
                if intake_result['new_symbols']:
                    original_count = len(all_tickers)
                    all_tickers.extend(intake_result['new_symbols'])
                    logger.info(f"F17: Added {len(intake_result['new_symbols'])} new symbols to current run "
                               f"(total symbols: {original_count} -> {len(all_tickers)})")
                else:
                    logger.info("F17: No new symbols added to current run")
            else:
                logger.info("F17: No candidate symbols found for intake")
                
        except Exception as e:
            logger.error(f"F17: Symbol intake processing failed: {e}")
    # END F17_WIRE
    
    logger.info("Main pipeline completed")

def start_api_server():
    """Start the API server if enabled"""
    from config.feature_flags import feature_flags
    
    if feature_flags.is_enabled('enable_api_endpoints'):
        from api.app import create_app
        from config.config import API_HOST, API_PORT, API_DEBUG
        
        app = create_app()
        app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--api':
        # Start API server mode
        start_api_server()
    else:
        # Run main pipeline
        main()
