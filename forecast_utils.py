import os
from datetime import datetime, timedelta
import pandas as pd

def aggregate_long_term_sentiment(sentiment_series: pd.Series,
                                  window_days: int = 90) -> float:
    """
    Compute the average sentiment over the past `window_days`.
    sentiment_series index must be datetime-like.
    Returns 0.0 if no data in window.
    """
    if sentiment_series is None or sentiment_series.empty:
        return 0.0
    end = pd.to_datetime(datetime.utcnow())
    start = end - timedelta(days=window_days)
    s = sentiment_series.copy()
    s.index = pd.to_datetime(s.index)
    window = s[(s.index >= start) & (s.index <= end)]
    return float(window.mean()) if not window.empty else 0.0

def prepare_features(price_df: pd.DataFrame,
                     market_df: pd.DataFrame,
                     peer_df: pd.DataFrame,
                     sentiment_series: pd.Series,
                     long_sentiment_window: int = 90) -> pd.DataFrame:
    """
    Merge stock, market index, and peer price DataFrames on 'Date',
    compute features:
      - MR: market daily return
      - PR: peer daily return
      - Volatility: 5-day rolling std of stock returns
      - SentimentShort: short-term EWMA(halflife=7) of daily sentiment
      - SentimentLong: average sentiment over last long_sentiment_window days
    price_df, market_df, peer_df must have columns ['Date','Close'].
    sentiment_series: pd.Series indexed by date with daily sentiment.
    Returns DataFrame with Date + feature columns, sorted ascending.
    """
    # Ensure datetime
    price = price_df.copy()
    market = market_df.copy()
    peer = peer_df.copy()
    for df in (price, market, peer):
        df['Date'] = pd.to_datetime(df['Date'])
    # Merge
    df = price.merge(market.rename(columns={'Close': 'Market_Close'}),
                     on='Date', how='outer') \
              .merge(peer.rename(columns={'Close': 'Peer_Close'}),
                     on='Date', how='outer')
    df.sort_values('Date', inplace=True)
    # Forward fill
    df[['Close', 'Market_Close', 'Peer_Close']] = \
        df[['Close', 'Market_Close', 'Peer_Close']].ffill()
    # Compute returns
    df['MR'] = df['Market_Close'].pct_change().fillna(0)
    df['PR'] = df['Peer_Close'].pct_change().fillna(0)
    df['Return'] = df['Close'].pct_change().fillna(0)
    # Volatility
    df['Volatility'] = df['Return'].rolling(5, min_periods=1).std().fillna(0)
    # Short-term sentiment EWMA
    if sentiment_series is not None and not sentiment_series.empty:
        s = sentiment_series.copy()
        s.index = pd.to_datetime(s.index)
        daily = s.groupby(s.index).mean()
        daily = daily.reindex(df['Date'], method='ffill').fillna(0)
        df['SentimentShort'] = daily.ewm(halflife=7).mean().values
    else:
        df['SentimentShort'] = 0.0
    # Long-term sentiment
    df['SentimentLong'] = aggregate_long_term_sentiment(sentiment_series, window_days=long_sentiment_window)
    # Drop rows still containing NaN
    df = df.dropna().reset_index(drop=True)
    return df[['Date', 'Close', 'MR', 'PR', 'Volatility', 'SentimentShort', 'SentimentLong', 'Return']]

