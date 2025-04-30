import pandas as pd
import numpy as np
from prediction import fetch_price_history, train_and_forecast
from sklearn.metrics import mean_absolute_error, mean_squared_error

portfolio = pd.read_csv("data/portfolio.csv")["Ticker"].tolist()
watchlist = pd.read_csv("data/watchlist.csv")["Ticker"].tolist()
all_tickers = portfolio + watchlist

results = []
for tic in all_tickers:
    hist = fetch_price_history(tic, period="180d")
    if hist.empty or len(hist) < 100:
        continue

    errors_price = []
    errors_return = []

    # Rolling 1-day-ahead forecasts over last 90 days
    for split in range(60, len(hist)-1):
        train = hist.iloc[:split].copy()
        test_row = hist.iloc[split+1]
        market = fetch_price_history("SPY", "180d").iloc[:split]
        peer   = fetch_price_history("QQQ", "180d").iloc[:split]
        # No sentiment series available here—skip sentiment or load from logs
        fdf, _, _ = train_and_forecast(tic, train, market, peer, None)
        if fdf.empty:
            continue
        pred_price = fdf["Forecast_Close"].iloc[0]
        actual_price = test_row["Close"]
        errors_price.append(abs(pred_price - actual_price))
        errors_return.append(abs((pred_price/actual_price - 1)))

    if not errors_price:
        continue

    mae_price = np.mean(errors_price)
    rmse_price = np.sqrt(np.mean(np.square(errors_price)))
    mae_ret = np.mean(errors_return)
    rmse_ret = np.sqrt(np.mean(np.square(errors_return)))

    results.append({
        "Ticker": tic,
        "MAE_Price": mae_price,
        "RMSE_Price": rmse_price,
        "MAE_Return": mae_ret,
        "RMSE_Return": rmse_ret
    })

df = pd.DataFrame(results)
print(df)
print("\nOverall MAE Price:", df["MAE_Price"].mean())
print("Overall MAE Return:", df["MAE_Return"].mean())
