# email_report.py

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage
from dotenv               import load_dotenv

load_dotenv("config/secrets.env")
USER = os.getenv("EMAIL_ADDRESS")
PWD  = os.getenv("EMAIL_PASSWORD")
TO   = os.getenv("TO_EMAIL")

def send_report(portfolio_results, watchlist_results,
                summary7, summary30, attachments, top_headlines):
    """
    portfolio_results: list of dicts from train_predict_stock()
    watchlist_results: same but for watchlist
    summary7: pd.DataFrame of 7-day mentions summary
    summary30: pd.DataFrame of 30-day sentiment summary
    top_headlines: dict[ticker] -> list of (headline:str, url:str)
    """

    # 1) Build subject + container
    subject = "Daily Stock Forecast Report"
    msg = MIMEMultipart("related")
    msg["From"]    = USER
    msg["To"]      = TO
    msg["Subject"] = subject

    # 2) Style block (embedded CSS)
    style = """
    <style>
      body { font-family: Arial, sans-serif; color: #333; }
      h1 { text-align: center; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
      th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: center; }
      th { background: #f4f4f4; }
      .section { margin-top: 30px; }
      .ticker-headlines { margin-bottom: 15px; }
      .ticker-headlines h4 { margin: 5px 0; }
    </style>
    """

    # 3) 7-day mentions table HTML
    t7 = summary7.to_html(index=False, classes="table", border=0)


    # 4a) Split 30-day sentiment into Portfolio vs Watchlist
    port_tickers  = [r["ticker"] for r in portfolio_results]
    watch_tickers = [r["ticker"] for r in watchlist_results]

    summary30_port  = summary30[summary30["Ticker"].isin(port_tickers)]
    summary30_watch = summary30[summary30["Ticker"].isin(watch_tickers)]

    t30_port  = summary30_port.to_html(index=False, classes="table", border=0)
    t30_watch = summary30_watch.to_html(index=False, classes="table", border=0)


    # 5) Portfolio forecasts table
    rows = ""
    for r in portfolio_results:
        rows += f"<tr><td>{r['ticker']}</td><td>{', '.join(f'{d.date()}:${p}' for d,p in zip(r['dates'],r['predictions']))}</td></tr>"
    t_port = f"""
      <table>
        <thead><tr><th>Ticker</th><th>3-Day Forecasts</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    """

    # 6) Watchlist forecasts table
    rows = ""
    for r in watchlist_results:
        rows += f"<tr><td>{r['ticker']}</td><td>{', '.join(f'{d.date()}:${p}' for d,p in zip(r['dates'],r['predictions']))}</td></tr>"
    t_watch = f"""
      <table>
        <thead><tr><th>Ticker</th><th>3-Day Forecasts</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    """

    # 7) Top-10 headlines block
    hl_block = ""
    for tic, items in top_headlines.items():
        hl_block += f"<div class='ticker-headlines'><h4>{tic}</h4><ul>"
        for head, link in items:
            hl_block += f"<li><a href='{link}' target='_blank'>{head}</a></li>"
        hl_block += "</ul></div>"

    # 8) Assemble full HTML body
    html_body = f"""
    <html><head>{style}</head><body>
      <h1>{subject}</h1>

      <div class='section'>
        <h2>7-Day Mention Leaders & Headlines</h2>
        {t7}
        {hl_block}
      </div>

      <div class='section'>
        <h2>30-Day Sentiment (Portfolio)</h2>
        {t30_port}
      </div>

      <div class='section'>
        <h2>30-Day Sentiment (Watchlist) </h2>
        {t30_watch}
      </div>

      <div class='section'>
        <h2>Portfolio Forecasts</h2>
        {t_port}
      </div>

      <div class='section'>
        <h2>Watchlist Forecasts</h2>
        {t_watch}
      </div>
    </body></html>
    """

    # 9) Attach HTML
    msg.attach(MIMEText(html_body, "html"))

    # 10) Attach images (plots)
    for path in attachments or []:
        if not path or not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-Disposition", "attachment",
                           filename=os.path.basename(path))
            msg.attach(img)

    # 11) Send
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(USER, PWD)
            smtp.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print(f"[email_report] send error: {e}")
