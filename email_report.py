import os
import logging
import smtplib
import pandas as pd

from email.message import EmailMessage
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger("email_report")
logging.basicConfig(level=logging.INFO)

# ─── Load SMTP credentials ────────────────────────────────
load_dotenv("config/secrets.env")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_TO  = os.getenv("EMAIL_TO")

def send_report(
    watchlist: list[str],
    portfolio:  list[str],
    head_data:  dict,
    preds:      dict,
    collage_path: str = None,
    out_path:     str = "report.html"
):
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    # Build HTML body
    html = f"<html><body><h2>Stock News Forecast Report — {now}</h2>"

    # Watchlist Forecasts
    html += "<h3>Watchlist Forecasts</h3>"
    html += "<table style='border-collapse: collapse;'><tr><th style='padding:4px;'>Ticker</th><th>Confidence</th><th>RedFlag</th><th>3-Day Forecast</th></tr>"
    for t in watchlist:
        p = preds[t]
        preds_str = ", ".join(str(x) for x in p['predictions'])
        html += f"<tr><td style='padding:4px;'>{t}</td><td>{p['confidence']:.2f}</td><td>{p['red_flag']}</td><td>{preds_str}</td></tr>"
    html += "</table>"

    # Portfolio Forecasts
    html += "<h3>Portfolio Forecasts</h3>"
    html += "<table style='border-collapse: collapse;'><tr><th style='padding:4px;'>Ticker</th><th>Confidence</th><th>RedFlag</th><th>3-Day Forecast</th></tr>"
    for t in portfolio:
        p = preds[t]
        preds_str = ", ".join(str(x) for x in p['predictions'])
        html += f"<tr><td style='padding:4px;'>{t}</td><td>{p['confidence']:.2f}</td><td>{p['red_flag']}</td><td>{preds_str}</td></tr>"
    html += "</table>"

    # 7-Day Mention Leaders & Headlines
    html += "<h3>7-Day Mention Leaders & Headlines</h3>"
    html += "<table style='border-collapse: collapse;'><tr><th style='padding:4px;'>Ticker</th><th>Avg_Sentiment</th><th>Count_Positive</th><th>Count_Negative</th><th>Count_Neutral</th><th>Total_Headlines</th></tr>"
    recs = []
    for t, info in head_data.items():
        headlines = info['headlines']
        total     = info['count']
        pos = info.get('count_positive', 0)
        neg = info.get('count_negative', 0)
        neu = info.get('count_neutral', 0)
        avg_sent = sum(info['daily_sentiment'].values()) / len(info['daily_sentiment']) if info['daily_sentiment'] else 0.0
        recs.append((t, avg_sent, pos, neg, neu, total, headlines))
    recs.sort(key=lambda x: x[5], reverse=True)  # sort by total headlines
    top10 = recs[:10]
    for t, avg_sent, pos, neg, neu, total, headlines in top10:
        html += (f"<tr><td style='padding:4px;'>{t}</td>"
                 f"<td>{avg_sent:.2f}</td><td>{pos}</td><td>{neg}</td><td>{neu}</td><td>{total}</td></tr>")
    html += "</table>"

    # Headlines for each top ticker
    for t, avg_sent, pos, neg, neu, total, headlines in top10:
        html += f"<p><strong>{t} Headlines:</strong></p><ul>"
        for title, link, date in headlines:
            safe_title = title.replace('<','&lt;').replace('>','&gt;')
            html += f"<li><a href='{link}'>{safe_title}</a> ({date})</li>"
        html += "</ul>"

    # 30-Day Sentiment (Portfolio vs Watchlist combined) – using same head_data as placeholder
    html += "<h3>30-Day Sentiment</h3>"
    html += "<table style='border-collapse: collapse;'><tr><th style='padding:4px;'>Ticker</th><th>Avg_Sentiment</th><th>Count_Positive</th><th>Count_Negative</th><th>Count_Neutral</th><th>Total_Headlines</th></tr>"
    for t, info in head_data.items():
        total = info['count']
        pos = info.get('count_positive', 0)
        neg = info.get('count_negative', 0)
        neu = info.get('count_neutral', 0)
        avg_sent = sum(info['daily_sentiment'].values()) / len(info['daily_sentiment']) if info['daily_sentiment'] else 0.0
        html += (f"<tr><td style='padding:4px;'>{t}</td>"
                 f"<td>{avg_sent:.2f}</td><td>{pos}</td><td>{neg}</td><td>{neu}</td><td>{total}</td></tr>")
    html += "</table>"

    # Embed collage image if provided
    if collage_path and os.path.exists(collage_path):
        html += "<h3>Portfolio & Watchlist Collage</h3>"
        cid = 'collage_img'
        html += f"<img src='cid:{cid}' alt='Collage'/>"

    html += "</body></html>"

    # Write HTML report to file
    with open(out_path, "w") as f:
        f.write(html)
    logger.info(f"Report written to {out_path}")

    # Send email with HTML content
    if SMTP_HOST and SMTP_USER and SMTP_PASS and EMAIL_TO:
        msg = EmailMessage()
        msg["Subject"] = f"Stock News Forecast — {now}"
        msg["From"]    = SMTP_USER
        msg["To"]      = EMAIL_TO
        msg.add_alternative(html, subtype='html')
        if collage_path and os.path.exists(collage_path):
            with open(collage_path, "rb") as img:
                img_data = img.read()
            msg.get_payload()[0].add_related(img_data, 'image', 'png', cid=cid)
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
                smtp.starttls()
                smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
                logger.info(f"Email sent to {EMAIL_TO}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
