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
    watchlist:         list[str],
    portfolio:         list[str],
    head7:             dict,      # 7-day mention data
    head30:            dict,      # 30-day sentiment data
    preds:             dict,
    portfolio_collage: str = None,
    watchlist_collage: str = None,
    out_path:          str = "report.html"
):
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = f"<html><body><h2>Stock News Forecast Report — {now}</h2>"

    #
    # 30-Day Sentiment — Portfolio
    #
    html += "<h3>30-Day Sentiment — Portfolio</h3>"
    html += (
        "<table style='border-collapse: collapse;'>"
        "<tr>"
          "<th style='padding:4px;'>Ticker</th>"
          "<th>Avg_Sentiment</th>"
          "<th>Count_Positive</th>"
          "<th>Count_Negative</th>"
          "<th>Count_Neutral</th>"
          "<th>Total_Headlines</th>"
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
            f"<td style='padding:4px;'>{t}</td>"
            f"<td>{avg_sent:.2f}</td>"
            f"<td>{pos}</td>"
            f"<td>{neg}</td>"
            f"<td>{neu}</td>"
            f"<td>{total}</td>"
          f"</tr>"
        )
    html += "</table>"

    #
    # 30-Day Sentiment — Watchlist
    #
    html += "<h3>30-Day Sentiment — Watchlist</h3>"
    html += (
        "<table style='border-collapse: collapse;'>"
        "<tr>"
          "<th style='padding:4px;'>Ticker</th>"
          "<th>Avg_Sentiment</th>"
          "<th>Count_Positive</th>"
          "<th>Count_Negative</th>"
          "<th>Count_Neutral</th>"
          "<th>Total_Headlines</th>"
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
            f"<td style='padding:4px;'>{t}</td>"
            f"<td>{avg_sent:.2f}</td>"
            f"<td>{pos}</td>"
            f"<td>{neg}</td>"
            f"<td>{neu}</td>"
            f"<td>{total}</td>"
          f"</tr>"
        )
    html += "</table>"

    #
    # Watchlist Forecasts
    #
    html += "<h3>Watchlist Forecasts</h3>"
    html += (
      "<table style='border-collapse: collapse;'>"
      "<tr>"
        "<th style='padding:4px;'>Ticker</th>"
        "<th>Confidence</th>"
        "<th>RedFlag</th>"
        "<th>3-Day Forecast</th>"
      "</tr>"
    )
    for t in watchlist:
        p = preds[t]
        preds_str = ", ".join(str(x) for x in p['predictions'])
        html += (
          f"<tr>"
            f"<td style='padding:4px;'>{t}</td>"
            f"<td>{p['confidence']:.2f}</td>"
            f"<td>{p['red_flag']}</td>"
            f"<td>{preds_str}</td>"
          f"</tr>"
        )
    html += "</table>"

    #
    # Portfolio Forecasts
    #
    html += "<h3>Portfolio Forecasts</h3>"
    html += (
      "<table style='border-collapse: collapse;'>"
      "<tr>"
        "<th style='padding:4px;'>Ticker</th>"
        "<th>Confidence</th>"
        "<th>RedFlag</th>"
        "<th>3-Day Forecast</th>"
      "</tr>"
    )
    for t in portfolio:
        p = preds[t]
        preds_str = ", ".join(str(x) for x in p['predictions'])
        html += (
          f"<tr>"
            f"<td style='padding:4px;'>{t}</td>"
            f"<td>{p['confidence']:.2f}</td>"
            f"<td>{p['red_flag']}</td>"
            f"<td>{preds_str}</td>"
          f"</tr>"
        )
    html += "</table>"

    #
    # 7-Day Mention Leaders & Headlines
    #
    html += "<h3>7-Day Mention Leaders & Headlines</h3>"
    html += (
      "<table style='border-collapse: collapse;'>"
      "<tr>"
        "<th style='padding:4px;'>Ticker</th>"
        "<th>Avg_Sentiment</th>"
        "<th>Count_Positive</th>"
        "<th>Count_Negative</th>"
        "<th>Count_Neutral</th>"
        "<th>Total_Headlines</th>"
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
            f"<td style='padding:4px;'>{t}</td>"
            f"<td>{avg_s:.2f}</td>"
            f"<td>{pos}</td>"
            f"<td>{neg}</td>"
            f"<td>{neu}</td>"
            f"<td>{total}</td>"
          f"</tr>"
        )
    html += "</table>"

    # Top 5 headlines per ticker
    for t, avg_s, pos, neg, neu, total, headlines in top10:
        html += f"<p><strong>{t} Headlines (up to 5):</strong></p><ul>"
        for title, link, date in headlines[:5]:
            safe = title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            html += f"<li><a href='{link}' target='_blank'>{safe}</a> ({date})</li>"
        html += "</ul>"

    # Attachments note
    html += "<p>See attached collage images for watchlist & portfolio forecasts.</p>"
    html += "</body></html>"

    # Write HTML to disk
    with open(out_path, "w") as f:
        f.write(html)
    logger.info(f"Report written to {out_path}")

    # Send via SMTP
    if SMTP_HOST and SMTP_USER and SMTP_PASS and EMAIL_TO:
        msg = EmailMessage()
        msg["Subject"] = f"Stock News Forecast — {now}"
        msg["From"]    = SMTP_USER
        msg["To"]      = EMAIL_TO
        msg.add_alternative(html, subtype='html')

        # Attach both collages
        for path in (portfolio_collage, watchlist_collage):
            if path and os.path.exists(path):
                with open(path, "rb") as img:
                    data = img.read()
                msg.add_attachment(
                    data,
                    maintype="image",
                    subtype="png",
                    filename=os.path.basename(path)
                )
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
                smtp.starttls()
                smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
                logger.info(f"Email sent to {EMAIL_TO}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
