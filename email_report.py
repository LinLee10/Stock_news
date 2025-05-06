import os
import logging
import smtplib
import pandas as pd

from email.message import EmailMessage
from prettytable import PrettyTable
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger("email_report")
logging.basicConfig(level=logging.INFO)

# ─── Load SMTP creds ─────────────────────────────────────────────────────────
load_dotenv("config/secrets.env")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_TO  = os.getenv("EMAIL_TO")


def format_table(df: pd.DataFrame, title: str) -> str:
    tbl = PrettyTable()
    tbl.field_names = df.columns.tolist()
    for _, row in df.iterrows():
        tbl.add_row(row.tolist())
    return f"\n{title}\n{tbl}\n"


def send_report(
    watchlist: list[str],
    portfolio:  list[str],
    head_data:  dict,
    preds:      dict,
    collage_path: str = None,
    out_path:     str = "report.txt"
):
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    body = f"Stock News Forecast Report — {now}\n"

    # Watchlist
    wl_rows = []
    for t in watchlist:
        p = preds[t]
        wl_rows.append({
            "Ticker":      t,
            "Confidence":  f"{p['confidence']:.2f}",
            "RedFlag":     p["red_flag"],
            "Predictions": ", ".join(str(x) for x in p["predictions"])
        })
    df_wl = pd.DataFrame(wl_rows)
    body += format_table(df_wl, "Watchlist Forecasts")

    # Portfolio
    pf_rows = []
    for t in portfolio:
        p = preds[t]
        pf_rows.append({
            "Ticker":      t,
            "Confidence":  f"{p['confidence']:.2f}",
            "RedFlag":     p["red_flag"],
            "Predictions": ", ".join(str(x) for x in p["predictions"])
        })
    df_pf = pd.DataFrame(pf_rows)
    body += format_table(df_pf, "Portfolio Forecasts")

    # Top 10 mentions
    recs = []
    for t, info in head_data.items():
        d_sent   = info["daily_sentiment"]
        avg_sent = sum(d_sent.values()) / len(d_sent) if d_sent else 0.0
        heads    = "; ".join([h for h, _, _ in info["headlines"]])
        recs.append({
            "Ticker":   t,
            "Mentions": info["count"],
            "AvgSent":  f"{avg_sent:.2f}",
            "Headlines": heads
        })
    top10   = sorted(recs, key=lambda r: r["Mentions"], reverse=True)[:10]
    df_top = pd.DataFrame(top10)
    body += format_table(df_top, "Top 10 Most Mentioned (Last 10 days)")

    # write local file
    with open(out_path, "w") as f:
        f.write(body)
    logger.info(f"Report written to {out_path}")

    # send email if creds present
    if SMTP_HOST and SMTP_USER and SMTP_PASS and EMAIL_TO:
        msg = EmailMessage()
        msg["Subject"] = f"Stock News Forecast — {now}"
        msg["From"]    = SMTP_USER
        msg["To"]      = EMAIL_TO
        msg.set_content(body)

        if collage_path and os.path.exists(collage_path):
            with open(collage_path, "rb") as img:
                data = img.read()
            msg.add_attachment(data,
                maintype="image", subtype="png",
                filename=os.path.basename(collage_path)
            )

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
                smtp.starttls()
                smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
            logger.info(f"Email sent to {EMAIL_TO}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
