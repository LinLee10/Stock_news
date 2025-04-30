import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
from dotenv import load_dotenv

def send_report(subject: str, html_body: str, attachments: list[str]):
    """
    Sends an HTML email with the given subject and body,
    attaching each file in `attachments`.
    Credentials loaded from config/secrets.env.
    """
    load_dotenv("config/secrets.env")
    USER = os.getenv("EMAIL_ADDRESS")
    PWD  = os.getenv("EMAIL_PASSWORD")
    TO   = os.getenv("TO_EMAIL")

    msg = MIMEMultipart()
    msg["From"]    = USER
    msg["To"]      = TO
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    for path in attachments or []:
        if not path or not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            img = MIMEImage(f.read())
        img.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
        msg.attach(img)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(USER, PWD)
            smtp.send_message(msg)
        print("Email sent successfully.")
    except Exception as ex:
        print(f"[email_report] send error: {ex}")
