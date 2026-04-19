"""
Sends the HTML report via SMTP (Gmail).
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def send_report(subject: str, html_body: str) -> bool:
    sender    = os.environ.get("REPORT_EMAIL_FROM", "")
    recipient = os.environ.get("REPORT_EMAIL_TO", "")
    password  = os.environ.get("SMTP_PASSWORD", "")
    server    = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    port      = int(os.environ.get("SMTP_PORT", "587"))

    if not all([sender, recipient, password]):
        logger.error("Missing SMTP credentials (REPORT_EMAIL_FROM, REPORT_EMAIL_TO, SMTP_PASSWORD)")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(server, port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(sender, password)
            smtp.sendmail(sender, recipient, msg.as_string())
        logger.info("Email sent to %s", recipient)
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed — check REPORT_EMAIL_FROM and SMTP_PASSWORD")
        return False
    except Exception as e:
        logger.error("SMTP error: %s", e)
        return False
