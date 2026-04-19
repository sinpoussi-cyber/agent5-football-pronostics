import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

def send_report(subject: str, html_body: str) -> bool:
    sender      = os.environ.get("REPORT_EMAIL_FROM", "")
    recipient   = os.environ.get("REPORT_EMAIL_TO", "")
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    port_str    = os.environ.get("SMTP_PORT", "587")
    port        = int(port_str.strip()) if port_str and port_str.strip().isdigit() else 587
    password    = os.environ.get("SMTP_PASSWORD", "")

    if not all([sender, recipient, password]):
        logger.error("Missing SMTP credentials (REPORT_EMAIL_FROM, REPORT_EMAIL_TO, SMTP_PASSWORD)")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(html_body, "html"))

    try:
        logger.info(f"Connexion SMTP à {smtp_server}:{port} depuis {sender}")
        with smtplib.SMTP(smtp_server, port, timeout=30) as smtp:
            logger.info("SMTP connecté")
            smtp.ehlo()
            logger.info("ehlo() 1 OK")
            smtp.starttls()
            logger.info("starttls() OK")
            smtp.ehlo()
            logger.info("ehlo() 2 OK")
            smtp.login(sender, password)
            logger.info("login() OK")
            smtp.sendmail(sender, recipient, msg.as_string())
            logger.info(f"Email envoyé avec succès à {recipient}")
        return True
    except Exception as e:
        logger.error(f"SMTP error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
