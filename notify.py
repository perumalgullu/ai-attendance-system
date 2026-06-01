"""
Notification module — plug in your own email/SMS/WhatsApp provider here.

For Streamlit Cloud deployment, set credentials in:
  .streamlit/secrets.toml

Example secrets.toml:
  [email]
  smtp_host = "smtp.gmail.com"
  smtp_port = 587
  sender = "you@gmail.com"
  password = "app_password"

  [twilio]
  account_sid = "ACxxxxxxxx"
  auth_token  = "xxxxxxxx"
  from_phone  = "+1XXXXXXXXXX"
  from_whatsapp = "whatsapp:+14155238886"
"""

import streamlit as st
import smtplib
from email.mime.text import MIMEText


# ────────────────────────────────────────────
# EMAIL
# ────────────────────────────────────────────

def send_email(to_email: str, subject: str, body: str):
    """Send email via SMTP. Credentials pulled from st.secrets."""
    try:
        cfg = st.secrets["email"]
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = cfg["sender"]
        msg["To"]      = to_email

        with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"])) as server:
            server.starttls()
            server.login(cfg["sender"], cfg["password"])
            server.sendmail(cfg["sender"], to_email, msg.as_string())

        print(f"[Email] Sent to {to_email}")

    except KeyError:
        print("[Email] Skipped — no email credentials in secrets.toml")
    except Exception as e:
        print(f"[Email Error] {e}")


# ────────────────────────────────────────────
# SMS  (Twilio)
# ────────────────────────────────────────────

def send_sms(to_phone: str, message: str):
    """Send SMS via Twilio. Credentials pulled from st.secrets."""
    try:
        from twilio.rest import Client
        cfg = st.secrets["twilio"]
        client = Client(cfg["account_sid"], cfg["auth_token"])
        client.messages.create(body=message, from_=cfg["from_phone"], to=to_phone)
        print(f"[SMS] Sent to {to_phone}")

    except KeyError:
        print("[SMS] Skipped — no Twilio credentials in secrets.toml")
    except ImportError:
        print("[SMS] Skipped — twilio package not installed")
    except Exception as e:
        print(f"[SMS Error] {e}")


# ────────────────────────────────────────────
# WHATSAPP  (Twilio Sandbox)
# ────────────────────────────────────────────

def send_whatsapp(to_phone: str, message: str):
    """Send WhatsApp message via Twilio sandbox."""
    try:
        from twilio.rest import Client
        cfg = st.secrets["twilio"]
        client = Client(cfg["account_sid"], cfg["auth_token"])
        client.messages.create(
            body=message,
            from_=cfg["from_whatsapp"],
            to=f"whatsapp:{to_phone}"
        )
        print(f"[WhatsApp] Sent to {to_phone}")

    except KeyError:
        print("[WhatsApp] Skipped — no Twilio credentials in secrets.toml")
    except ImportError:
        print("[WhatsApp] Skipped — twilio package not installed")
    except Exception as e:
        print(f"[WhatsApp Error] {e}")
