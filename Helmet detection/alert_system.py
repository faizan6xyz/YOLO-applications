"""
Email Alert System for Helmet Violations
Sends email with violation photo when no-helmet is detected
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your_email@gmail.com",
    "sender_password": "your_app_password",  # Use Gmail App Password
    "recipient_emails": ["supervisor@company.com", "safety@company.com"],
    "alert_cooldown_minutes": 5,  # Don't spam alerts
}

last_alert_time = None


def send_violation_alert(photo_path: str, confidence: float, location: str = "Main Gate"):
    """Send email alert with violation photo attached."""
    global last_alert_time

    now = datetime.now()

    # Cooldown check
    if last_alert_time:
        elapsed = (now - last_alert_time).total_seconds() / 60
        if elapsed < EMAIL_CONFIG["alert_cooldown_minutes"]:
            print(f"[EMAIL] Alert cooldown active ({elapsed:.1f} min elapsed). Skipping.")
            return False

    last_alert_time = now

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG["sender_email"]
        msg['To'] = ", ".join(EMAIL_CONFIG["recipient_emails"])
        msg['Subject'] = f"⚠ HELMET VIOLATION DETECTED — {location} — {now.strftime('%H:%M:%S')}"

        # Email body
        body = f"""
        <html><body>
        <h2 style="color:red;">⚠ Safety Violation Alert</h2>
        <table>
            <tr><td><b>Location:</b></td><td>{location}</td></tr>
            <tr><td><b>Time:</b></td><td>{now.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            <tr><td><b>Confidence:</b></td><td>{confidence:.0%}</td></tr>
            <tr><td><b>Photo:</b></td><td>{os.path.basename(photo_path)}</td></tr>
        </table>
        <br>
        <p>A person without a helmet was detected. Photo attached.</p>
        <p><i>Automated Helmet Detection System</i></p>
        </body></html>
        """
        msg.attach(MIMEText(body, 'html'))

        # Attach photo
        if os.path.exists(photo_path):
            with open(photo_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment',
                               filename=os.path.basename(photo_path))
                msg.attach(img)

        # Send
        with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
            server.sendmail(
                EMAIL_CONFIG["sender_email"],
                EMAIL_CONFIG["recipient_emails"],
                msg.as_string()
            )

        print(f"[EMAIL] Alert sent to: {', '.join(EMAIL_CONFIG['recipient_emails'])}")
        return True

    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False


# ─────────────────────────────────────────────
# Integrate with helmet_detector.py
# ─────────────────────────────────────────────
# In helmet_detector.py, after capture_violation():
#
# from alert_system import send_violation_alert
#
# photo_path = detector.capture_violation(frame, bbox, confidence)
# if photo_path:
#     send_violation_alert(photo_path, confidence, location="Zone A")


if __name__ == "__main__":
    # Test the alert system
    test_photo = "captures/test.jpg"
    print("Testing email alert system...")
    send_violation_alert(test_photo, 0.92, location="Test Zone")
