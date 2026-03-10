"""
Notification and Alert System for Statistical Arbitrage Simulator

Supports email notifications for trading signals and alerts.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, List, Dict
import json

from config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL
from database.models import Alert, get_db, AlertRepository
from utils.logging_config import logger


class EmailNotifier:
    """Send email notifications for trading alerts."""
    
    def __init__(
        self,
        smtp_host: str = SMTP_HOST,
        smtp_port: int = SMTP_PORT,
        smtp_user: str = SMTP_USER,
        smtp_password: str = SMTP_PASSWORD
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.enabled = bool(smtp_host and smtp_user and smtp_password)
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> bool:
        """Send an email notification."""
        if not self.enabled:
            logger.warning("Email notifications not configured")
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = to_email
            
            # Attach plain text
            msg.attach(MIMEText(body, "plain"))
            
            # Attach HTML if provided
            if html_body:
                msg.attach(MIMEText(html_body, "html"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, to_email, msg.as_string())
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_signal_alert(
        self,
        to_email: str,
        pair: str,
        signal_type: str,
        zscore: float,
        action: str
    ) -> bool:
        """Send a trading signal alert."""
        subject = f"[Stat Arb] {signal_type.upper()} Signal: {pair}"
        
        body = f"""
Trading Signal Alert
=====================

Pair: {pair}
Signal: {signal_type.upper()}
Z-Score: {zscore:.3f}
Recommended Action: {action}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is an automated alert from Statistical Arbitrage Simulator.
        """
        
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; color: white;">
        <h1 style="margin: 0;">Trading Signal Alert</h1>
    </div>
    <div style="padding: 20px; background: #f9f9f9;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Pair</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{pair}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Signal</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">
                    <span style="background: {'#28a745' if signal_type == 'entry' else '#dc3545'}; 
                                 color: white; padding: 5px 10px; border-radius: 3px;">
                        {signal_type.upper()}
                    </span>
                </td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Z-Score</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{zscore:.3f}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Action</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{action}</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><strong>Time</strong></td>
                <td style="padding: 10px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
        </table>
    </div>
    <div style="padding: 10px; text-align: center; color: #666; font-size: 12px;">
        Statistical Arbitrage Simulator - Automated Alert
    </div>
</body>
</html>
        """
        
        return self.send_email(to_email, subject, body, html_body)
    
    def send_backtest_report(
        self,
        to_email: str,
        pair: str,
        results: Dict
    ) -> bool:
        """Send backtest results report."""
        subject = f"[Stat Arb] Backtest Results: {pair}"
        
        body = f"""
Backtest Results Report
========================

Pair: {pair}
Total Return: {results.get('total_return', 0):.2%}
Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}
Max Drawdown: {results.get('max_drawdown', 0):.2%}
Total Trades: {results.get('total_trades', 0)}
Win Rate: {results.get('win_rate', 0):.1%}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_email(to_email, subject, body)


class AlertManager:
    """Manage and monitor trading alerts."""
    
    def __init__(self, notifier: Optional[EmailNotifier] = None):
        self.notifier = notifier or EmailNotifier()
        self.active_alerts: List[Alert] = []
    
    def load_alerts(self, user_id: Optional[int] = None):
        """Load active alerts from database."""
        with get_db() as db:
            self.active_alerts = AlertRepository.get_active_alerts(db, user_id)
    
    def create_alert(
        self,
        user_id: int,
        pair: str,
        alert_type: str,
        condition: str,
        threshold: float,
        notification_method: str = "email"
    ) -> Alert:
        """Create a new alert."""
        with get_db() as db:
            alert = AlertRepository.create_alert(
                db, user_id, pair, alert_type, condition, threshold, notification_method
            )
            self.active_alerts.append(alert)
            logger.info(f"Created alert for {pair}: {alert_type} {condition} {threshold}")
            return alert
    
    def check_alerts(self, current_data: Dict[str, float]) -> List[Dict]:
        """
        Check all active alerts against current data.
        
        Args:
            current_data: Dict mapping pair names to current z-scores
        
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for alert in self.active_alerts:
            if not alert.is_active or alert.is_triggered:
                continue
            
            current_value = current_data.get(alert.pair)
            if current_value is None:
                continue
            
            should_trigger = False
            
            if alert.condition == "above":
                should_trigger = current_value > alert.threshold
            elif alert.condition == "below":
                should_trigger = current_value < alert.threshold
            elif alert.condition == "crosses":
                # This would need historical data to properly implement
                should_trigger = abs(current_value - alert.threshold) < 0.1
            
            if should_trigger:
                triggered.append({
                    "alert": alert,
                    "current_value": current_value,
                    "threshold": alert.threshold,
                    "triggered_at": datetime.now()
                })
                
                # Mark as triggered in database
                with get_db() as db:
                    AlertRepository.trigger_alert(db, alert.id)
                
                # Send notification
                self._send_notification(alert, current_value)
        
        return triggered
    
    def _send_notification(self, alert: Alert, current_value: float):
        """Send notification for triggered alert."""
        if alert.notification_method == "email" and NOTIFICATION_EMAIL:
            action = f"Z-Score {alert.condition} {alert.threshold:.2f} (current: {current_value:.2f})"
            self.notifier.send_signal_alert(
                NOTIFICATION_EMAIL,
                alert.pair,
                alert.alert_type,
                current_value,
                action
            )
            logger.info(f"Sent notification for alert {alert.id}")


# Webhook support for external integrations
class WebhookNotifier:
    """Send notifications via webhooks (Discord, Slack, etc.)."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
    
    def send_discord(self, message: str, embed: Optional[Dict] = None) -> bool:
        """Send a Discord webhook notification."""
        if not self.webhook_url:
            return False
        
        try:
            import requests
            
            payload = {"content": message}
            if embed:
                payload["embeds"] = [embed]
            
            response = requests.post(self.webhook_url, json=payload)
            return response.status_code == 204
            
        except Exception as e:
            logger.error(f"Failed to send Discord webhook: {e}")
            return False
    
    def send_slack(self, message: str, blocks: Optional[List] = None) -> bool:
        """Send a Slack webhook notification."""
        if not self.webhook_url:
            return False
        
        try:
            import requests
            
            payload = {"text": message}
            if blocks:
                payload["blocks"] = blocks
            
            response = requests.post(self.webhook_url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to send Slack webhook: {e}")
            return False


# Singleton instances
email_notifier = EmailNotifier()
alert_manager = AlertManager(email_notifier)
