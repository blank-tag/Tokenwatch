"""
Alert system for LLM cost monitoring.
Supports console, webhook, email, and custom callback handlers.
"""

from __future__ import annotations

import json
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    _RICH_AVAILABLE = True
    _rich_console = Console(stderr=True)
except ImportError:
    _RICH_AVAILABLE = False
    _rich_console = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Alert type constants
# ---------------------------------------------------------------------------

ALERT_BUDGET_WARNING = "budget_warning"
ALERT_BUDGET_EXCEEDED = "budget_exceeded"
ALERT_COST_SPIKE = "cost_spike"
ALERT_DAILY_SUMMARY = "daily_summary"

_ALERT_COLORS = {
    ALERT_BUDGET_WARNING: "yellow",
    ALERT_BUDGET_EXCEEDED: "red",
    ALERT_COST_SPIKE: "magenta",
    ALERT_DAILY_SUMMARY: "cyan",
}

_ALERT_LEVELS = {
    ALERT_BUDGET_WARNING: "WARNING",
    ALERT_BUDGET_EXCEEDED: "ERROR",
    ALERT_COST_SPIKE: "WARNING",
    ALERT_DAILY_SUMMARY: "INFO",
}


# ---------------------------------------------------------------------------
# Handler factory helpers
# ---------------------------------------------------------------------------

def _make_console_handler(level: str = "WARNING") -> Callable[[str, str, Dict[str, Any]], None]:
    """Return a handler that prints colored alerts to the console."""
    level = level.upper()
    level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    min_level = level_order.get(level, 1)

    def handler(alert_type: str, message: str, data: Dict[str, Any]) -> None:
        this_level = _ALERT_LEVELS.get(alert_type, "INFO")
        if level_order.get(this_level, 1) < min_level:
            return

        color = _ALERT_COLORS.get(alert_type, "white")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if _RICH_AVAILABLE and _rich_console is not None:
            title = f"[bold {color}]TokenWatch · {alert_type.upper()}[/bold {color}]"
            body = f"[{color}]{message}[/{color}]\n"
            if data:
                for k, v in data.items():
                    if isinstance(v, float):
                        body += f"  [dim]{k}:[/dim] [bold]{v:.4f}[/bold]\n"
                    else:
                        body += f"  [dim]{k}:[/dim] {v}\n"
            _rich_console.print(
                Panel(body.rstrip(), title=title, subtitle=f"[dim]{timestamp}[/dim]",
                      border_style=color)
            )
        else:
            print(f"[{timestamp}] [{this_level}] TokenWatch - {alert_type}: {message}")
            if data:
                for k, v in data.items():
                    print(f"  {k}: {v}")

    return handler


def _make_webhook_handler(
    url: str,
    headers: Optional[Dict[str, str]] = None,
) -> Callable[[str, str, Dict[str, Any]], None]:
    """Return a handler that POSTs alert data to a webhook URL."""

    def handler(alert_type: str, message: str, data: Dict[str, Any]) -> None:
        if not _HTTPX_AVAILABLE:
            print(
                "tokenwatch: httpx is required for webhook alerts. "
                "Install it with: pip install httpx"
            )
            return

        payload = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        try:
            httpx.post(
                url,
                json=payload,
                headers=headers or {},
                timeout=5.0,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"tokenwatch: webhook alert failed: {exc}")

    return handler


def _make_email_handler(
    smtp_config: Dict[str, Any],
    to_email: str,
) -> Callable[[str, str, Dict[str, Any]], None]:
    """
    Return a handler that sends email alerts.

    smtp_config keys:
        host, port, username, password, use_tls (bool, default True),
        from_email (optional, defaults to username)
    """

    def handler(alert_type: str, message: str, data: Dict[str, Any]) -> None:
        host = smtp_config.get("host", "smtp.gmail.com")
        port = int(smtp_config.get("port", 587))
        username = smtp_config.get("username", "")
        password = smtp_config.get("password", "")
        use_tls = smtp_config.get("use_tls", True)
        from_email = smtp_config.get("from_email", username)

        subject = f"TokenWatch Alert: {alert_type.upper()}"
        body_lines = [
            f"Alert Type: {alert_type}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Message: {message}",
            "",
            "Details:",
        ]
        for k, v in data.items():
            if isinstance(v, float):
                body_lines.append(f"  {k}: {v:.6f}")
            else:
                body_lines.append(f"  {k}: {v}")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.attach(MIMEText("\n".join(body_lines), "plain"))

        try:
            if use_tls:
                server = smtplib.SMTP(host, port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(host, port)  # type: ignore[assignment]
            server.login(username, password)
            server.sendmail(from_email, to_email, msg.as_string())
            server.quit()
        except Exception as exc:  # noqa: BLE001
            print(f"tokenwatch: email alert failed: {exc}")

    return handler


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Manages multiple alert handlers and dispatches alerts to all of them.

    Example:
        am = AlertManager()
        am.add_console_handler()
        am.add_webhook_handler("https://hooks.slack.com/...")
        am.fire("budget_warning", "80% of daily budget used", {"spent": 0.80})
    """

    def __init__(self) -> None:
        self.handlers: List[Callable[[str, str, Dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def add_console_handler(self, level: str = "WARNING") -> "AlertManager":
        """Add a handler that prints colored warnings to the console."""
        with self._lock:
            self.handlers.append(_make_console_handler(level))
        return self

    def add_webhook_handler(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> "AlertManager":
        """Add a handler that HTTP POSTs alerts to a webhook URL."""
        with self._lock:
            self.handlers.append(_make_webhook_handler(url, headers))
        return self

    def add_callback_handler(
        self,
        fn: Callable[[str, str, Dict[str, Any]], None],
    ) -> "AlertManager":
        """Add a custom Python callable as an alert handler."""
        with self._lock:
            self.handlers.append(fn)
        return self

    def add_email_handler(
        self,
        smtp_config: Dict[str, Any],
        to_email: str,
    ) -> "AlertManager":
        """Add a handler that sends email alerts via SMTP."""
        with self._lock:
            self.handlers.append(_make_email_handler(smtp_config, to_email))
        return self

    def remove_all_handlers(self) -> None:
        """Remove all registered handlers."""
        with self._lock:
            self.handlers.clear()

    # ------------------------------------------------------------------
    # Dispatching
    # ------------------------------------------------------------------

    def fire(
        self,
        alert_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Dispatch an alert to all registered handlers.

        Args:
            alert_type: One of the ALERT_* constants.
            message: Human-readable alert message.
            data: Optional dict with additional context.
        """
        data = data or {}
        with self._lock:
            handlers = list(self.handlers)

        for handler in handlers:
            try:
                handler(alert_type, message, data)
            except Exception as exc:  # noqa: BLE001
                print(f"tokenwatch: alert handler raised an exception: {exc}")
