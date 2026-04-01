"""
Budget management and enforcement for LLM cost tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional

if TYPE_CHECKING:
    from .alerts import AlertManager
    from .storage import Storage


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    """Raised when an LLM call would exceed a configured budget limit."""

    def __init__(self, message: str, period: str, limit: float, spent: float) -> None:
        super().__init__(message)
        self.period = period
        self.limit = limit
        self.spent = spent


# ---------------------------------------------------------------------------
# BudgetStatus
# ---------------------------------------------------------------------------

@dataclass
class BudgetStatus:
    """Result of a budget check."""
    within_budget: bool
    daily_percent: Optional[float] = None      # 0.0 – 1.0+ (>1 means exceeded)
    monthly_percent: Optional[float] = None
    session_percent: Optional[float] = None
    total_percent: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    exceeded: List[str] = field(default_factory=list)

    @property
    def any_exceeded(self) -> bool:
        return bool(self.exceeded)

    def __str__(self) -> str:
        parts = []
        if self.daily_percent is not None:
            parts.append(f"daily={self.daily_percent*100:.1f}%")
        if self.monthly_percent is not None:
            parts.append(f"monthly={self.monthly_percent*100:.1f}%")
        if self.session_percent is not None:
            parts.append(f"session={self.session_percent*100:.1f}%")
        if self.total_percent is not None:
            parts.append(f"total={self.total_percent*100:.1f}%")
        status = "OK" if self.within_budget else "EXCEEDED"
        return f"BudgetStatus({status}, {', '.join(parts)})"


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class Budget:
    """
    Configures spending limits and enforcement policy.

    Args:
        daily_limit:    Maximum spend per calendar day (UTC) in USD.
        monthly_limit:  Maximum spend per calendar month (UTC) in USD.
        session_limit:  Maximum spend per session in USD.
        total_limit:    Maximum all-time spend in USD.
        alert_threshold: Fraction of any limit at which a warning alert fires (default 0.80).
        on_exceed:      Policy when a limit is exceeded:
                        "warn"  – fire alert and continue (default)
                        "raise" – raise BudgetExceededError
                        "block" – alias for "raise"
    """

    def __init__(
        self,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None,
        session_limit: Optional[float] = None,
        total_limit: Optional[float] = None,
        alert_threshold: float = 0.80,
        on_exceed: Literal["warn", "raise", "block"] = "warn",
    ) -> None:
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.session_limit = session_limit
        self.total_limit = total_limit
        self.alert_threshold = alert_threshold
        self.on_exceed = on_exceed

    # ------------------------------------------------------------------

    def check(
        self,
        storage: "Storage",
        alert_manager: "AlertManager",
        session_id: Optional[str] = None,
    ) -> BudgetStatus:
        """
        Check current spend against configured limits.

        Fires alerts for threshold warnings and/or exceeded limits.
        Raises BudgetExceededError if on_exceed is "raise"/"block".

        Returns:
            BudgetStatus with current percentages and any warning/exceeded messages.
        """
        from .alerts import ALERT_BUDGET_WARNING, ALERT_BUDGET_EXCEEDED

        warnings: List[str] = []
        exceeded: List[str] = []
        within_budget = True

        daily_pct = monthly_pct = session_pct = total_pct = None

        checks = [
            ("daily",   self.daily_limit,   "daily"),
            ("monthly", self.monthly_limit, "monthly"),
            ("session", self.session_limit, "session"),
            ("total",   self.total_limit,   "total"),
        ]

        for period, limit, label in checks:
            if limit is None:
                continue

            spent = storage.get_spend(period=period, session_id=session_id)
            pct = spent / limit if limit > 0 else 0.0

            if period == "daily":
                daily_pct = pct
            elif period == "monthly":
                monthly_pct = pct
            elif period == "session":
                session_pct = pct
            elif period == "total":
                total_pct = pct

            if pct >= 1.0:
                within_budget = False
                msg = (
                    f"{label.capitalize()} budget exceeded: "
                    f"${spent:.4f} spent of ${limit:.2f} limit "
                    f"({pct*100:.1f}%)"
                )
                exceeded.append(msg)
                alert_manager.fire(
                    ALERT_BUDGET_EXCEEDED,
                    msg,
                    {
                        "period": label,
                        "spent_usd": spent,
                        "limit_usd": limit,
                        "percent": pct,
                    },
                )
            elif pct >= self.alert_threshold:
                msg = (
                    f"{label.capitalize()} budget at {pct*100:.1f}%: "
                    f"${spent:.4f} of ${limit:.2f}"
                )
                warnings.append(msg)
                alert_manager.fire(
                    ALERT_BUDGET_WARNING,
                    msg,
                    {
                        "period": label,
                        "spent_usd": spent,
                        "limit_usd": limit,
                        "percent": pct,
                        "threshold": self.alert_threshold,
                    },
                )

        status = BudgetStatus(
            within_budget=within_budget,
            daily_percent=daily_pct,
            monthly_percent=monthly_pct,
            session_percent=session_pct,
            total_percent=total_pct,
            warnings=warnings,
            exceeded=exceeded,
        )

        if not within_budget and self.on_exceed in ("raise", "block"):
            first_exceeded = exceeded[0] if exceeded else "Budget exceeded"
            # Parse period from the first exceeded message for the exception
            period_name = exceeded[0].split(" ")[0].lower() if exceeded else "unknown"
            # Find matching limit/spent for the exception
            for period, limit, label in checks:
                if limit is None:
                    continue
                spent = storage.get_spend(period=period, session_id=session_id)
                if spent >= limit:
                    raise BudgetExceededError(
                        first_exceeded,
                        period=label,
                        limit=limit,
                        spent=spent,
                    )

        return status

    def has_any_limit(self) -> bool:
        """Return True if at least one limit is configured."""
        return any(
            x is not None
            for x in (self.daily_limit, self.monthly_limit, self.session_limit, self.total_limit)
        )
