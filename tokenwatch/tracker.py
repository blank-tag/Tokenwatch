"""
Main CostTracker class — the primary interface for tokenwatch.
"""

from __future__ import annotations

import functools
import uuid
from typing import Any, Callable, Dict, Optional, TypeVar

from .alerts import AlertManager, ALERT_COST_SPIKE, ALERT_DAILY_SUMMARY
from .budget import Budget, BudgetExceededError
from .pricing.tables import get_cost
from .storage import Storage

F = TypeVar("F", bound=Callable[..., Any])


class CostTracker:
    """
    Track LLM API costs across OpenAI, Anthropic, and Gemini.

    Usage::

        tracker = CostTracker(
            budget=Budget(daily_limit=1.00, on_exceed="warn"),
            auto_alert=True,
        )
        client = tracker.wrap_openai(openai.OpenAI(api_key="..."))
        response = client.chat.completions.create(model="gpt-4o-mini", ...)
        print(f"Session cost: ${tracker.get_session_cost():.4f}")

    Context manager::

        with CostTracker() as tracker:
            client = tracker.wrap_openai(openai.OpenAI(api_key="..."))
            ...
        # Prints summary on exit
    """

    def __init__(
        self,
        budget: Optional[Budget] = None,
        session_id: Optional[str] = None,
        db_path: Optional[str] = None,
        alert_manager: Optional[AlertManager] = None,
        auto_alert: bool = True,
        spike_threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            budget:          Optional Budget configuration for limits and alerts.
            session_id:      Session identifier; auto-generated UUID if None.
            db_path:         Path to SQLite database file.
            alert_manager:   Custom AlertManager; if None and auto_alert=True,
                             a console AlertManager is created automatically.
            auto_alert:      If True, create a default console AlertManager when
                             none is provided.
            spike_threshold: If set, fire a "cost_spike" alert when a single
                             call exceeds this USD amount.
        """
        self.session_id: str = session_id or str(uuid.uuid4())
        self.budget = budget
        self.spike_threshold = spike_threshold
        self.storage = Storage(db_path=db_path)

        if alert_manager is not None:
            self.alert_manager = alert_manager
        elif auto_alert:
            self.alert_manager = AlertManager()
            self.alert_manager.add_console_handler(level="WARNING")
        else:
            self.alert_manager = AlertManager()

    # ------------------------------------------------------------------
    # Provider wrapping
    # ------------------------------------------------------------------

    def wrap_openai(self, client: Any) -> Any:
        """
        Wrap an OpenAI client with cost tracking.

        Returns the same client object with ``chat.completions.create``
        intercepted for tracking. Your existing code needs no changes.
        """
        from .providers.openai_provider import OpenAIProvider
        return OpenAIProvider(self).wrap(client)

    def wrap_anthropic(self, client: Any) -> Any:
        """
        Wrap an Anthropic client with cost tracking.

        Returns the same client with ``messages.create`` intercepted.
        """
        from .providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(self).wrap(client)

    def wrap_gemini(self, client_or_model: Any) -> Any:
        """
        Wrap a Gemini GenerativeModel or client with cost tracking.

        Returns the same object with ``generate_content`` intercepted.
        """
        from .providers.gemini_provider import GeminiProvider
        return GeminiProvider(self).wrap(client_or_model)

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def watch(self, fn: F) -> F:
        """
        Decorator that ensures all LLM calls made within ``fn`` are tracked.

        All provider clients used inside the function must already be wrapped
        (via wrap_openai / wrap_anthropic / wrap_gemini).
        The decorator doesn't add extra wrapping — it just associates a
        consistent session_id and prints a per-call summary.

        Example::

            @tracker.watch
            def my_pipeline():
                client.chat.completions.create(...)
        """
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cost_before = self.get_session_cost()
            result = fn(*args, **kwargs)
            cost_after = self.get_session_cost()
            delta = cost_after - cost_before
            if delta > 0:
                print(
                    f"[tokenwatch] {fn.__name__} used ${delta:.6f} "
                    f"(session total: ${cost_after:.6f})"
                )
            return result

        return wrapper  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Cost queries
    # ------------------------------------------------------------------

    def get_session_cost(self) -> float:
        """Return total cost for the current session."""
        return self.storage.get_spend(period="session", session_id=self.session_id)

    def get_daily_cost(self) -> float:
        """Return total cost for today (UTC)."""
        return self.storage.get_spend(period="daily")

    def get_monthly_cost(self) -> float:
        """Return total cost for the current month (UTC)."""
        return self.storage.get_spend(period="monthly")

    def get_total_cost(self) -> float:
        """Return all-time total cost."""
        return self.storage.get_spend(period="total")

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a full cost breakdown dict.

        Keys: session_cost, daily_cost, monthly_cost, total_cost,
              session_id, call_count_today, call_count_session,
              provider_breakdown (dict provider -> daily spend)
        """
        return {
            "session_id": self.session_id,
            "session_cost": self.get_session_cost(),
            "daily_cost": self.get_daily_cost(),
            "monthly_cost": self.get_monthly_cost(),
            "total_cost": self.get_total_cost(),
            "call_count_today": self.storage.get_call_count(period="daily"),
            "call_count_session": self.storage.get_call_count(
                period="session", session_id=self.session_id
            ),
            "provider_breakdown": self.storage.get_spend_by_provider(period="daily"),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_report(self, filepath: str) -> None:
        """Export full call history to a CSV file."""
        self.storage.export_csv(filepath)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CostTracker":
        return self

    def __exit__(self, *args: Any) -> None:
        summary = self.get_summary()
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table(title="TokenWatch — Session Summary", show_header=True)
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="bold green")
            table.add_row("Session ID", summary["session_id"][:8] + "...")
            table.add_row("Session Cost", f"${summary['session_cost']:.6f}")
            table.add_row("Daily Cost", f"${summary['daily_cost']:.6f}")
            table.add_row("Monthly Cost", f"${summary['monthly_cost']:.6f}")
            table.add_row("Total Cost", f"${summary['total_cost']:.6f}")
            table.add_row("Calls Today", str(summary["call_count_today"]))
            table.add_row("Calls This Session", str(summary["call_count_session"]))
            console.print(table)
        except ImportError:
            print("\n=== TokenWatch Session Summary ===")
            print(f"  Session ID     : {summary['session_id'][:8]}...")
            print(f"  Session Cost   : ${summary['session_cost']:.6f}")
            print(f"  Daily Cost     : ${summary['daily_cost']:.6f}")
            print(f"  Monthly Cost   : ${summary['monthly_cost']:.6f}")
            print(f"  Total Cost     : ${summary['total_cost']:.6f}")
            print(f"  Calls Today    : {summary['call_count_today']}")
            print(f"  Calls Session  : {summary['call_count_session']}")
            print("=================================\n")

    # ------------------------------------------------------------------
    # Internal recording
    # ------------------------------------------------------------------

    def _record_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute cost, persist to storage, check budget, and fire alerts.

        Returns:
            Computed cost in USD.
        """
        # Compute cost — fall back gracefully if model is unknown
        try:
            cost = get_cost(provider, model, input_tokens, output_tokens)
        except ValueError:
            cost = 0.0

        # Persist
        self.storage.log_call(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            session_id=self.session_id,
            metadata=metadata,
        )

        # Spike detection
        if self.spike_threshold is not None and cost > self.spike_threshold:
            self.alert_manager.fire(
                ALERT_COST_SPIKE,
                f"Single call cost ${cost:.4f} exceeds spike threshold ${self.spike_threshold:.4f}",
                {
                    "provider": provider,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                    "spike_threshold": self.spike_threshold,
                },
            )

        # Budget check
        if self.budget is not None and self.budget.has_any_limit():
            try:
                self.budget.check(
                    storage=self.storage,
                    alert_manager=self.alert_manager,
                    session_id=self.session_id,
                )
            except BudgetExceededError:
                raise  # re-raise so caller sees it

        return cost

    def fire_daily_summary(self) -> None:
        """Manually fire a daily summary alert."""
        summary = self.get_summary()
        self.alert_manager.fire(
            ALERT_DAILY_SUMMARY,
            f"Daily LLM spend: ${summary['daily_cost']:.4f}",
            {
                "daily_cost": summary["daily_cost"],
                "monthly_cost": summary["monthly_cost"],
                "total_cost": summary["total_cost"],
                "calls_today": summary["call_count_today"],
                "provider_breakdown": summary["provider_breakdown"],
            },
        )
