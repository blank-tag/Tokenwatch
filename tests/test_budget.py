"""
Tests for Budget and BudgetStatus.
"""

from __future__ import annotations

import pytest

from tokenwatch import AlertManager, Budget, BudgetExceededError
from tokenwatch.budget import BudgetStatus
from tokenwatch.storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def alert_manager():
    return AlertManager()


class TestBudgetInit:
    def test_defaults(self):
        b = Budget()
        assert b.daily_limit is None
        assert b.monthly_limit is None
        assert b.session_limit is None
        assert b.total_limit is None
        assert b.alert_threshold == 0.80
        assert b.on_exceed == "warn"

    def test_custom_values(self):
        b = Budget(
            daily_limit=5.0,
            monthly_limit=50.0,
            session_limit=1.0,
            alert_threshold=0.90,
            on_exceed="raise",
        )
        assert b.daily_limit == 5.0
        assert b.monthly_limit == 50.0
        assert b.session_limit == 1.0
        assert b.alert_threshold == 0.90
        assert b.on_exceed == "raise"

    def test_has_any_limit_false_when_no_limits(self):
        b = Budget()
        assert b.has_any_limit() is False

    def test_has_any_limit_true_when_limit_set(self):
        b = Budget(daily_limit=1.0)
        assert b.has_any_limit() is True


class TestBudgetCheck:
    def test_within_budget_returns_true(self, storage, alert_manager):
        b = Budget(daily_limit=100.0)
        storage.log_call("openai", "gpt-4o", 100, 50, 0.10, session_id="s1")
        status = b.check(storage, alert_manager, session_id="s1")
        assert status.within_budget is True
        assert len(status.exceeded) == 0

    def test_exceeded_sets_within_budget_false(self, storage, alert_manager):
        b = Budget(daily_limit=0.05, on_exceed="warn")
        storage.log_call("openai", "gpt-4o", 100, 50, 0.10, session_id="s1")
        status = b.check(storage, alert_manager, session_id="s1")
        assert status.within_budget is False
        assert len(status.exceeded) > 0

    def test_threshold_fires_warning(self, storage, alert_manager):
        fired = []
        alert_manager.add_callback_handler(lambda t, m, d: fired.append(t))
        b = Budget(daily_limit=1.0, alert_threshold=0.05)  # fire at 5%
        storage.log_call("openai", "gpt-4o", 100, 50, 0.10, session_id="s1")
        b.check(storage, alert_manager, session_id="s1")
        assert "budget_warning" in fired

    def test_raise_on_exceed_raises_error(self, storage, alert_manager):
        b = Budget(session_limit=0.001, on_exceed="raise")
        storage.log_call("openai", "gpt-4o", 100, 50, 0.10, session_id="sess")
        with pytest.raises(BudgetExceededError) as exc_info:
            b.check(storage, alert_manager, session_id="sess")
        assert exc_info.value.period == "session"
        assert exc_info.value.limit == 0.001
        assert exc_info.value.spent >= 0.10

    def test_warn_mode_does_not_raise(self, storage, alert_manager):
        b = Budget(session_limit=0.001, on_exceed="warn")
        storage.log_call("openai", "gpt-4o", 100, 50, 0.10, session_id="sess")
        # Should not raise
        status = b.check(storage, alert_manager, session_id="sess")
        assert not status.within_budget

    def test_multiple_limits_checked(self, storage, alert_manager):
        fired = []
        alert_manager.add_callback_handler(lambda t, m, d: fired.append((t, d.get("period"))))

        b = Budget(
            daily_limit=0.001,   # exceeded
            monthly_limit=100.0,  # not exceeded
            session_limit=0.001,  # exceeded
            on_exceed="warn",
        )
        storage.log_call("openai", "gpt-4o", 100, 50, 0.10, session_id="s1")
        status = b.check(storage, alert_manager, session_id="s1")

        assert not status.within_budget
        # At least 1 exceeded event was fired for daily or session
        exceeded_events = [t for t, p in fired if t == "budget_exceeded"]
        assert len(exceeded_events) >= 1

    def test_daily_percent_computed(self, storage, alert_manager):
        b = Budget(daily_limit=1.0)
        storage.log_call("openai", "gpt-4o", 100, 50, 0.50)
        status = b.check(storage, alert_manager)
        assert status.daily_percent == pytest.approx(0.50)

    def test_session_percent_computed(self, storage, alert_manager):
        b = Budget(session_limit=1.0)
        storage.log_call("openai", "gpt-4o", 100, 50, 0.25, session_id="sess-x")
        status = b.check(storage, alert_manager, session_id="sess-x")
        assert status.session_percent == pytest.approx(0.25)

    def test_no_limits_no_check(self, storage, alert_manager):
        b = Budget()  # no limits
        fired = []
        alert_manager.add_callback_handler(lambda t, m, d: fired.append(t))
        storage.log_call("openai", "gpt-4o", 100, 50, 99.99)
        status = b.check(storage, alert_manager)
        assert status.within_budget is True
        assert fired == []

    def test_budget_exceeded_error_attributes(self, storage, alert_manager):
        b = Budget(daily_limit=0.01, on_exceed="raise")
        storage.log_call("openai", "gpt-4o", 100, 50, 0.50)
        with pytest.raises(BudgetExceededError) as exc_info:
            b.check(storage, alert_manager)
        err = exc_info.value
        assert err.period == "daily"
        assert err.limit == pytest.approx(0.01)
        assert err.spent >= 0.50
        assert "daily" in str(err).lower() or "budget" in str(err).lower()


class TestBudgetStatus:
    def test_str_representation(self):
        s = BudgetStatus(
            within_budget=True,
            daily_percent=0.45,
            monthly_percent=0.12,
        )
        text = str(s)
        assert "OK" in text
        assert "daily=45.0%" in text
        assert "monthly=12.0%" in text

    def test_any_exceeded_property(self):
        s = BudgetStatus(within_budget=False, exceeded=["daily exceeded"])
        assert s.any_exceeded is True

        s2 = BudgetStatus(within_budget=True)
        assert s2.any_exceeded is False
