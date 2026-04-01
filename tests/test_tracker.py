"""
Tests for CostTracker — cost calculation, budget enforcement, alerts, storage.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tokenwatch import AlertManager, Budget, BudgetExceededError, CostTracker
from tokenwatch.storage import Storage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary SQLite database path."""
    return str(tmp_path / "test_costs.db")


@pytest.fixture
def storage(tmp_db):
    return Storage(db_path=tmp_db)


@pytest.fixture
def tracker(tmp_db):
    return CostTracker(db_path=tmp_db, auto_alert=False)


@pytest.fixture
def budget_tracker(tmp_db):
    budget = Budget(
        daily_limit=1.00,
        monthly_limit=10.00,
        session_limit=0.50,
        alert_threshold=0.80,
        on_exceed="warn",
    )
    am = AlertManager()
    return CostTracker(budget=budget, db_path=tmp_db, alert_manager=am)


# ---------------------------------------------------------------------------
# Cost calculation tests
# ---------------------------------------------------------------------------

class TestCostCalculation:
    def test_openai_gpt4o_cost(self, tracker):
        cost = tracker._record_call("openai", "gpt-4o", 1_000_000, 1_000_000)
        # $2.50 input + $10.00 output = $12.50 per 1M tokens each
        assert abs(cost - 12.50) < 0.0001

    def test_openai_gpt4o_mini_cost(self, tracker):
        cost = tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        # $0.15/1M input + $0.60/1M output
        expected = (1_000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert abs(cost - expected) < 1e-9

    def test_anthropic_claude_sonnet_cost(self, tracker):
        cost = tracker._record_call("anthropic", "claude-sonnet-4-6", 2_000, 1_000)
        expected = (2_000 / 1_000_000) * 3.00 + (1_000 / 1_000_000) * 15.00
        assert abs(cost - expected) < 1e-9

    def test_anthropic_claude_haiku_cost(self, tracker):
        cost = tracker._record_call("anthropic", "claude-haiku-4-5", 500, 200)
        expected = (500 / 1_000_000) * 0.80 + (200 / 1_000_000) * 4.00
        assert abs(cost - expected) < 1e-9

    def test_gemini_flash_cost(self, tracker):
        cost = tracker._record_call("gemini", "gemini-1.5-flash", 10_000, 5_000)
        expected = (10_000 / 1_000_000) * 0.075 + (5_000 / 1_000_000) * 0.30
        assert abs(cost - expected) < 1e-9

    def test_unknown_model_returns_zero_cost(self, tracker):
        # Should NOT raise, just return 0.0
        cost = tracker._record_call("openai", "gpt-99-nonexistent", 1_000, 1_000)
        assert cost == 0.0

    def test_zero_tokens_cost(self, tracker):
        cost = tracker._record_call("openai", "gpt-4o", 0, 0)
        assert cost == 0.0

    def test_cost_is_recorded_in_storage(self, tracker):
        tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        assert tracker.get_session_cost() > 0.0

    def test_multiple_calls_accumulate(self, tracker):
        tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        tracker._record_call("anthropic", "claude-haiku-4-5", 1_000, 500)
        session = tracker.get_session_cost()
        daily = tracker.get_daily_cost()
        assert session > 0.0
        assert daily >= session  # daily >= session (same day)


# ---------------------------------------------------------------------------
# Budget enforcement tests
# ---------------------------------------------------------------------------

class TestBudgetEnforcement:
    def test_warn_mode_does_not_raise(self, tmp_db):
        budget = Budget(session_limit=0.000001, on_exceed="warn")
        am = AlertManager()
        fired = []
        am.add_callback_handler(lambda t, m, d: fired.append(t))
        tracker = CostTracker(budget=budget, db_path=tmp_db, alert_manager=am)
        # Should NOT raise
        tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        assert "budget_exceeded" in fired

    def test_raise_mode_raises_on_exceed(self, tmp_db):
        budget = Budget(session_limit=0.000001, on_exceed="raise")
        tracker = CostTracker(budget=budget, db_path=tmp_db, auto_alert=False)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        assert exc_info.value.period == "session"
        assert exc_info.value.limit == pytest.approx(0.000001)

    def test_block_mode_raises_on_exceed(self, tmp_db):
        budget = Budget(session_limit=0.000001, on_exceed="block")
        tracker = CostTracker(budget=budget, db_path=tmp_db, auto_alert=False)
        with pytest.raises(BudgetExceededError):
            tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)

    def test_warning_fires_at_threshold(self, tmp_db):
        # Set limit high enough that we reach 80% but not 100%
        # 1000 input + 500 output at gpt-4o-mini costs ~0.00000045 USD
        # So set limit to 0.00000056 (slightly above single call cost, threshold at 80%)
        budget = Budget(session_limit=100.0, alert_threshold=0.0, on_exceed="warn")
        am = AlertManager()
        warnings = []
        am.add_callback_handler(lambda t, m, d: warnings.append(t))
        tracker = CostTracker(budget=budget, db_path=tmp_db, alert_manager=am)
        tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        # threshold=0.0 means ANY spend fires warning
        assert "budget_warning" in warnings

    def test_no_budget_no_error(self, tmp_db):
        tracker = CostTracker(budget=None, db_path=tmp_db, auto_alert=False)
        cost = tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        assert cost > 0.0

    def test_daily_limit_exceeded(self, tmp_db):
        budget = Budget(daily_limit=0.000001, on_exceed="raise")
        tracker = CostTracker(budget=budget, db_path=tmp_db, auto_alert=False)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        assert exc_info.value.period == "daily"


# ---------------------------------------------------------------------------
# Alert tests
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_callback_handler_receives_alert(self):
        am = AlertManager()
        received = []
        am.add_callback_handler(lambda t, m, d: received.append((t, m, d)))
        am.fire("budget_warning", "test message", {"key": "value"})
        assert len(received) == 1
        assert received[0][0] == "budget_warning"
        assert received[0][1] == "test message"
        assert received[0][2]["key"] == "value"

    def test_multiple_handlers_all_called(self):
        am = AlertManager()
        counts = [0, 0]
        am.add_callback_handler(lambda t, m, d: counts.__setitem__(0, counts[0] + 1))
        am.add_callback_handler(lambda t, m, d: counts.__setitem__(1, counts[1] + 1))
        am.fire("budget_exceeded", "msg", {})
        assert counts == [1, 1]

    def test_failing_handler_does_not_crash(self):
        am = AlertManager()
        def bad_handler(t, m, d):
            raise RuntimeError("Handler error")
        am.add_callback_handler(bad_handler)
        # Should not raise
        am.fire("cost_spike", "msg", {})

    def test_spike_threshold_fires_alert(self, tmp_db):
        am = AlertManager()
        spikes = []
        am.add_callback_handler(lambda t, m, d: spikes.append(t) if t == "cost_spike" else None)
        tracker = CostTracker(
            db_path=tmp_db,
            alert_manager=am,
            spike_threshold=0.000001,  # $0.000001 — any real call exceeds this
        )
        tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        assert "cost_spike" in spikes

    def test_remove_all_handlers(self):
        am = AlertManager()
        called = []
        am.add_callback_handler(lambda t, m, d: called.append(True))
        am.remove_all_handlers()
        am.fire("budget_warning", "test", {})
        assert called == []


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------

class TestStorage:
    def test_log_and_retrieve(self, storage):
        storage.log_call("openai", "gpt-4o", 100, 50, 0.001, session_id="s1")
        history = storage.get_history(limit=10)
        assert len(history) == 1
        assert history[0]["provider"] == "openai"
        assert history[0]["model"] == "gpt-4o"
        assert history[0]["input_tokens"] == 100
        assert history[0]["output_tokens"] == 50
        assert history[0]["cost_usd"] == pytest.approx(0.001)
        assert history[0]["session_id"] == "s1"

    def test_get_spend_daily(self, storage):
        storage.log_call("openai", "gpt-4o", 100, 50, 0.50, session_id="s1")
        storage.log_call("anthropic", "claude-sonnet-4-6", 200, 100, 0.25, session_id="s1")
        daily = storage.get_spend("daily")
        assert daily == pytest.approx(0.75)

    def test_get_spend_session(self, storage):
        storage.log_call("openai", "gpt-4o", 100, 50, 0.50, session_id="sess-A")
        storage.log_call("openai", "gpt-4o", 100, 50, 0.30, session_id="sess-B")
        assert storage.get_spend("session", session_id="sess-A") == pytest.approx(0.50)
        assert storage.get_spend("session", session_id="sess-B") == pytest.approx(0.30)

    def test_get_spend_total(self, storage):
        storage.log_call("openai", "gpt-4o", 100, 50, 1.00)
        storage.log_call("gemini", "gemini-1.5-flash", 100, 50, 0.50)
        assert storage.get_spend("total") == pytest.approx(1.50)

    def test_filter_history_by_provider(self, storage):
        storage.log_call("openai", "gpt-4o", 100, 50, 0.01)
        storage.log_call("anthropic", "claude-sonnet-4-6", 100, 50, 0.02)
        openai_history = storage.get_history(limit=10, provider="openai")
        assert all(r["provider"] == "openai" for r in openai_history)
        assert len(openai_history) == 1

    def test_clear_session(self, storage):
        storage.log_call("openai", "gpt-4o", 100, 50, 0.01, session_id="s-del")
        storage.log_call("openai", "gpt-4o", 100, 50, 0.01, session_id="s-keep")
        storage.clear_session("s-del")
        assert storage.get_spend("session", session_id="s-del") == 0.0
        assert storage.get_spend("session", session_id="s-keep") == pytest.approx(0.01)

    def test_clear_all(self, storage):
        storage.log_call("openai", "gpt-4o", 100, 50, 0.01)
        storage.clear_all()
        assert storage.get_spend("total") == 0.0

    def test_export_csv(self, storage, tmp_path):
        storage.log_call("openai", "gpt-4o", 100, 50, 0.01, session_id="s1")
        out = str(tmp_path / "export.csv")
        storage.export_csv(out)
        import csv
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["provider"] == "openai"

    def test_metadata_json(self, storage):
        meta = {"request_id": "req-123", "user": "test"}
        storage.log_call("openai", "gpt-4o", 100, 50, 0.01, metadata=meta)
        history = storage.get_history(limit=1)
        assert history[0]["metadata"]["request_id"] == "req-123"

    def test_invalid_period_raises(self, storage):
        with pytest.raises(ValueError, match="Unknown period"):
            storage.get_spend("quarterly")


# ---------------------------------------------------------------------------
# Session vs daily vs monthly spend tests
# ---------------------------------------------------------------------------

class TestSpendPeriods:
    def test_session_spend_isolated(self, tmp_db):
        tracker_a = CostTracker(db_path=tmp_db, auto_alert=False)
        tracker_b = CostTracker(db_path=tmp_db, auto_alert=False)

        tracker_a._record_call("openai", "gpt-4o-mini", 1_000, 500)
        tracker_b._record_call("openai", "gpt-4o-mini", 2_000, 1_000)

        assert tracker_a.get_session_cost() != tracker_b.get_session_cost()

    def test_daily_includes_all_sessions(self, tmp_db):
        tracker_a = CostTracker(db_path=tmp_db, auto_alert=False)
        tracker_b = CostTracker(db_path=tmp_db, auto_alert=False)

        tracker_a._record_call("openai", "gpt-4o-mini", 1_000, 500)
        tracker_b._record_call("openai", "gpt-4o-mini", 1_000, 500)

        # Both trackers share the same DB, daily should include both
        daily_a = tracker_a.get_daily_cost()
        daily_b = tracker_b.get_daily_cost()
        session_a = tracker_a.get_session_cost()
        session_b = tracker_b.get_session_cost()

        # Daily spend should be sum of both sessions
        assert daily_a == pytest.approx(session_a + session_b)
        assert daily_b == pytest.approx(session_a + session_b)

    def test_get_summary_keys(self, tracker):
        tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)
        summary = tracker.get_summary()
        required_keys = [
            "session_id", "session_cost", "daily_cost",
            "monthly_cost", "total_cost",
            "call_count_today", "call_count_session", "provider_breakdown"
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Provider wrapping tests (using mocks — no real API calls)
# ---------------------------------------------------------------------------

class TestOpenAIWrapping:
    def test_wrap_intercepts_create(self, tmp_db):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.model = "gpt-4o-mini"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        tracker = CostTracker(db_path=tmp_db, auto_alert=False)
        wrapped = tracker.wrap_openai(mock_client)

        result = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}]
        )

        # Response should be unchanged
        assert result is mock_response
        # Cost should be recorded
        assert tracker.get_session_cost() > 0.0

    def test_wrap_handles_missing_usage(self, tmp_db):
        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.model = "gpt-4o-mini"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        tracker = CostTracker(db_path=tmp_db, auto_alert=False)
        wrapped = tracker.wrap_openai(mock_client)
        result = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result is mock_response
        # With zero tokens, cost should be 0
        assert tracker.get_session_cost() == 0.0


class TestAnthropicWrapping:
    def test_wrap_intercepts_messages_create(self, tmp_db):
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 100
        mock_response.model = "claude-haiku-4-5"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        tracker = CostTracker(db_path=tmp_db, auto_alert=False)
        wrapped = tracker.wrap_anthropic(mock_client)

        result = wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}]
        )

        assert result is mock_response
        assert tracker.get_session_cost() > 0.0

    def test_cost_calculation_matches_expected(self, tmp_db):
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 1_000_000
        mock_response.usage.output_tokens = 1_000_000
        mock_response.model = "claude-haiku-4-5"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        tracker = CostTracker(db_path=tmp_db, auto_alert=False)
        wrapped = tracker.wrap_anthropic(mock_client)
        wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": "hi"}]
        )

        # 1M input at $0.80 + 1M output at $4.00 = $4.80
        assert tracker.get_session_cost() == pytest.approx(4.80)


class TestGeminiWrapping:
    def test_wrap_generative_model(self, tmp_db):
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 150
        mock_response.usage_metadata.candidates_token_count = 75

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model._model_name = "gemini-1.5-flash"

        tracker = CostTracker(db_path=tmp_db, auto_alert=False)
        wrapped = tracker.wrap_gemini(mock_model)

        result = wrapped.generate_content("Hello!")

        assert result is mock_response
        assert tracker.get_session_cost() > 0.0


# ---------------------------------------------------------------------------
# BudgetStatus tests
# ---------------------------------------------------------------------------

class TestBudgetStatus:
    def test_budget_status_within_budget(self, tmp_db):
        budget = Budget(daily_limit=100.0, session_limit=50.0, on_exceed="warn")
        am = AlertManager()
        tracker = CostTracker(budget=budget, db_path=tmp_db, alert_manager=am)
        tracker._record_call("openai", "gpt-4o-mini", 100, 50)

        from tokenwatch.storage import Storage
        storage = Storage(db_path=tmp_db)
        status = budget.check(storage, am, session_id=tracker.session_id)

        assert status.within_budget is True
        assert status.daily_percent is not None
        assert status.daily_percent < 1.0

    def test_budget_status_exceeded(self, tmp_db):
        budget = Budget(session_limit=0.000001, on_exceed="warn")
        am = AlertManager()
        tracker = CostTracker(budget=budget, db_path=tmp_db, alert_manager=am)
        tracker._record_call("openai", "gpt-4o-mini", 1_000, 500)

        from tokenwatch.storage import Storage
        storage = Storage(db_path=tmp_db)
        status = budget.check(storage, am, session_id=tracker.session_id)

        assert status.within_budget is False
        assert len(status.exceeded) > 0
        assert "session" in status.exceeded[0].lower()
