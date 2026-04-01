"""
Live manual test for tokenwatch.
Tests Anthropic, OpenAI, and Gemini — skip any you don't have a key for.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    export GEMINI_API_KEY=AI...

    python3 test_live.py
"""

import os
import sys

from tokenwatch import AlertManager, Budget, BudgetExceededError, CostTracker
from tokenwatch.pricing.tables import get_cost, list_models

# ─────────────────────────────────────────────────────────────
# STEP 1: Smoke test — pricing tables (no API key needed)
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  STEP 1: Pricing table smoke test (no API key needed)")
print("═" * 60)

models = list_models()  # returns dict: provider -> model -> (in_price, out_price)
total = sum(len(v) for v in models.values())
print(f"  Total models loaded: {total}")
for provider, model_dict in models.items():
    for model_name, (inp, outp) in model_dict.items():
        cost_1k = get_cost(provider, model_name, 1000, 1000)
        print(f"  {provider:12s}  {model_name:40s}  ~${cost_1k:.6f} / 1K tokens each way")

# ─────────────────────────────────────────────────────────────
# STEP 2: Budget enforcement (no API key needed)
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  STEP 2: Budget enforcement test (no API key needed)")
print("═" * 60)

from tokenwatch.storage import Storage
from tokenwatch.alerts import AlertManager as AM

storage = Storage()  # in-memory effectively — uses ~/.tokenwatch/costs.db
storage.log_call(
    provider="openai", model="gpt-4o-mini",
    input_tokens=50000, output_tokens=20000,
    cost=get_cost("openai", "gpt-4o-mini", 50000, 20000),
    session_id="test-budget-session",
    metadata={"note": "budget test call"},
)

budget = Budget(daily_limit=0.001, alert_threshold=0.5, on_exceed="warn")
am = AM()

fired_alerts = []
am.add_callback_handler(lambda t, msg, d: fired_alerts.append(t))

status = budget.check(storage, am, session_id="test-budget-session")
print(f"  within_budget   : {status.within_budget}")
print(f"  daily_percent   : {status.daily_percent:.1f}%")
print(f"  warnings        : {status.warnings}")
print(f"  exceeded        : {status.exceeded}")
print(f"  alerts fired    : {fired_alerts}")

# ─────────────────────────────────────────────────────────────
# STEP 3: Anthropic live test
# ─────────────────────────────────────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
print("\n" + "═" * 60)
print("  STEP 3: Anthropic live test")
print("═" * 60)

if not ANTHROPIC_KEY:
    print("  SKIPPED — set ANTHROPIC_API_KEY to run this test")
else:
    try:
        import anthropic

        tracker = CostTracker(
            budget=Budget(daily_limit=2.00, alert_threshold=0.80, on_exceed="warn"),
            auto_alert=True,
        )
        client = tracker.wrap_anthropic(anthropic.Anthropic(api_key=ANTHROPIC_KEY))

        print("  Calling claude-haiku-4-5 ...")
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=50,
            messages=[{"role": "user", "content": "Reply with exactly: LLMGUARD_TEST_OK"}],
        )
        text = response.content[0].text
        print(f"  Response        : {text.strip()}")
        print(f"  Input tokens    : {response.usage.input_tokens}")
        print(f"  Output tokens   : {response.usage.output_tokens}")
        print(f"  Session cost    : ${tracker.get_session_cost():.6f}")
        print(f"  Daily cost      : ${tracker.get_daily_cost():.6f}")
        print(f"  Monthly cost    : ${tracker.get_monthly_cost():.6f}")
        summary = tracker.get_summary()
        print(f"  Calls today     : {summary['call_count_today']}")

        # Budget block test
        print("\n  Testing budget block (on_exceed=raise) ...")
        strict_tracker = CostTracker(
            budget=Budget(session_limit=0.000001, on_exceed="raise"),
            auto_alert=False,
        )
        strict_client = strict_tracker.wrap_anthropic(anthropic.Anthropic(api_key=ANTHROPIC_KEY))
        try:
            strict_client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            print("  ERROR: BudgetExceededError was NOT raised!")
        except BudgetExceededError as e:
            print(f"  BudgetExceededError caught correctly: {e}")

        print("  ANTHROPIC TEST PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# STEP 4: OpenAI live test
# ─────────────────────────────────────────────────────────────
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
print("\n" + "═" * 60)
print("  STEP 4: OpenAI live test")
print("═" * 60)

if not OPENAI_KEY:
    print("  SKIPPED — set OPENAI_API_KEY to run this test")
else:
    try:
        import openai

        tracker = CostTracker(
            budget=Budget(daily_limit=2.00, alert_threshold=0.80, on_exceed="warn"),
            auto_alert=True,
        )
        client = tracker.wrap_openai(openai.OpenAI(api_key=OPENAI_KEY))

        print("  Calling gpt-4o-mini ...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=50,
            messages=[{"role": "user", "content": "Reply with exactly: LLMGUARD_TEST_OK"}],
        )
        text = response.choices[0].message.content
        print(f"  Response        : {text.strip()}")
        print(f"  Input tokens    : {response.usage.prompt_tokens}")
        print(f"  Output tokens   : {response.usage.completion_tokens}")
        print(f"  Session cost    : ${tracker.get_session_cost():.6f}")
        print(f"  Daily cost      : ${tracker.get_daily_cost():.6f}")
        print(f"  Monthly cost    : ${tracker.get_monthly_cost():.6f}")
        print("  OPENAI TEST PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# STEP 5: Gemini live test
# ─────────────────────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
print("\n" + "═" * 60)
print("  STEP 5: Gemini live test")
print("═" * 60)

if not GEMINI_KEY:
    print("  SKIPPED — set GEMINI_API_KEY to run this test")
else:
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_KEY)
        tracker = CostTracker(auto_alert=True)
        model = tracker.wrap_gemini(genai.GenerativeModel("gemini-1.5-flash"))

        print("  Calling gemini-1.5-flash ...")
        response = model.generate_content("Reply with exactly: LLMGUARD_TEST_OK")
        print(f"  Response        : {response.text.strip()}")
        print(f"  Session cost    : ${tracker.get_session_cost():.6f}")
        print(f"  Daily cost      : ${tracker.get_daily_cost():.6f}")
        print("  GEMINI TEST PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# STEP 6: Context manager + full summary
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  STEP 6: Context manager summary (all session data)")
print("═" * 60)

with CostTracker() as final_tracker:
    # Just show accumulated costs from this run
    pass  # __exit__ prints the rich summary table

# ─────────────────────────────────────────────────────────────
# STEP 7: CLI check
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  STEP 7: CLI commands to try in your terminal")
print("═" * 60)
print("  tokenwatch report")
print("  tokenwatch history --limit 10")
print("  tokenwatch models")
print("  tokenwatch export --output costs.csv")
print()
