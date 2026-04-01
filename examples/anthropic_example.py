"""
Anthropic example — demonstrates tokenwatch with the Anthropic SDK.

Run:
    export ANTHROPIC_API_KEY=your-key
    python examples/anthropic_example.py
"""

import os

import anthropic

from tokenwatch import AlertManager, Budget, BudgetExceededError, CostTracker

# -----------------------------------------------------------------------
# 1. Budget: $2.00/day, $20.00/month, warn at 75%
# -----------------------------------------------------------------------
budget = Budget(
    daily_limit=2.00,
    monthly_limit=20.00,
    session_limit=0.50,
    alert_threshold=0.75,
    on_exceed="warn",
)

# -----------------------------------------------------------------------
# 2. Alert manager with console output
# -----------------------------------------------------------------------
alert_manager = AlertManager()
alert_manager.add_console_handler(level="INFO")

# Custom callback example:
def my_alert_callback(alert_type: str, message: str, data: dict) -> None:
    if alert_type == "budget_warning":
        print(f"[CUSTOM] Budget warning: {message}")

alert_manager.add_callback_handler(my_alert_callback)

# -----------------------------------------------------------------------
# 3. Create tracker and wrap the Anthropic client
# -----------------------------------------------------------------------
tracker = CostTracker(
    budget=budget,
    alert_manager=alert_manager,
    spike_threshold=0.10,
)

client = tracker.wrap_anthropic(
    anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key"))
)

# -----------------------------------------------------------------------
# 4. Make API calls
# -----------------------------------------------------------------------
print("Making API call to claude-sonnet-4-6...")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    system="You are a concise assistant.",
    messages=[
        {"role": "user", "content": "What are the three primary colors? List them briefly."}
    ],
)

print(f"Response: {response.content[0].text}")

# -----------------------------------------------------------------------
# 5. Cost breakdown
# -----------------------------------------------------------------------
print(f"\nSession cost  : ${tracker.get_session_cost():.6f}")
print(f"Daily cost    : ${tracker.get_daily_cost():.6f}")
print(f"Monthly cost  : ${tracker.get_monthly_cost():.6f}")

# -----------------------------------------------------------------------
# 6. Demonstrate BudgetExceededError (raise mode)
# -----------------------------------------------------------------------
print("\n--- Budget enforcement demo (raise mode) ---")
strict_budget = Budget(
    session_limit=0.000001,  # $0.000001 — will be exceeded immediately
    on_exceed="raise",
)
strict_tracker = CostTracker(budget=strict_budget, auto_alert=False)
strict_client = strict_tracker.wrap_anthropic(
    anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key"))
)

try:
    strict_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}],
    )
except BudgetExceededError as e:
    print(f"Caught BudgetExceededError: {e}")
    print(f"  Period : {e.period}")
    print(f"  Limit  : ${e.limit:.6f}")
    print(f"  Spent  : ${e.spent:.6f}")

# -----------------------------------------------------------------------
# 7. Context manager
# -----------------------------------------------------------------------
print("\n--- Context manager example ---")
with CostTracker() as ctx_tracker:
    ctx_client = ctx_tracker.wrap_anthropic(
        anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key"))
    )
    resp = ctx_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=20,
        messages=[{"role": "user", "content": "Say 'bonjour' and nothing else."}],
    )
    print(f"Response: {resp.content[0].text}")
# Summary table printed on exit
