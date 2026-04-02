"""
Groq example — demonstrates tokenwatch with the Groq SDK.

Run:
    export GROQ_API_KEY=gsk_...
    python examples/groq_example.py
"""

import os
import groq
from tokenwatch import AlertManager, Budget, BudgetExceededError, CostTracker

# -----------------------------------------------------------------------
# 1. Budget: $1.00/day, warn at 80%
# -----------------------------------------------------------------------
budget = Budget(
    daily_limit=1.00,
    monthly_limit=10.00,
    session_limit=0.50,
    alert_threshold=0.80,
    on_exceed="warn",
)

# -----------------------------------------------------------------------
# 2. Alert manager
# -----------------------------------------------------------------------
alert_manager = AlertManager()
alert_manager.add_console_handler(level="INFO")

# -----------------------------------------------------------------------
# 3. Create tracker and wrap the Groq client
# -----------------------------------------------------------------------
tracker = CostTracker(
    budget=budget,
    alert_manager=alert_manager,
    spike_threshold=0.05,
)

client = tracker.wrap_groq(
    groq.Groq(api_key=os.environ.get("GROQ_API_KEY", "your-api-key"))
)

# -----------------------------------------------------------------------
# 4. Make API calls — zero code change from normal Groq usage
# -----------------------------------------------------------------------
print("Making API call to llama-3.3-70b-versatile...")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=100,
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What are three benefits of open-source software?"},
    ],
)

print(f"Response: {response.choices[0].message.content}")

# -----------------------------------------------------------------------
# 5. Cost breakdown
# -----------------------------------------------------------------------
print(f"\nSession cost  : ${tracker.get_session_cost():.6f}")
print(f"Daily cost    : ${tracker.get_daily_cost():.6f}")
print(f"Monthly cost  : ${tracker.get_monthly_cost():.6f}")

# -----------------------------------------------------------------------
# 6. Fast & cheap model — llama-3.1-8b-instant
# -----------------------------------------------------------------------
print("\n--- Calling llama-3.1-8b-instant (fastest/cheapest) ---")
r2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say hello in 5 languages."}],
)
print(f"Response: {r2.choices[0].message.content}")
print(f"Session cost after 2 calls: ${tracker.get_session_cost():.6f}")

# -----------------------------------------------------------------------
# 7. BudgetExceededError demo
# -----------------------------------------------------------------------
print("\n--- Budget enforcement demo (raise mode) ---")
strict_tracker = CostTracker(
    budget=Budget(session_limit=0.000001, on_exceed="raise"),
    auto_alert=False,
)
strict_client = strict_tracker.wrap_groq(
    groq.Groq(api_key=os.environ.get("GROQ_API_KEY", "your-api-key"))
)

try:
    strict_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=5,
        messages=[{"role": "user", "content": "Hi"}],
    )
except BudgetExceededError as e:
    print(f"Caught BudgetExceededError: {e}")
    print(f"  Period : {e.period}")
    print(f"  Limit  : ${e.limit:.8f}")
    print(f"  Spent  : ${e.spent:.8f}")

# -----------------------------------------------------------------------
# 8. Context manager — auto summary on exit
# -----------------------------------------------------------------------
print("\n--- Context manager ---")
with CostTracker() as ctx:
    ctx_client = ctx.wrap_groq(
        groq.Groq(api_key=os.environ.get("GROQ_API_KEY", "your-api-key"))
    )
    resp = ctx_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=20,
        messages=[{"role": "user", "content": "Say 'tokenwatch rocks' and nothing else."}],
    )
    print(f"Response: {resp.choices[0].message.content}")
# Summary table printed automatically on exit
