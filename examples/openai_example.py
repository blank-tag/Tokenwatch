"""
OpenAI example — demonstrates tokenwatch with the OpenAI SDK.

Run:
    export OPENAI_API_KEY=your-key
    python examples/openai_example.py
"""

import os

import openai

from tokenwatch import AlertManager, Budget, CostTracker

# -----------------------------------------------------------------------
# 1. Set up a budget: $1.00/day, warn at 80%
# -----------------------------------------------------------------------
budget = Budget(
    daily_limit=1.00,
    monthly_limit=10.00,
    alert_threshold=0.80,
    on_exceed="warn",   # change to "raise" to raise BudgetExceededError
)

# -----------------------------------------------------------------------
# 2. Set up alert channels
# -----------------------------------------------------------------------
alert_manager = AlertManager()
alert_manager.add_console_handler(level="INFO")  # colored Rich output

# Optionally add a webhook (e.g. Slack):
# alert_manager.add_webhook_handler(
#     "https://hooks.slack.com/services/...",
#     headers={"Content-Type": "application/json"},
# )

# -----------------------------------------------------------------------
# 3. Create tracker and wrap the OpenAI client
# -----------------------------------------------------------------------
tracker = CostTracker(
    budget=budget,
    alert_manager=alert_manager,
    spike_threshold=0.05,   # alert if any single call costs > $0.05
)

client = tracker.wrap_openai(
    openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
)

# -----------------------------------------------------------------------
# 4. Make API calls — tracking happens automatically
# -----------------------------------------------------------------------
print("Making API call to gpt-4o-mini...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Answer in one word."},
    ],
    max_tokens=20,
)

print(f"Response: {response.choices[0].message.content}")

# -----------------------------------------------------------------------
# 5. Check costs
# -----------------------------------------------------------------------
print(f"\nSession cost  : ${tracker.get_session_cost():.6f}")
print(f"Daily cost    : ${tracker.get_daily_cost():.6f}")
print(f"Monthly cost  : ${tracker.get_monthly_cost():.6f}")

# -----------------------------------------------------------------------
# 6. Get full summary
# -----------------------------------------------------------------------
summary = tracker.get_summary()
print(f"\nFull summary:")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"  {key:25s}: ${value:.6f}")
    else:
        print(f"  {key:25s}: {value}")

# -----------------------------------------------------------------------
# 7. Use as context manager for automatic summary on exit
# -----------------------------------------------------------------------
print("\n--- Context manager example ---")
with CostTracker(budget=Budget(daily_limit=5.00)) as ctx_tracker:
    ctx_client = ctx_tracker.wrap_openai(
        openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
    )
    response2 = ctx_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'hello' in Spanish."}],
        max_tokens=10,
    )
    print(f"Response: {response2.choices[0].message.content}")
# Summary table printed automatically on __exit__

# -----------------------------------------------------------------------
# 8. Decorator usage
# -----------------------------------------------------------------------
@tracker.watch
def run_pipeline(question: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        max_tokens=30,
    )
    return resp.choices[0].message.content


result = run_pipeline("What is 2+2?")
print(f"\nPipeline result: {result}")
print(f"Final session cost: ${tracker.get_session_cost():.6f}")
