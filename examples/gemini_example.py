"""
Google Gemini example — demonstrates tokenwatch with the Google Generative AI SDK.

Run:
    export GOOGLE_API_KEY=your-key
    python examples/gemini_example.py
"""

import os

import google.generativeai as genai

from tokenwatch import AlertManager, Budget, CostTracker

# -----------------------------------------------------------------------
# 1. Configure Gemini SDK
# -----------------------------------------------------------------------
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", "your-api-key"))

# -----------------------------------------------------------------------
# 2. Budget
# -----------------------------------------------------------------------
budget = Budget(
    daily_limit=1.00,
    monthly_limit=5.00,
    alert_threshold=0.80,
    on_exceed="warn",
)

# -----------------------------------------------------------------------
# 3. Tracker with console alerts
# -----------------------------------------------------------------------
alert_manager = AlertManager()
alert_manager.add_console_handler(level="INFO")

tracker = CostTracker(
    budget=budget,
    alert_manager=alert_manager,
    spike_threshold=0.05,
)

# -----------------------------------------------------------------------
# 4. Wrap a GenerativeModel
# -----------------------------------------------------------------------
model = genai.GenerativeModel("gemini-1.5-flash")
model = tracker.wrap_gemini(model)

print("Making API call to gemini-1.5-flash...")

response = model.generate_content("What is the speed of light? Answer in one sentence.")

print(f"Response: {response.text}")

# -----------------------------------------------------------------------
# 5. Cost breakdown
# -----------------------------------------------------------------------
print(f"\nSession cost  : ${tracker.get_session_cost():.6f}")
print(f"Daily cost    : ${tracker.get_daily_cost():.6f}")
print(f"Monthly cost  : ${tracker.get_monthly_cost():.6f}")

# -----------------------------------------------------------------------
# 6. Another model (Gemini 1.5 Pro)
# -----------------------------------------------------------------------
print("\n--- Gemini 1.5 Pro example ---")
pro_model = genai.GenerativeModel("gemini-1.5-pro")
pro_model = tracker.wrap_gemini(pro_model)

response2 = pro_model.generate_content(
    "Name three renewable energy sources. Be brief."
)
print(f"Response: {response2.text}")

# -----------------------------------------------------------------------
# 7. Full summary
# -----------------------------------------------------------------------
summary = tracker.get_summary()
print("\nFull summary:")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"  {key:25s}: ${value:.6f}")
    else:
        print(f"  {key:25s}: {value}")

# -----------------------------------------------------------------------
# 8. Context manager
# -----------------------------------------------------------------------
print("\n--- Context manager example ---")
with CostTracker() as ctx_tracker:
    flash = genai.GenerativeModel("gemini-2.0-flash")
    flash = ctx_tracker.wrap_gemini(flash)
    resp = flash.generate_content("Say 'hello' in Japanese.")
    print(f"Response: {resp.text}")
# Summary printed on exit
