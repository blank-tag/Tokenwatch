"""
LIVE OpenAI Cost Validation Test for tokenwatch
================================================
Tests real token tracking across multiple calls,
simulates 1M token cost math, and validates alerts.
"""

import os
import sys
import time

# ── load .env ──────────────────────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

import openai
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import track

from tokenwatch import CostTracker, Budget, AlertManager, BudgetExceededError
from tokenwatch.pricing.tables import get_cost, PRICING

console = Console()
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_KEY or "sk-ant" in OPENAI_KEY:
    console.print("[red]ERROR:[/red] OPENAI_API_KEY not set in .env")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════
# TEST 1 — Pricing math validation (what 1M tokens costs per model)
# ══════════════════════════════════════════════════════════════════════════
console.print(Panel.fit("🧮  TEST 1 — OpenAI 1M Token Cost Reference", style="bold blue"))

table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
table.add_column("Model", style="white")
table.add_column("1M Input Tokens", justify="right", style="green")
table.add_column("1M Output Tokens", justify="right", style="yellow")
table.add_column("1M Input + 1M Output", justify="right", style="bold magenta")

for model, (inp, out) in PRICING["openai"].items():
    cost_in  = get_cost("openai", model, 1_000_000, 0)
    cost_out = get_cost("openai", model, 0, 1_000_000)
    cost_both = get_cost("openai", model, 1_000_000, 1_000_000)
    table.add_row(model, f"${cost_in:.4f}", f"${cost_out:.4f}", f"${cost_both:.4f}")

console.print(table)

# ══════════════════════════════════════════════════════════════════════════
# TEST 2 — Single real API call + live token tracking
# ══════════════════════════════════════════════════════════════════════════
console.print(Panel.fit("🔴  TEST 2 — Live API Call: gpt-4o-mini", style="bold blue"))

fired_alerts = []

def alert_callback(alert_type, message, data):
    fired_alerts.append({"type": alert_type, "message": message, "data": data})

alert_mgr = AlertManager()
alert_mgr.add_console_handler(level="INFO")
alert_mgr.add_callback_handler(alert_callback)

tracker = CostTracker(
    budget=Budget(
        daily_limit=1.00,
        session_limit=0.50,
        alert_threshold=0.01,   # fire at 1% — so any real spend triggers it
        on_exceed="warn",
    ),
    alert_manager=alert_mgr,
    spike_threshold=0.05,
)

client = tracker.wrap_openai(openai.OpenAI(api_key=OPENAI_KEY))

console.print("  Calling [bold]gpt-4o-mini[/bold] with a real prompt...")
t0 = time.time()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=200,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "List 5 interesting facts about token pricing in LLMs. "
                "Be concise but informative."
            ),
        },
    ],
)

elapsed = time.time() - t0
actual_input  = response.usage.prompt_tokens
actual_output = response.usage.completion_tokens
expected_cost = get_cost("openai", "gpt-4o-mini", actual_input, actual_output)
tracked_cost  = tracker.get_session_cost()

console.print(f"\n  [bold]Response:[/bold]\n  {response.choices[0].message.content}\n")

result_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
result_table.add_column("Metric", style="cyan")
result_table.add_column("Value", style="bold white")
result_table.add_row("Model",               "gpt-4o-mini")
result_table.add_row("Latency",             f"{elapsed:.2f}s")
result_table.add_row("Input tokens",        f"{actual_input:,}")
result_table.add_row("Output tokens",       f"{actual_output:,}")
result_table.add_row("Expected cost",       f"${expected_cost:.8f}")
result_table.add_row("Tracked by tokenwatch", f"${tracked_cost:.8f}")
result_table.add_row(
    "Cost match ✅" if abs(expected_cost - tracked_cost) < 1e-9 else "Cost MISMATCH ❌",
    "PASS" if abs(expected_cost - tracked_cost) < 1e-9 else "FAIL"
)
result_table.add_row("Alerts fired",        str(len(fired_alerts)))
console.print(result_table)

# ══════════════════════════════════════════════════════════════════════════
# TEST 3 — Multiple calls, accumulation & daily tracking
# ══════════════════════════════════════════════════════════════════════════
console.print(Panel.fit("📈  TEST 3 — 5 Rapid Calls: Accumulation Tracking", style="bold blue"))

messages = [
    "What is a token in the context of LLMs?",
    "How does GPT-4o-mini differ from GPT-4o?",
    "What is prompt engineering?",
    "Explain temperature in LLM sampling in one sentence.",
    "What does 'context window' mean?",
]

call_costs = []
for i, msg in enumerate(messages, 1):
    cost_before = tracker.get_session_cost()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=60,
        messages=[{"role": "user", "content": msg}],
    )
    cost_after = tracker.get_session_cost()
    delta = cost_after - cost_before
    call_costs.append({
        "call": i,
        "input": resp.usage.prompt_tokens,
        "output": resp.usage.completion_tokens,
        "cost": delta,
    })
    console.print(f"  Call {i}/5 → input={resp.usage.prompt_tokens} out={resp.usage.completion_tokens} cost=${delta:.6f}")
    time.sleep(0.3)

acc_table = Table(box=box.ROUNDED, header_style="bold cyan")
acc_table.add_column("Call #")
acc_table.add_column("Input Tokens", justify="right")
acc_table.add_column("Output Tokens", justify="right")
acc_table.add_column("Call Cost", justify="right", style="green")

for c in call_costs:
    acc_table.add_row(str(c["call"]), str(c["input"]), str(c["output"]), f"${c['cost']:.6f}")

console.print(acc_table)
console.print(f"\n  [bold cyan]Session total (6 calls):[/bold cyan] ${tracker.get_session_cost():.6f}")
console.print(f"  [bold cyan]Daily total:[/bold cyan]             ${tracker.get_daily_cost():.6f}")
console.print(f"  [bold cyan]Monthly total:[/bold cyan]           ${tracker.get_monthly_cost():.6f}")

# ══════════════════════════════════════════════════════════════════════════
# TEST 4 — Budget exceeded → raise mode
# ══════════════════════════════════════════════════════════════════════════
console.print(Panel.fit("🚨  TEST 4 — Budget Enforcement: on_exceed=raise", style="bold blue"))

strict_tracker = CostTracker(
    budget=Budget(session_limit=0.000001, on_exceed="raise"),
    auto_alert=False,
)
strict_client = strict_tracker.wrap_openai(openai.OpenAI(api_key=OPENAI_KEY))

try:
    strict_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        messages=[{"role": "user", "content": "Hi"}],
    )
    console.print("  [red]❌ FAIL — BudgetExceededError was NOT raised[/red]")
except BudgetExceededError as e:
    console.print(f"  [green]✅ BudgetExceededError raised correctly[/green]")
    console.print(f"     Period : [yellow]{e.period}[/yellow]")
    console.print(f"     Limit  : [yellow]${e.limit:.8f}[/yellow]")
    console.print(f"     Spent  : [yellow]${e.spent:.8f}[/yellow]")

# ══════════════════════════════════════════════════════════════════════════
# TEST 5 — 1M token cost extrapolation from real call
# ══════════════════════════════════════════════════════════════════════════
console.print(Panel.fit("💰  TEST 5 — Extrapolated 1M Token Cost from Real Usage", style="bold blue"))

real_input  = call_costs[-1]["input"]
real_output = call_costs[-1]["output"]
real_cost   = call_costs[-1]["cost"]

if real_input > 0 and real_output > 0:
    cost_per_input_token  = get_cost("openai", "gpt-4o-mini", 1, 0)
    cost_per_output_token = get_cost("openai", "gpt-4o-mini", 0, 1)

    extrap_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    extrap_table.add_column("Item", style="cyan")
    extrap_table.add_column("Value", style="bold white")
    extrap_table.add_row("Cost per 1 input token",  f"${cost_per_input_token:.10f}")
    extrap_table.add_row("Cost per 1 output token", f"${cost_per_output_token:.10f}")
    extrap_table.add_row("Cost for 1M input tokens",  f"${get_cost('openai','gpt-4o-mini',1_000_000,0):.4f}")
    extrap_table.add_row("Cost for 1M output tokens", f"${get_cost('openai','gpt-4o-mini',0,1_000_000):.4f}")
    extrap_table.add_row("Cost for 1M in + 1M out",   f"${get_cost('openai','gpt-4o-mini',1_000_000,1_000_000):.4f}")
    extrap_table.add_row("─" * 30, "─" * 15)
    extrap_table.add_row(f"Real call ({real_input} in, {real_output} out)", f"${real_cost:.8f}")
    console.print(extrap_table)

# ══════════════════════════════════════════════════════════════════════════
# TEST 6 — Context manager full summary
# ══════════════════════════════════════════════════════════════════════════
console.print(Panel.fit("📊  TEST 6 — Full Session Summary (Context Manager)", style="bold blue"))
with CostTracker() as final:
    pass  # __exit__ prints the rich table automatically

# ══════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════
console.print(Panel.fit("✅  ALL TESTS COMPLETE", style="bold green"))
console.print()
console.print("  [bold]Run CLI commands:[/bold]")
console.print("  [cyan]tokenwatch report[/cyan]")
console.print("  [cyan]tokenwatch history --limit 10[/cyan]")
console.print("  [cyan]tokenwatch export --output costs.csv[/cyan]")
console.print("  [cyan]tokenwatch models[/cyan]")
console.print()
