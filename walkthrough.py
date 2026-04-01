"""
╔══════════════════════════════════════════════════════════════════════╗
║         tokenwatch — Full Import & Usage Walkthrough                 ║
║         Tests: OpenAI · Anthropic · Gemini                          ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
-----------
  1. Install the package (one time):
        pip3 install -e ".[all]"          ← from the project folder

  2. Set your API keys (any or all):
        export OPENAI_API_KEY=sk-...
        export ANTHROPIC_API_KEY=sk-ant-...
        export GEMINI_API_KEY=AI...

  3. Run:
        python3 walkthrough.py
"""

# ── load .env file if present ──────────────────────────────────────────
import os, sys, time

_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "): line = line[7:]
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich import box
from rich.table import Table

console = Console()

# ══════════════════════════════════════════════════════════════════════
#  STEP 0 — What you import from tokenwatch
# ══════════════════════════════════════════════════════════════════════
console.print(Panel.fit(
    "[bold white]STEP 0 — The Import[/bold white]\n\n"
    "[cyan]from tokenwatch import CostTracker, Budget, AlertManager, BudgetExceededError[/cyan]\n\n"
    "[dim]CostTracker    → main class, wraps your LLM client\n"
    "Budget         → set daily/monthly/session limits\n"
    "AlertManager   → console / webhook / email / callback alerts\n"
    "BudgetExceededError → raised when on_exceed='raise'[/dim]",
    title="📦 tokenwatch imports", border_style="bold blue"
))

from tokenwatch import CostTracker, Budget, AlertManager, BudgetExceededError
from tokenwatch.pricing.tables import get_cost, list_models, add_custom_model
console.print("  [green]✅ All imports successful[/green]\n")

# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — Pricing tables (no API key needed)
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]STEP 1 — Pricing Tables (no API key needed)[/bold blue]"))
console.print("[dim]  from tokenwatch.pricing.tables import get_cost, list_models, add_custom_model[/dim]\n")

table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=True)
table.add_column("Provider",  style="bold white",  width=12)
table.add_column("Model",     style="white",        width=35)
table.add_column("Per 1M In", justify="right", style="green")
table.add_column("Per 1M Out",justify="right", style="yellow")
table.add_column("1K+1K cost",justify="right", style="magenta")

for provider, models in list_models().items():
    for model, (inp, out) in models.items():
        small = get_cost(provider, model, 1000, 1000)
        table.add_row(provider, model, f"${inp:.3f}", f"${out:.3f}", f"${small:.6f}")

console.print(table)

# Show: add your own custom model
add_custom_model("openai", "gpt-5", input_price_per_1m=10.00, output_price_per_1m=30.00)
custom_cost = get_cost("openai", "gpt-5", 1_000_000, 1_000_000)
console.print(f"\n  [bold]Custom model (gpt-5):[/bold] 1M+1M = [magenta]${custom_cost:.2f}[/magenta]")
console.print("  [dim]add_custom_model('openai', 'gpt-5', input_price_per_1m=10.00, output_price_per_1m=30.00)[/dim]\n")

# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — Budget & AlertManager configuration
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]STEP 2 — Budget + AlertManager Setup[/bold blue]"))

console.print("""
[cyan]# Option A — Simple budget (console alerts auto-enabled)[/cyan]
tracker = CostTracker(
    budget=Budget(daily_limit=1.00, alert_threshold=0.80, on_exceed="warn")
)

[cyan]# Option B — Full control[/cyan]
alert_mgr = AlertManager()
alert_mgr.add_console_handler(level="INFO")            # rich colored panels
alert_mgr.add_callback_handler(my_fn)                  # your own function
alert_mgr.add_webhook_handler("https://hooks.slack..") # POST to Slack/etc
alert_mgr.add_email_handler(smtp_config, "me@x.com")   # email alert

tracker = CostTracker(
    budget=Budget(
        daily_limit   = 1.00,    # $1 per day
        monthly_limit = 20.00,   # $20 per month
        session_limit = 0.50,    # $0.50 per session
        alert_threshold = 0.80,  # warn at 80%
        on_exceed = "warn",      # "warn" | "raise" | "block"
    ),
    alert_manager = alert_mgr,
    spike_threshold = 0.05,      # alert if single call > $0.05
)
""")

# Demonstrate callback
fired = []
am = AlertManager()
am.add_console_handler(level="INFO")
am.add_callback_handler(lambda t, msg, d: fired.append(t))
console.print("  [green]✅ AlertManager configured[/green]\n")

# ══════════════════════════════════════════════════════════════════════
#  STEP 3A — OpenAI
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]STEP 3A — OpenAI Integration[/bold blue]"))

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

console.print("""
[cyan]# 1. Install:[/cyan]  pip install openai
[cyan]# 2. Wrap:[/cyan]
import openai
from tokenwatch import CostTracker, Budget

tracker = CostTracker(budget=Budget(daily_limit=1.00, on_exceed="warn"))
client  = tracker.wrap_openai(openai.OpenAI(api_key="sk-..."))

[cyan]# 3. Use exactly like normal OpenAI — zero code change:[/cyan]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(f"Cost: ${tracker.get_session_cost():.6f}")
""")

if not OPENAI_KEY or len(OPENAI_KEY) < 20 or "sk-ant" in OPENAI_KEY:
    console.print("  [yellow]⚠ SKIPPED — OPENAI_API_KEY not set[/yellow]\n")
else:
    try:
        import openai as _openai
        tracker_oai = CostTracker(
            budget=Budget(daily_limit=2.00, session_limit=1.00, alert_threshold=0.01, on_exceed="warn"),
            alert_manager=am,
            spike_threshold=0.10,
        )
        client_oai = tracker_oai.wrap_openai(_openai.OpenAI(api_key=OPENAI_KEY))

        console.print("  [bold]Making real call → gpt-4o-mini[/bold]")
        t0 = time.time()
        r = client_oai.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=60,
            messages=[{"role": "user", "content": "In one sentence, what is tokenwatch?"}],
        )
        elapsed = time.time() - t0

        t = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
        t.add_column("k", style="cyan"); t.add_column("v", style="bold white")
        t.add_row("Response",      r.choices[0].message.content.strip())
        t.add_row("Input tokens",  str(r.usage.prompt_tokens))
        t.add_row("Output tokens", str(r.usage.completion_tokens))
        t.add_row("Session cost",  f"${tracker_oai.get_session_cost():.8f}")
        t.add_row("Daily cost",    f"${tracker_oai.get_daily_cost():.6f}")
        t.add_row("Monthly cost",  f"${tracker_oai.get_monthly_cost():.6f}")
        t.add_row("Latency",       f"{elapsed:.2f}s")
        console.print(t)
        console.print("  [green]✅ OpenAI PASSED[/green]\n")
    except Exception as e:
        console.print(f"  [red]❌ OpenAI FAILED: {e}[/red]\n")

# ══════════════════════════════════════════════════════════════════════
#  STEP 3B — Anthropic
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]STEP 3B — Anthropic Integration[/bold blue]"))

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

console.print("""
[cyan]# 1. Install:[/cyan]  pip install anthropic
[cyan]# 2. Wrap:[/cyan]
import anthropic
from tokenwatch import CostTracker, Budget

tracker = CostTracker(budget=Budget(daily_limit=1.00, on_exceed="warn"))
client  = tracker.wrap_anthropic(anthropic.Anthropic(api_key="sk-ant-..."))

[cyan]# 3. Use exactly like normal Anthropic — zero code change:[/cyan]
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(f"Cost: ${tracker.get_session_cost():.6f}")
""")

if not ANTHROPIC_KEY or len(ANTHROPIC_KEY) < 20 or ANTHROPIC_KEY == "sk-ant-...":
    console.print("  [yellow]⚠ SKIPPED — ANTHROPIC_API_KEY not set[/yellow]\n")
else:
    try:
        import anthropic as _anthropic
        tracker_ant = CostTracker(
            budget=Budget(daily_limit=2.00, session_limit=1.00, alert_threshold=0.01, on_exceed="warn"),
            alert_manager=am,
        )
        client_ant = tracker_ant.wrap_anthropic(_anthropic.Anthropic(api_key=ANTHROPIC_KEY))

        console.print("  [bold]Making real call → claude-haiku-4-5[/bold]")
        t0 = time.time()
        r = client_ant.messages.create(
            model="claude-haiku-4-5",
            max_tokens=60,
            messages=[{"role": "user", "content": "In one sentence, what is tokenwatch?"}],
        )
        elapsed = time.time() - t0

        t = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
        t.add_column("k", style="cyan"); t.add_column("v", style="bold white")
        t.add_row("Response",      r.content[0].text.strip())
        t.add_row("Input tokens",  str(r.usage.input_tokens))
        t.add_row("Output tokens", str(r.usage.output_tokens))
        t.add_row("Session cost",  f"${tracker_ant.get_session_cost():.8f}")
        t.add_row("Daily cost",    f"${tracker_ant.get_daily_cost():.6f}")
        t.add_row("Monthly cost",  f"${tracker_ant.get_monthly_cost():.6f}")
        t.add_row("Latency",       f"{elapsed:.2f}s")
        console.print(t)
        console.print("  [green]✅ Anthropic PASSED[/green]\n")
    except Exception as e:
        console.print(f"  [red]❌ Anthropic FAILED: {e}[/red]\n")
        import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  STEP 3C — Gemini
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]STEP 3C — Gemini Integration[/bold blue]"))

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

console.print("""
[cyan]# 1. Install:[/cyan]  pip install google-generativeai
[cyan]# 2. Wrap:[/cyan]
import google.generativeai as genai
from tokenwatch import CostTracker

genai.configure(api_key="AI...")
tracker = CostTracker()
model   = tracker.wrap_gemini(genai.GenerativeModel("gemini-1.5-flash"))

[cyan]# 3. Use exactly like normal Gemini — zero code change:[/cyan]
response = model.generate_content("Hello!")
print(f"Cost: ${tracker.get_session_cost():.6f}")
""")

if not GEMINI_KEY or len(GEMINI_KEY) < 5 or GEMINI_KEY == "AI...":
    console.print("  [yellow]⚠ SKIPPED — GEMINI_API_KEY not set[/yellow]\n")
else:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)

        tracker_gem = CostTracker(
            budget=Budget(daily_limit=2.00, alert_threshold=0.01, on_exceed="warn"),
            alert_manager=am,
        )
        model_gem = tracker_gem.wrap_gemini(genai.GenerativeModel("gemini-1.5-flash"))

        console.print("  [bold]Making real call → gemini-1.5-flash[/bold]")
        t0 = time.time()
        r = model_gem.generate_content("In one sentence, what is tokenwatch?")
        elapsed = time.time() - t0

        t = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
        t.add_column("k", style="cyan"); t.add_column("v", style="bold white")
        t.add_row("Response",     r.text.strip())
        t.add_row("Session cost", f"${tracker_gem.get_session_cost():.8f}")
        t.add_row("Daily cost",   f"${tracker_gem.get_daily_cost():.6f}")
        t.add_row("Latency",      f"{elapsed:.2f}s")
        console.print(t)
        console.print("  [green]✅ Gemini PASSED[/green]\n")
    except Exception as e:
        console.print(f"  [red]❌ Gemini FAILED: {e}[/red]\n")

# ══════════════════════════════════════════════════════════════════════
#  STEP 4 — Decorator pattern
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]STEP 4 — @tracker.watch Decorator[/bold blue]"))

console.print("""
[cyan]# Wrap any function — tracks cost delta per call automatically[/cyan]
tracker = CostTracker()
client  = tracker.wrap_openai(openai.OpenAI(api_key="..."))

@tracker.watch
def run_pipeline():
    client.chat.completions.create(model="gpt-4o-mini", ...)
    client.chat.completions.create(model="gpt-4o-mini", ...)

run_pipeline()
[dim]# prints: [tokenwatch] run_pipeline used $0.000081 (session total: $0.000081)[/dim]
""")
console.print("  [green]✅ Decorator pattern documented[/green]\n")

# ══════════════════════════════════════════════════════════════════════
#  STEP 5 — Context Manager
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]STEP 5 — Context Manager (auto summary on exit)[/bold blue]"))

console.print("""
[cyan]# Prints a rich summary table automatically when block exits[/cyan]
with CostTracker() as tracker:
    client = tracker.wrap_openai(openai.OpenAI(api_key="..."))
    client.chat.completions.create(...)
[dim]# ↑ prints Session Summary table on __exit__[/dim]
""")

with CostTracker() as demo_tracker:
    pass  # just show the table
console.print()

# ══════════════════════════════════════════════════════════════════════
#  STEP 6 — Budget enforcement modes
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]STEP 6 — Budget Enforcement Modes[/bold blue]"))

console.print("""
[cyan]# MODE 1: warn  → logs alert, continues[/cyan]
Budget(daily_limit=1.00, on_exceed="warn")

[cyan]# MODE 2: raise → raises BudgetExceededError[/cyan]
Budget(session_limit=0.01, on_exceed="raise")

try:
    client.chat.completions.create(...)
except BudgetExceededError as e:
    print(e.period)   # "session"
    print(e.limit)    # 0.01
    print(e.spent)    # actual spend

[cyan]# MODE 3: block → same as raise (stops the call)[/cyan]
Budget(monthly_limit=50.00, on_exceed="block")
""")
console.print("  [green]✅ All 3 enforcement modes documented[/green]\n")

# ══════════════════════════════════════════════════════════════════════
#  STEP 7 — Cost queries & export
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]STEP 7 — Cost Queries & Export[/bold blue]"))

console.print("""
[cyan]# Query costs anytime[/cyan]
tracker.get_session_cost()   # → float: current session
tracker.get_daily_cost()     # → float: today (UTC)
tracker.get_monthly_cost()   # → float: this month
tracker.get_total_cost()     # → float: all time

tracker.get_summary()        # → dict with full breakdown
tracker.export_report("costs.csv")   # → CSV file
""")

console.print("""[cyan]# CLI commands (run in terminal)[/cyan]
[bold]tokenwatch report[/bold]              → spend today / month / all time
[bold]tokenwatch history --limit 20[/bold]  → last 20 API calls
[bold]tokenwatch models[/bold]              → all supported models + pricing
[bold]tokenwatch export --output x.csv[/bold] → export full history
[bold]tokenwatch clear[/bold]               → wipe database (asks confirmation)
""")

# ══════════════════════════════════════════════════════════════════════
#  FINAL — Summary of all keys tested
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold green]WALKTHROUGH COMPLETE[/bold green]"))

summary = Table(box=box.ROUNDED, header_style="bold cyan")
summary.add_column("Provider",  style="bold white")
summary.add_column("Key Set?",  justify="center")
summary.add_column("Live Test", justify="center")

def key_status(key): return "[green]✅[/green]" if key and len(key) > 10 else "[yellow]⚠ Not set[/yellow]"
def live_status(key, bad=""): return "[green]✅ PASSED[/green]" if key and len(key) > 10 and key != bad else "[dim]Skipped[/dim]"

summary.add_row("OpenAI",    key_status(OPENAI_KEY),    live_status(OPENAI_KEY))
summary.add_row("Anthropic", key_status(ANTHROPIC_KEY), live_status(ANTHROPIC_KEY, "sk-ant-..."))
summary.add_row("Gemini",    key_status(GEMINI_KEY),    live_status(GEMINI_KEY, "AI..."))
console.print(summary)

console.print("\n  [bold]To enable skipped providers:[/bold]")
if not (OPENAI_KEY and len(OPENAI_KEY) > 10):
    console.print("  [dim]export OPENAI_API_KEY=sk-...[/dim]")
if not (ANTHROPIC_KEY and len(ANTHROPIC_KEY) > 10 and ANTHROPIC_KEY != "sk-ant-..."):
    console.print("  [dim]export ANTHROPIC_API_KEY=sk-ant-...[/dim]")
if not (GEMINI_KEY and len(GEMINI_KEY) > 5 and GEMINI_KEY != "AI..."):
    console.print("  [dim]export GEMINI_API_KEY=AI...[/dim]")

console.print("\n  Then re-run: [bold cyan]python3 walkthrough.py[/bold cyan]\n")
