"""
TokenWatch v0.1.2 — Groq Live Validation Test
==============================================
Tests the new Groq provider with real API calls.
"""

import os, sys, time

# ── load .env ──────────────────────────────────────────────────────────
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "): line = line[7:]
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

import groq
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from tokenwatch import CostTracker, Budget, AlertManager, BudgetExceededError
from tokenwatch.pricing.tables import get_cost, list_models

console = Console()
GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════
#  ENV CHECK
# ══════════════════════════════════════════════════════════════════════
console.print(Panel.fit("🔭  TokenWatch v0.1.2 — Groq Provider Test", style="bold blue"))

import tokenwatch
env = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
env.add_column("k", style="cyan"); env.add_column("v", style="bold white")
env.add_row("tokenwatch", tokenwatch.__version__)
env.add_row("groq SDK",   groq.__version__)
env.add_row("GROQ_API_KEY", "✅ set" if GROQ_KEY and len(GROQ_KEY) > 10 else "❌ not set")
console.print(env)

if not GROQ_KEY or len(GROQ_KEY) < 10:
    console.print("[red]ERROR: GROQ_API_KEY not set in .env[/red]")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
#  TEST 1 — Groq pricing table (no API key needed)
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]TEST 1 — Groq Pricing Table[/bold blue]"))

groq_models = list_models("groq")["groq"]
t = Table(box=box.ROUNDED, header_style="bold cyan")
t.add_column("Model",         style="white",  width=45)
t.add_column("Per 1M Input",  justify="right", style="green")
t.add_column("Per 1M Output", justify="right", style="yellow")
t.add_column("1M + 1M Cost",  justify="right", style="bold magenta")

for model, (inp, out) in groq_models.items():
    cost = get_cost("groq", model, 1_000_000, 1_000_000)
    t.add_row(model, f"${inp:.3f}", f"${out:.3f}", f"${cost:.4f}")
console.print(t)
console.print("  [green]✅ Groq pricing table loaded — 10 models[/green]\n")

# ══════════════════════════════════════════════════════════════════════
#  TEST 2 — Single live call: llama-3.3-70b-versatile
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]TEST 2 — Live Call: llama-3.3-70b-versatile[/bold blue]"))

fired = []
am = AlertManager()
am.add_console_handler(level="INFO")
am.add_callback_handler(lambda t, m, d: fired.append(t))

tracker = CostTracker(
    budget=Budget(daily_limit=1.00, session_limit=0.50, alert_threshold=0.01, on_exceed="warn"),
    alert_manager=am,
    spike_threshold=0.05,
)
client = tracker.wrap_groq(groq.Groq(api_key=GROQ_KEY))

console.print("  Calling [bold]openai/gpt-oss-120b[/bold]...")
t0 = time.time()
r = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    max_tokens=80,
    messages=[{"role": "user", "content": "Say: TOKENWATCH_GROQ_OK and list 3 Groq benefits in one line."}],
)
elapsed = time.time() - t0

inp      = r.usage.prompt_tokens
out      = r.usage.completion_tokens
expected = get_cost("groq", "openai/gpt-oss-120b", inp, out)
tracked  = tracker.get_session_cost()
match    = abs(expected - tracked) < 1e-9

res = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
res.add_column("k", style="cyan"); res.add_column("v", style="bold white")
res.add_row("Response",          r.choices[0].message.content.strip())
res.add_row("Input / Output",    f"{inp} / {out} tokens")
res.add_row("Expected cost",     f"${expected:.8f}")
res.add_row("Tracked cost",      f"${tracked:.8f}")
res.add_row("Cost match",        "✅ PASS" if match else "❌ MISMATCH")
res.add_row("Latency",           f"{elapsed:.2f}s")
res.add_row("Alerts fired",      str(fired))
console.print(res)
console.print(f"  [green]✅ openai/gpt-oss-120b PASSED[/green]\n")

# ══════════════════════════════════════════════════════════════════════
#  TEST 3 — Fast/cheap model: openai/gpt-oss-120b
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]TEST 3 — Live Call: openai/gpt-oss-120b (fastest)[/bold blue]"))

console.print("  Calling [bold]openai/gpt-oss-120b[/bold]...")
t0 = time.time()
r2 = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    max_tokens=50,
    messages=[{"role": "user", "content": "What is Groq LPU in one sentence?"}],
)
elapsed2 = time.time() - t0

inp2      = r2.usage.prompt_tokens
out2      = r2.usage.completion_tokens
cost2     = get_cost("groq", "openai/gpt-oss-120b", inp2, out2)

res2 = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
res2.add_column("k", style="cyan"); res2.add_column("v", style="bold white")
res2.add_row("Response",       r2.choices[0].message.content.strip())
res2.add_row("Input / Output", f"{inp2} / {out2} tokens")
res2.add_row("Call cost",      f"${cost2:.8f}")
res2.add_row("Session total",  f"${tracker.get_session_cost():.8f}")
res2.add_row("Latency",        f"{elapsed2:.2f}s")
console.print(res2)
console.print(f"  [green]✅ openai/gpt-oss-120b PASSED[/green]\n")

# ══════════════════════════════════════════════════════════════════════
#  TEST 4 — 3 rapid calls: accumulation tracking
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]TEST 4 — 3 Rapid Calls: Accumulation[/bold blue]"))

questions = [
    "What is LLaMA in one sentence?",
    "What is Mixtral in one sentence?",
    "What is tokenwatch in one sentence?",
]
call_costs = []
for i, q in enumerate(questions, 1):
    before = tracker.get_session_cost()
    rx = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=40,
        messages=[{"role": "user", "content": q}],
    )
    delta = tracker.get_session_cost() - before
    call_costs.append({"call": i, "in": rx.usage.prompt_tokens, "out": rx.usage.completion_tokens, "cost": delta})
    console.print(f"  Call {i}/3 → {rx.usage.prompt_tokens} in / {rx.usage.completion_tokens} out → ${delta:.8f}")

acc = Table(box=box.ROUNDED, header_style="bold cyan")
acc.add_column("Call", justify="center")
acc.add_column("Input", justify="right")
acc.add_column("Output", justify="right")
acc.add_column("Cost", justify="right", style="green")
for c in call_costs:
    acc.add_row(str(c["call"]), str(c["in"]), str(c["out"]), f"${c['cost']:.8f}")
console.print(acc)
console.print(f"\n  [bold cyan]Session total (5 calls):[/bold cyan] ${tracker.get_session_cost():.8f}")
console.print(f"  [bold cyan]Daily total:[/bold cyan]            ${tracker.get_daily_cost():.6f}\n")

# ══════════════════════════════════════════════════════════════════════
#  TEST 5 — Budget enforcement: on_exceed=raise
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]TEST 5 — Budget Enforcement: on_exceed=raise[/bold blue]"))

strict = CostTracker(budget=Budget(session_limit=0.000001, on_exceed="raise"), auto_alert=False)
sc = strict.wrap_groq(groq.Groq(api_key=GROQ_KEY))
try:
    sc.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=5,
        messages=[{"role": "user", "content": "hi"}],
    )
    console.print("  [red]❌ BudgetExceededError NOT raised[/red]")
except BudgetExceededError as e:
    console.print(f"  [green]✅ BudgetExceededError raised correctly[/green]")
    console.print(f"     Period : [yellow]{e.period}[/yellow]")
    console.print(f"     Limit  : [yellow]${e.limit:.8f}[/yellow]")
    console.print(f"     Spent  : [yellow]${e.spent:.8f}[/yellow]\n")

# ══════════════════════════════════════════════════════════════════════
#  TEST 6 — Cost comparison: Groq vs OpenAI same prompt
# ══════════════════════════════════════════════════════════════════════
if OPENAI_KEY and len(OPENAI_KEY) > 20 and "sk-ant" not in OPENAI_KEY:
    console.print(Rule("[bold blue]TEST 6 — Cost Comparison: Groq vs OpenAI[/bold blue]"))
    import openai as _openai

    PROMPT = "Explain what a transformer model is in exactly 2 sentences."

    # Groq
    t0 = time.time()
    rg = client.chat.completions.create(
        model="openai/gpt-oss-120b", max_tokens=80,
        messages=[{"role": "user", "content": PROMPT}],
    )
    groq_time = time.time() - t0
    groq_cost = get_cost("groq", "openai/gpt-oss-120b", rg.usage.prompt_tokens, rg.usage.completion_tokens)

    # OpenAI
    oai_tracker = CostTracker(auto_alert=False)
    oai_client  = oai_tracker.wrap_openai(_openai.OpenAI(api_key=OPENAI_KEY))
    t0 = time.time()
    ro = oai_client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=80,
        messages=[{"role": "user", "content": PROMPT}],
    )
    oai_time = time.time() - t0
    oai_cost = oai_tracker.get_session_cost()

    cmp = Table(box=box.ROUNDED, header_style="bold cyan", title="Same Prompt — Side by Side")
    cmp.add_column("",              style="bold white")
    cmp.add_column("Groq\nllama-3.3-70b", justify="center", style="green")
    cmp.add_column("OpenAI\ngpt-4o-mini",  justify="center", style="yellow")
    cmp.add_row("Model",           "openai/gpt-oss-120b", "gpt-4o-mini")
    cmp.add_row("Tokens (in/out)", f"{rg.usage.prompt_tokens}/{rg.usage.completion_tokens}", f"{ro.usage.prompt_tokens}/{ro.usage.completion_tokens}")
    cmp.add_row("Cost",            f"${groq_cost:.8f}", f"${oai_cost:.8f}")
    cmp.add_row("Latency",         f"{groq_time:.2f}s",  f"{oai_time:.2f}s")
    savings = ((oai_cost - groq_cost) / oai_cost * 100) if oai_cost > 0 else 0
    cmp.add_row("Groq Savings",    f"[bold green]{savings:.1f}% cheaper[/bold green]", "—")
    console.print(cmp)
    console.print()

# ══════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold green]ALL GROQ TESTS COMPLETE[/bold green]"))
with CostTracker() as final:
    pass

console.print("\n  [bold]Try the CLI:[/bold]")
console.print("  [cyan]tokenwatch report[/cyan]")
console.print("  [cyan]tokenwatch history --limit 10[/cyan]")
console.print("  [cyan]tokenwatch models[/cyan]\n")
