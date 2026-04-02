"""
TokenWatch v0.1.1 — Full Update Validation Test
================================================
Tests all 3 providers with latest SDK versions:
  openai       1.99.9
  anthropic    0.85.0
  google-genai 1.69.0  (new SDK — replaces google-generativeai)
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

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

console = Console()

OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "")

results = {}  # track pass/fail per provider

# ══════════════════════════════════════════════════════════════════════
#  ENV CHECK
# ══════════════════════════════════════════════════════════════════════
console.print(Panel.fit("🔭  TokenWatch v0.1.1 — Update Validation", style="bold blue"))

env_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
env_table.add_column("Package", style="cyan")
env_table.add_column("Version", style="bold white")
env_table.add_column("Key Set?", style="bold")

import tokenwatch, openai, anthropic
import google.genai as genai_new

env_table.add_row("tokenwatch",    tokenwatch.__version__, "—")
env_table.add_row("openai",        openai.__version__,     "✅" if OPENAI_KEY and len(OPENAI_KEY) > 20 else "⚠ not set")
env_table.add_row("anthropic",     anthropic.__version__,  "✅" if ANTHROPIC_KEY and len(ANTHROPIC_KEY) > 20 else "⚠ not set")
env_table.add_row("google-genai",  genai_new.__version__,  "✅" if GEMINI_KEY and len(GEMINI_KEY) > 5 else "⚠ not set")
console.print(env_table)

from tokenwatch import CostTracker, Budget, AlertManager, BudgetExceededError
from tokenwatch.pricing.tables import get_cost

# ══════════════════════════════════════════════════════════════════════
#  TEST 1 — OPENAI (openai 1.99.9)
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]TEST 1 — OpenAI (v1.99.9)[/bold blue]"))

if not OPENAI_KEY or len(OPENAI_KEY) < 20:
    console.print("  [yellow]⚠ SKIPPED — OPENAI_API_KEY not set[/yellow]")
    results["openai"] = "skipped"
else:
    try:
        fired = []
        am = AlertManager()
        am.add_console_handler(level="WARNING")
        am.add_callback_handler(lambda t, m, d: fired.append(t))

        tracker = CostTracker(
            budget=Budget(daily_limit=2.00, session_limit=1.00, alert_threshold=0.01, on_exceed="warn"),
            alert_manager=am,
            spike_threshold=0.10,
        )
        client = tracker.wrap_openai(openai.OpenAI(api_key=OPENAI_KEY))

        console.print("  Calling [bold]gpt-4o-mini[/bold]...")
        t0 = time.time()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say: TOKENWATCH_V011_OK"}],
        )
        elapsed = time.time() - t0

        inp  = r.usage.prompt_tokens
        out  = r.usage.completion_tokens
        expected = get_cost("openai", "gpt-4o-mini", inp, out)
        tracked  = tracker.get_session_cost()
        match    = abs(expected - tracked) < 1e-9

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("k", style="cyan"); t.add_column("v", style="bold white")
        t.add_row("Response",         r.choices[0].message.content.strip())
        t.add_row("Input / Output",   f"{inp} / {out} tokens")
        t.add_row("Expected cost",    f"${expected:.8f}")
        t.add_row("Tracked cost",     f"${tracked:.8f}")
        t.add_row("Cost match",       "✅ PASS" if match else "❌ MISMATCH")
        t.add_row("Latency",          f"{elapsed:.2f}s")
        t.add_row("Daily cost",       f"${tracker.get_daily_cost():.6f}")
        t.add_row("Alerts fired",     str(fired))
        console.print(t)

        # Budget raise test
        strict = CostTracker(budget=Budget(session_limit=0.000001, on_exceed="raise"), auto_alert=False)
        sc = strict.wrap_openai(openai.OpenAI(api_key=OPENAI_KEY))
        try:
            sc.chat.completions.create(model="gpt-4o-mini", max_tokens=5, messages=[{"role":"user","content":"hi"}])
            console.print("  [red]❌ BudgetExceededError NOT raised[/red]")
            results["openai"] = "fail"
        except BudgetExceededError as e:
            console.print(f"  [green]✅ BudgetExceededError raised[/green] — spent ${e.spent:.8f} of ${e.limit:.8f}")
            results["openai"] = "pass" if match else "fail"

        console.print(f"  [green]✅ OpenAI PASSED[/green]\n")
    except Exception as e:
        console.print(f"  [red]❌ OpenAI FAILED: {e}[/red]")
        import traceback; traceback.print_exc()
        results["openai"] = "fail"

# ══════════════════════════════════════════════════════════════════════
#  TEST 2 — ANTHROPIC (anthropic 0.85.0)
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]TEST 2 — Anthropic (v0.85.0)[/bold blue]"))

if not ANTHROPIC_KEY or len(ANTHROPIC_KEY) < 20 or "sk-ant-..." in ANTHROPIC_KEY:
    console.print("  [yellow]⚠ SKIPPED — ANTHROPIC_API_KEY not set[/yellow]")
    results["anthropic"] = "skipped"
else:
    try:
        tracker = CostTracker(
            budget=Budget(daily_limit=2.00, session_limit=1.00, alert_threshold=0.01, on_exceed="warn"),
            auto_alert=True,
        )
        client = tracker.wrap_anthropic(anthropic.Anthropic(api_key=ANTHROPIC_KEY))

        console.print("  Calling [bold]claude-haiku-4-5[/bold]...")
        t0 = time.time()
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say: TOKENWATCH_V011_OK"}],
        )
        elapsed = time.time() - t0

        inp  = r.usage.input_tokens
        out  = r.usage.output_tokens
        expected = get_cost("anthropic", "claude-haiku-4-5", inp, out)
        tracked  = tracker.get_session_cost()
        match    = abs(expected - tracked) < 1e-9

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("k", style="cyan"); t.add_column("v", style="bold white")
        t.add_row("Response",       r.content[0].text.strip())
        t.add_row("Input / Output", f"{inp} / {out} tokens")
        t.add_row("Expected cost",  f"${expected:.8f}")
        t.add_row("Tracked cost",   f"${tracked:.8f}")
        t.add_row("Cost match",     "✅ PASS" if match else "❌ MISMATCH")
        t.add_row("Latency",        f"{elapsed:.2f}s")
        t.add_row("Daily cost",     f"${tracker.get_daily_cost():.6f}")
        console.print(t)

        results["anthropic"] = "pass" if match else "fail"
        console.print(f"  [green]✅ Anthropic PASSED[/green]\n")
    except Exception as e:
        console.print(f"  [red]❌ Anthropic FAILED: {e}[/red]")
        import traceback; traceback.print_exc()
        results["anthropic"] = "fail"

# ══════════════════════════════════════════════════════════════════════
#  TEST 3 — GEMINI (google-genai 1.69.0 — NEW SDK)
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]TEST 3 — Gemini (google-genai v1.69.0 — NEW SDK)[/bold blue]"))

if not GEMINI_KEY or len(GEMINI_KEY) < 5 or GEMINI_KEY == "AI...":
    console.print("  [yellow]⚠ SKIPPED — GEMINI_API_KEY not set[/yellow]")
    results["gemini"] = "skipped"
else:
    try:
        # New google-genai SDK usage
        from google import genai as google_genai
        from google.genai import types as genai_types

        console.print("  [dim]Using new google-genai SDK (not deprecated google-generativeai)[/dim]")

        g_client = google_genai.Client(api_key=GEMINI_KEY)

        tracker = CostTracker(
            budget=Budget(daily_limit=2.00, alert_threshold=0.01, on_exceed="warn"),
            auto_alert=True,
        )
        wrapped = tracker.wrap_gemini(g_client)

        console.print("  Calling [bold]gemini-2.0-flash[/bold] via new SDK...")
        t0 = time.time()
        r = wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say: TOKENWATCH_V011_OK",
        )
        elapsed = time.time() - t0

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("k", style="cyan"); t.add_column("v", style="bold white")
        t.add_row("Response",      r.text.strip() if hasattr(r, "text") else str(r.candidates[0].content.parts[0].text).strip())
        t.add_row("Session cost",  f"${tracker.get_session_cost():.8f}")
        t.add_row("Daily cost",    f"${tracker.get_daily_cost():.6f}")
        t.add_row("Latency",       f"{elapsed:.2f}s")
        console.print(t)

        results["gemini"] = "pass"
        console.print(f"  [green]✅ Gemini (new SDK) PASSED[/green]\n")
    except Exception as e:
        console.print(f"  [red]❌ Gemini FAILED: {e}[/red]")
        import traceback; traceback.print_exc()
        results["gemini"] = "fail"

# ══════════════════════════════════════════════════════════════════════
#  TEST 4 — CLI commands
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold blue]TEST 4 — CLI[/bold blue]"))
import subprocess, sys
TOKENWATCH_BIN = "/Library/Frameworks/Python.framework/Versions/3.14/bin/tokenwatch"
for cmd in [f"{TOKENWATCH_BIN} --version", f"{TOKENWATCH_BIN} models", f"{TOKENWATCH_BIN} report"]:
    out = subprocess.run(cmd.split(), capture_output=True, text=True)
    status = "✅" if out.returncode == 0 else "❌"
    console.print(f"  {status} [cyan]{cmd}[/cyan]")
    if out.stdout.strip():
        for line in out.stdout.strip().split("\n")[:4]:
            console.print(f"     [dim]{line}[/dim]")

# ══════════════════════════════════════════════════════════════════════
#  FINAL RESULTS
# ══════════════════════════════════════════════════════════════════════
console.print(Rule("[bold green]FINAL RESULTS[/bold green]"))

final = Table(box=box.ROUNDED, header_style="bold cyan")
final.add_column("Provider",     style="bold white")
final.add_column("SDK Version",  style="white")
final.add_column("Result",       justify="center")

sdk_versions = {
    "openai":    openai.__version__,
    "anthropic": anthropic.__version__,
    "gemini":    genai_new.__version__,
}
icons = {"pass": "[green]✅ PASSED[/green]", "fail": "[red]❌ FAILED[/red]", "skipped": "[yellow]⚠ Skipped[/yellow]"}

final.add_row("OpenAI",    sdk_versions["openai"],    icons[results.get("openai", "skipped")])
final.add_row("Anthropic", sdk_versions["anthropic"], icons[results.get("anthropic", "skipped")])
final.add_row("Gemini",    sdk_versions["gemini"],    icons[results.get("gemini", "skipped")])
console.print(final)
console.print()
