<div align="center">

# 🔭 TokenWatch

### by [Neurify](https://github.com/Neurify-PVT-LTD)

**Track, alert, and control your LLM API spending — across OpenAI, Anthropic & Gemini**

[![PyPI version](https://img.shields.io/pypi/v/neurify-tokenwatch.svg)](https://pypi.org/project/neurify-tokenwatch/)
[![Python](https://img.shields.io/pypi/pyversions/neurify-tokenwatch.svg)](https://pypi.org/project/neurify-tokenwatch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/neurify/tokenwatch.svg)](https://github.com/neurify/tokenwatch/stargazers)

**TokenWatch** is Neurify's first open-source tool — a lightweight Python package that wraps your existing LLM clients transparently, tracks every token consumed, calculates real-time costs, enforces budget limits, and fires alerts before you overspend.

Zero code changes to your existing LLM calls. Just wrap, and watch.

</div>

---

## ✨ Features

- 🔌 **Zero-change integration** — wrap your existing client, all calls tracked automatically
- 💰 **Real-time cost tracking** — per call, session, daily, monthly, all-time
- 🚨 **Smart alerts** — console, webhook, Slack, email, custom callbacks
- 🛡️ **Budget enforcement** — warn, raise error, or block when limits are hit
- 📊 **13 models supported** — OpenAI, Anthropic, Claude, Gemini
- 💾 **SQLite persistence** — local cost history, exportable to CSV
- 🖥️ **CLI dashboard** — `tokenwatch report`, `tokenwatch history`, `tokenwatch models`
- ➕ **Custom models** — add any model with your own pricing

---

## 🚀 Installation

```bash
# All providers
pip install neurify-tokenwatch[all]

# Individual providers
pip install neurify-tokenwatch[openai]
pip install neurify-tokenwatch[anthropic]
pip install neurify-tokenwatch[gemini]
```

---

## ⚡ Quick Start

### OpenAI
```python
import openai
from tokenwatch import CostTracker, Budget

tracker = CostTracker(
    budget=Budget(daily_limit=1.00, alert_threshold=0.80, on_exceed="warn")
)
client = tracker.wrap_openai(openai.OpenAI(api_key="sk-..."))

# Use exactly like normal OpenAI — zero code change
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(f"Session cost : ${tracker.get_session_cost():.6f}")
print(f"Daily cost   : ${tracker.get_daily_cost():.6f}")
```

### Anthropic
```python
import anthropic
from tokenwatch import CostTracker, Budget

tracker = CostTracker(budget=Budget(daily_limit=2.00, on_exceed="warn"))
client = tracker.wrap_anthropic(anthropic.Anthropic(api_key="sk-ant-..."))

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(f"Session cost : ${tracker.get_session_cost():.6f}")
```

### Gemini
```python
import google.generativeai as genai
from tokenwatch import CostTracker

genai.configure(api_key="AI...")
tracker = CostTracker()
model = tracker.wrap_gemini(genai.GenerativeModel("gemini-1.5-flash"))

response = model.generate_content("Hello!")
print(f"Session cost : ${tracker.get_session_cost():.6f}")
```

---

## 🛡️ Budget Enforcement

```python
from tokenwatch import CostTracker, Budget, BudgetExceededError

# Warn mode — alert and continue
tracker = CostTracker(budget=Budget(
    daily_limit=1.00,
    monthly_limit=20.00,
    session_limit=0.50,
    alert_threshold=0.80,   # alert at 80%
    on_exceed="warn"
))

# Raise mode — throws BudgetExceededError
tracker = CostTracker(budget=Budget(session_limit=0.01, on_exceed="raise"))
try:
    client.chat.completions.create(...)
except BudgetExceededError as e:
    print(f"Over budget! Spent ${e.spent:.4f} of ${e.limit:.4f} ({e.period})")

# Block mode — stops the call entirely
tracker = CostTracker(budget=Budget(monthly_limit=50.00, on_exceed="block"))
```

---

## 🔔 Alert System

```python
from tokenwatch import AlertManager, CostTracker

alert_mgr = AlertManager()

# Console (default — rich colored panels)
alert_mgr.add_console_handler(level="WARNING")

# Custom callback
def my_alert(alert_type, message, data):
    print(f"[{alert_type}] {message}")
alert_mgr.add_callback_handler(my_alert)

# Webhook (Slack, Discord, etc.)
alert_mgr.add_webhook_handler("https://hooks.slack.com/...")

# Email
alert_mgr.add_email_handler(
    smtp_config={"host": "smtp.gmail.com", "port": 587, "user": "x", "password": "y"},
    to_email="team@yourcompany.com"
)

tracker = CostTracker(alert_manager=alert_mgr, spike_threshold=0.05)
```

---

## 📊 Cost Queries

```python
tracker.get_session_cost()    # current session
tracker.get_daily_cost()      # today (UTC)
tracker.get_monthly_cost()    # this month
tracker.get_total_cost()      # all time
tracker.get_summary()         # full breakdown dict
tracker.export_report("costs.csv")  # export to CSV
```

---

## 🎨 Usage Patterns

### Decorator
```python
@tracker.watch
def run_pipeline():
    client.chat.completions.create(...)
    client.chat.completions.create(...)

run_pipeline()
# prints: [tokenwatch] run_pipeline used $0.000081 (session total: $0.000081)
```

### Context Manager
```python
with CostTracker() as tracker:
    client = tracker.wrap_openai(openai.OpenAI(api_key="..."))
    client.chat.completions.create(...)
# Prints rich summary table on exit
```

---

## 💰 Supported Models & Pricing

| Provider | Model | Input /1M | Output /1M |
|---|---|---|---|
| OpenAI | gpt-4o | $2.50 | $10.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| OpenAI | gpt-4-turbo | $10.00 | $30.00 |
| OpenAI | gpt-3.5-turbo | $0.50 | $1.50 |
| Anthropic | claude-opus-4-6 | $15.00 | $75.00 |
| Anthropic | claude-sonnet-4-6 | $3.00 | $15.00 |
| Anthropic | claude-haiku-4-5 | $0.80 | $4.00 |
| Gemini | gemini-1.5-pro | $1.25 | $5.00 |
| Gemini | gemini-1.5-flash | $0.075 | $0.30 |
| Gemini | gemini-2.0-flash | $0.10 | $0.40 |

### Add custom models
```python
from tokenwatch.pricing.tables import add_custom_model
add_custom_model("openai", "gpt-5", input_price_per_1m=10.00, output_price_per_1m=30.00)
```

---

## 🖥️ CLI

```bash
tokenwatch report                    # spend today / month / all time
tokenwatch history --limit 20        # last 20 API calls
tokenwatch history --provider openai # filter by provider
tokenwatch models                    # all supported models + pricing
tokenwatch export --output costs.csv # export to CSV
tokenwatch clear                     # clear database
```

---

## 🤝 Contributing

TokenWatch is Neurify's first open-source project and we welcome contributions!

1. Fork the repo
2. Create your branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/neurify">Neurify</a> — Our first open-source release 🎉
</div>
