"""
tokenwatch — LLM Cost Optimizer & Budget Guardian by Neurify.

Track, alert on, and control your LLM API spending across
OpenAI, Anthropic, and Google Gemini.

Quick start::

    from tokenwatch import CostTracker, Budget

    tracker = CostTracker(
        budget=Budget(daily_limit=1.00, on_exceed="warn"),
    )

    import openai
    client = tracker.wrap_openai(openai.OpenAI(api_key="..."))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(f"Session cost: ${tracker.get_session_cost():.4f}")
"""

from .alerts import AlertManager
from .budget import Budget, BudgetExceededError, BudgetStatus
from .tracker import CostTracker

__version__ = "0.1.1"
__all__ = [
    "CostTracker",
    "Budget",
    "BudgetStatus",
    "AlertManager",
    "BudgetExceededError",
    "__version__",
]
