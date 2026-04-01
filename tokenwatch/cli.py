"""
Command-line interface for tokenwatch.

Commands:
  tokenwatch report   — show spend summary (today, month, total)
  tokenwatch history  — show call history
  tokenwatch export   — export history to CSV
  tokenwatch clear    — clear all data
  tokenwatch models   — list all supported models and pricing
"""

from __future__ import annotations

import sys
from typing import Optional

import click

from .pricing.tables import list_models
from .storage import Storage


def _get_storage(db_path: Optional[str] = None) -> Storage:
    return Storage(db_path=db_path)


def _require_rich() -> None:
    try:
        import rich  # noqa: F401
    except ImportError:
        click.echo("rich is required for table output. Install with: pip install rich", err=True)
        sys.exit(1)


@click.group()
@click.version_option(package_name="neurify-tokenwatch")
def main() -> None:
    """TokenWatch — LLM Cost Optimizer & Budget Guardian by Neurify."""


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@main.command()
@click.option("--db", default=None, help="Path to the SQLite database file.")
def report(db: Optional[str]) -> None:
    """Show spend summary: today, this month, and all-time."""
    _require_rich()
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    storage = _get_storage(db)
    console = Console()

    daily = storage.get_spend("daily")
    monthly = storage.get_spend("monthly")
    total = storage.get_spend("total")
    calls_today = storage.get_call_count("daily")
    calls_month = storage.get_call_count("monthly")
    breakdown = storage.get_spend_by_provider("daily")

    table = Table(title="TokenWatch — Spend Report", show_header=True, header_style="bold cyan")
    table.add_column("Period", style="bold")
    table.add_column("Cost (USD)", justify="right", style="green")
    table.add_column("API Calls", justify="right")

    table.add_row("Today", f"${daily:.6f}", str(calls_today))
    table.add_row("This Month", f"${monthly:.6f}", str(calls_month))
    table.add_row("All Time", f"${total:.6f}", "—")

    console.print(table)

    if breakdown:
        breakdown_table = Table(
            title="Today's Spend by Provider", show_header=True, header_style="bold magenta"
        )
        breakdown_table.add_column("Provider", style="bold")
        breakdown_table.add_column("Cost (USD)", justify="right", style="yellow")

        for provider, cost in sorted(breakdown.items(), key=lambda x: -x[1]):
            breakdown_table.add_row(provider, f"${cost:.6f}")

        console.print(breakdown_table)


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

@main.command()
@click.option("--limit", default=20, show_default=True, help="Number of records to show.")
@click.option("--provider", default=None, help="Filter by provider (openai/anthropic/gemini).")
@click.option("--db", default=None, help="Path to the SQLite database file.")
def history(limit: int, provider: Optional[str], db: Optional[str]) -> None:
    """Show recent LLM call history."""
    _require_rich()
    from rich.console import Console
    from rich.table import Table

    storage = _get_storage(db)
    console = Console()

    rows = storage.get_history(limit=limit, provider=provider)

    if not rows:
        console.print("[yellow]No call history found.[/yellow]")
        return

    table = Table(
        title=f"LLM Call History (last {limit})", show_header=True, header_style="bold cyan"
    )
    table.add_column("ID", style="dim", width=6)
    table.add_column("Timestamp", width=20)
    table.add_column("Provider", style="bold")
    table.add_column("Model")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Cost (USD)", justify="right", style="green")
    table.add_column("Session", style="dim", width=10)

    for row in rows:
        ts = row.get("timestamp", "")[:19].replace("T", " ")
        session = str(row.get("session_id") or "")[:8]
        table.add_row(
            str(row.get("id", "")),
            ts,
            row.get("provider", ""),
            row.get("model", ""),
            str(row.get("input_tokens", 0)),
            str(row.get("output_tokens", 0)),
            f"${float(row.get('cost_usd', 0)):.6f}",
            session,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@main.command()
@click.option("--output", default="tokenwatch_report.csv", show_default=True,
              help="Output CSV file path.")
@click.option("--db", default=None, help="Path to the SQLite database file.")
def export(output: str, db: Optional[str]) -> None:
    """Export full call history to a CSV file."""
    storage = _get_storage(db)
    storage.export_csv(output)
    click.echo(f"Exported call history to: {output}")


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

@main.command()
@click.option("--db", default=None, help="Path to the SQLite database file.")
@click.confirmation_option(prompt="This will permanently delete ALL call history. Are you sure?")
def clear(db: Optional[str]) -> None:
    """Clear all stored call history."""
    storage = _get_storage(db)
    storage.clear_all()
    click.echo("All call history has been cleared.")


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

@main.command()
@click.option("--provider", default=None,
              help="Filter by provider (openai/anthropic/gemini).")
def models(provider: Optional[str]) -> None:
    """List all supported models and their pricing."""
    _require_rich()
    from rich.console import Console
    from rich.table import Table

    console = Console()

    try:
        pricing = list_models(provider)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    for prov, model_dict in pricing.items():
        table = Table(
            title=f"{prov.capitalize()} Models",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Model", style="bold")
        table.add_column("Input ($/1M tokens)", justify="right", style="green")
        table.add_column("Output ($/1M tokens)", justify="right", style="yellow")
        table.add_column("Example: 1K+1K tokens", justify="right", style="dim")

        for model_name, (inp, out) in sorted(model_dict.items()):
            # Example cost: 1000 input + 1000 output tokens
            example = (inp / 1000) + (out / 1000)
            table.add_row(
                model_name,
                f"${inp:.4f}",
                f"${out:.4f}",
                f"${example:.6f}",
            )

        console.print(table)


if __name__ == "__main__":
    main()
