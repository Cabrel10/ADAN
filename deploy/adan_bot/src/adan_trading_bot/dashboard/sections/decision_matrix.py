"""
Decision Matrix section renderer for ADAN Dashboard

Displays ADAN's current signal, confidence, and market context.
"""

from rich.table import Table
from rich.panel import Panel
from rich import box

from ..models import Signal, MarketContext
from ..formatters import format_confidence, format_adx_strength, format_rsi_level
from ..colors import get_confidence_color, get_signal_color


def _format_worker_votes(worker_votes: dict) -> str:
    """
    Format worker votes with color coding based on confidence level.

    Args:
        worker_votes: Dictionary of worker votes {worker_id: confidence}

    Returns:
        Formatted string with colored worker votes
    """
    if not worker_votes:
        return "[dim red]No votes[/]"

    formatted_votes = []
    for worker_id, vote in sorted(worker_votes.items()):
        # Color code based on vote strength
        if vote >= 0.8:
            color = "green"
        elif vote >= 0.6:
            color = "yellow"
        elif vote >= 0.4:
            color = "cyan"
        else:
            color = "red"

        vote_str = f"[bold {color}]{worker_id}:{vote:.2f}[/]"
        formatted_votes.append(vote_str)

    return " | ".join(formatted_votes)


def render_decision_matrix(signal: Signal, market_context: MarketContext) -> Panel:
    """
    Render the decision matrix section.

    Args:
        signal: Current ADAN signal
        market_context: Current market context

    Returns:
        Rich Panel containing decision matrix
    """
    # Create table
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Metric", style="bold cyan", width=15)
    table.add_column("Value", style="bold white")

    # Signal direction with color
    signal_color = get_signal_color(signal.direction)
    table.add_row(
        "Signal",
        f"[bold {signal_color}]{signal.direction}[/]"
    )

    # Confidence with color
    confidence_color = get_confidence_color(signal.confidence)
    table.add_row(
        "Confidence",
        f"[bold {confidence_color}]{format_confidence(signal.confidence)}[/]"
    )

    # Horizon
    table.add_row(
        "Horizon",
        f"[bold yellow]{signal.horizon}[/]"
    )

    # Worker votes - show individual votes with colors
    if signal.worker_votes:
        worker_votes_display = _format_worker_votes(signal.worker_votes)
        table.add_row(
            "Workers",
            worker_votes_display
        )
    else:
        table.add_row(
            "Workers",
            "[dim red]No votes[/]"
        )

    # Decision driver
    table.add_row(
        "Driver",
        f"[bold magenta]{signal.decision_driver}[/]"
    )

    # Market context
    table.add_row("", "")  # Spacer

    # Volatility
    table.add_row(
        "Volatility",
        f"[bold yellow]{market_context.volatility_atr:.2f}%[/]"
    )

    # RSI with level
    rsi_level = format_rsi_level(market_context.rsi)
    rsi_str = f"[bold cyan]{market_context.rsi}[/] ([dim]{rsi_level}[/])"
    table.add_row("RSI", rsi_str)

    # ADX with strength
    adx_strength = format_adx_strength(market_context.adx)
    adx_str = f"[bold cyan]{market_context.adx}[/] ([dim]{adx_strength}[/])"
    table.add_row("ADX", adx_str)

    # Trend strength
    table.add_row(
        "Trend",
        f"[bold green]{market_context.trend_strength}[/]"
    )

    # Market regime
    table.add_row(
        "Regime",
        f"[bold magenta]{market_context.market_regime}[/]"
    )

    # Volume change
    volume_color = "green" if market_context.volume_change > 0 else "red"
    volume_sign = "+" if market_context.volume_change > 0 else ""
    table.add_row(
        "Volume",
        f"[bold {volume_color}]{volume_sign}{market_context.volume_change:.1f}%[/]"
    )

    # Create panel
    return Panel(
        table,
        title="[bold cyan]📊 DECISION MATRIX[/]",
        border_style="cyan",
        box=box.ROUNDED,
    )
