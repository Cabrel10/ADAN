"""
Decision Matrix section renderer for ADAN Dashboard

Displays ADAN's current signal, confidence, and market context.
"""

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ..models import Signal, MarketContext
from ..formatters import format_confidence, format_adx_strength, format_rsi_level
from ..colors import get_confidence_color, get_signal_color


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
    
    # Worker votes
    worker_votes_str = " | ".join(
        f"W{i+1}:{v:.2f}" for i, v in enumerate(signal.worker_votes.values())
    )
    table.add_row(
        "Workers",
        f"[dim cyan]{worker_votes_str}[/]"
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
    table.add_row(
        "RSI",
        f"[bold cyan]{market_context.rsi}[/] ([dim]{rsi_level}[/])"
    )
    
    # ADX with strength
    adx_strength = format_adx_strength(market_context.adx)
    table.add_row(
        "ADX",
        f"[bold cyan]{market_context.adx}[/] ([dim]{adx_strength}[/])"
    )
    
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
