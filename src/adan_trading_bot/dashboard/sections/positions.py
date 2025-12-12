"""
Active Positions section renderer for ADAN Dashboard

Displays detailed information about each open position.
"""

from typing import List
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ..models import Position
from ..formatters import (
    format_btc,
    format_price,
    format_percentage,
    format_time,
)
from ..colors import get_pnl_color


def render_positions(positions: List[Position]) -> Panel:
    """
    Render the active positions section.
    
    Args:
        positions: List of open positions
    
    Returns:
        Rich Panel containing positions table
    """
    # Create table
    table = Table(show_header=True, box=box.ROUNDED, padding=(0, 1))
    table.add_column("#", style="dim cyan", width=3)
    table.add_column("Pair", style="bold cyan", width=10)
    table.add_column("Side", style="bold white", width=6)
    table.add_column("Size (BTC)", justify="right", style="bold white", width=12)
    table.add_column("Entry", justify="right", style="bold yellow", width=12)
    table.add_column("Current", justify="right", style="bold white", width=12)
    table.add_column("SL", justify="right", style="dim white", width=12)
    table.add_column("TP", justify="right", style="dim white", width=12)
    table.add_column("P&L", justify="right", width=14)
    table.add_column("Time", justify="right", style="dim cyan", width=10)
    
    if not positions:
        # Empty state
        table.add_row(
            "",
            "[dim yellow]No active positions[/]",
            "", "", "", "", "", "", "", ""
        )
    else:
        # Add each position
        for i, pos in enumerate(positions, 1):
            # Side with color
            side_color = "green" if pos.side == "LONG" else "red"
            side_icon = "🚀" if pos.side == "LONG" else "🔻"
            
            # P&L with color
            pnl_color = get_pnl_color(pos.unrealized_pnl_pct)
            pnl_icon = "📈" if pos.unrealized_pnl_usd >= 0 else "📉"
            
            table.add_row(
                f"[dim]{i}[/]",
                f"[bold cyan]{pos.pair}[/]",
                f"[bold {side_color}]{side_icon} {pos.side}[/]",
                f"[bold cyan]{format_btc(pos.size_btc)}[/]",
                f"[bold yellow]{format_price(pos.entry_price)}[/]",
                f"[bold white]{format_price(pos.current_price)}[/]",
                f"[dim]{format_price(pos.sl_price)}[/]",
                f"[dim]{format_price(pos.tp_price)}[/]",
                f"[bold {pnl_color}]{pnl_icon} {format_price(pos.unrealized_pnl_usd)} ({format_percentage(pos.unrealized_pnl_pct)})[/]",
                f"[dim cyan]{format_time(pos.duration)}[/]",
            )
            
            # Add context row for each position
            context_text = Text()
            context_text.append("  Context: ", style="dim white")
            context_text.append(f"RSI {pos.entry_rsi} | ", style="dim cyan")
            context_text.append(f"Vol {pos.entry_volatility:.1f}% | ", style="dim yellow")
            context_text.append(f"Regime {pos.entry_market_regime} | ", style="dim magenta")
            context_text.append(f"Signal {pos.entry_signal_strength:.2f}", style="dim green")
            
            table.add_row(
                "",
                context_text,
                "", "", "", "", "", "", "", ""
            )
    
    # Create panel
    return Panel(
        table,
        title="[bold cyan]📍 ACTIVE POSITIONS[/]",
        border_style="cyan",
        box=box.ROUNDED,
    )
