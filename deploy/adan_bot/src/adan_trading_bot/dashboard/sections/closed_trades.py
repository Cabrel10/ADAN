"""
Closed Trades section renderer for ADAN Dashboard

Displays recent closed trades with outcomes and analysis.
"""

from typing import List
from rich.table import Table
from rich.panel import Panel
from rich import box

from ..models import ClosedTrade
from ..formatters import (
    format_btc,
    format_price,
    format_percentage,
    format_time,
    format_outcome_symbol,
    format_confidence,
)
from ..colors import get_pnl_color


def render_closed_trades(trades: List[ClosedTrade]) -> Panel:
    """
    Render the closed trades section.
    
    Args:
        trades: List of closed trades (should be sorted by close_time descending)
    
    Returns:
        Rich Panel containing closed trades table
    """
    # Create table
    table = Table(show_header=True, box=box.ROUNDED, padding=(0, 1))
    table.add_column("#", style="dim cyan", width=3)
    table.add_column("Out", justify="center", width=5)
    table.add_column("Duration", justify="right", style="dim cyan", width=10)
    table.add_column("Size (BTC)", justify="right", style="bold white", width=12)
    table.add_column("Entry", justify="right", style="bold yellow", width=12)
    table.add_column("Exit", justify="right", style="bold yellow", width=12)
    table.add_column("P&L", justify="right", width=14)
    table.add_column("Reason", justify="left", style="dim white", width=10)
    table.add_column("Conf", justify="right", style="dim cyan", width=6)
    
    if not trades:
        # Empty state
        table.add_row(
            "",
            "[dim yellow]No closed trades[/]",
            "", "", "", "", "", "", ""
        )
    else:
        # Add each trade (last 5)
        for i, trade in enumerate(trades[:5], 1):
            # Outcome symbol
            outcome_symbol = format_outcome_symbol(trade.is_win, trade.is_breakeven)
            
            # P&L with color
            pnl_color = get_pnl_color(trade.realized_pnl_pct)
            
            table.add_row(
                f"[dim]{i}[/]",
                outcome_symbol,
                f"[dim cyan]{format_time(trade.duration)}[/]",
                f"[bold cyan]{format_btc(trade.size_btc)}[/]",
                f"[bold yellow]{format_price(trade.entry_price)}[/]",
                f"[bold yellow]{format_price(trade.exit_price)}[/]",
                f"[bold {pnl_color}]{format_price(trade.realized_pnl_usd)} ({format_percentage(trade.realized_pnl_pct)})[/]",
                f"[dim white]{trade.close_reason}[/]",
                f"[dim cyan]{format_confidence(trade.entry_confidence)}[/]",
            )
    
    # Create panel
    return Panel(
        table,
        title="[bold green]📈 LAST 5 CLOSED TRADES[/]",
        border_style="green",
        box=box.ROUNDED,
    )
