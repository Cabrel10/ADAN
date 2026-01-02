"""
Performance Analytics section renderer for ADAN Dashboard

Displays comprehensive trading statistics.
"""

from typing import List
from rich.table import Table
from rich.panel import Panel
from rich import box

from ..models import ClosedTrade
from ..formatters import (
    format_percentage,
    format_ratio,
    format_price,
    format_time,
)


def render_performance(trades: List[ClosedTrade]) -> Panel:
    """
    Render the performance analytics section.
    
    Args:
        trades: List of closed trades
    
    Returns:
        Rich Panel containing performance metrics
    """
    # Create table
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Metric", style="bold cyan", width=20)
    table.add_column("Value", style="bold white")
    
    if not trades:
        # No trades yet
        table.add_row("Win Rate", "[dim]0.0%[/]")
        table.add_row("Profit Factor", "[dim]0.00[/]")
        table.add_row("Total Trades", "[dim]0[/]")
        table.add_row("Sharpe Ratio", "[dim]N/A[/]")
        table.add_row("Sortino Ratio", "[dim]N/A[/]")
        table.add_row("Max Drawdown", "[dim]N/A[/]")
    else:
        # Calculate metrics
        wins = sum(1 for t in trades if t.is_win)
        win_rate = (wins / len(trades)) * 100 if trades else 0.0
        
        gross_profit = sum(t.realized_pnl_usd for t in trades if t.is_win)
        gross_loss = abs(sum(t.realized_pnl_usd for t in trades if not t.is_win))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        
        best_trade = max((t.realized_pnl_usd for t in trades), default=0.0)
        worst_trade = min((t.realized_pnl_usd for t in trades), default=0.0)
        
        durations = [t.duration.total_seconds() / 3600 for t in trades]
        avg_holding_time = sum(durations) / len(durations) if durations else 0.0
        
        # Win rate
        win_rate_color = "green" if win_rate >= 50 else "yellow" if win_rate >= 40 else "red"
        table.add_row(
            "Win Rate",
            f"[bold {win_rate_color}]{format_percentage(win_rate, decimals=1, include_sign=False)}[/]"
        )
        
        # Profit factor
        pf_color = "green" if profit_factor > 1.5 else "yellow" if profit_factor > 1.0 else "red"
        pf_str = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
        table.add_row(
            "Profit Factor",
            f"[bold {pf_color}]{pf_str}[/]"
        )
        
        # Total trades
        table.add_row(
            "Total Trades",
            f"[bold cyan]{len(trades)}[/]"
        )
        
        # Best trade
        best_color = "green" if best_trade > 0 else "red"
        table.add_row(
            "Best Trade",
            f"[bold {best_color}]{format_price(best_trade)}[/]"
        )
        
        # Worst trade
        worst_color = "red" if worst_trade < 0 else "green"
        table.add_row(
            "Worst Trade",
            f"[bold {worst_color}]{format_price(worst_trade)}[/]"
        )
        
        # Average holding time
        table.add_row(
            "Avg Holding Time",
            f"[bold cyan]{format_time(avg_holding_time * 3600)}[/]"
        )
        
        # Gross profit/loss
        table.add_row(
            "Gross Profit",
            f"[bold green]{format_price(gross_profit)}[/]"
        )
        
        table.add_row(
            "Gross Loss",
            f"[bold red]{format_price(gross_loss)}[/]"
        )
    
    # Create panel
    return Panel(
        table,
        title="[bold magenta]📊 PERFORMANCE ANALYTICS[/]",
        border_style="magenta",
        box=box.ROUNDED,
    )
