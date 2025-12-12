"""
Header section renderer for ADAN Dashboard

Displays global portfolio state and system status at a glance.
"""

from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from datetime import datetime

from ..models import PortfolioState
from ..formatters import format_usd, format_percentage, format_time_hms
from ..colors import get_pnl_color, RICH_STYLES


def render_header(portfolio: PortfolioState) -> Panel:
    """
    Render the header section with global portfolio state.
    
    Args:
        portfolio: Current portfolio state
    
    Returns:
        Rich Panel containing header information
    """
    # Calculate runtime
    runtime = datetime.now() - portfolio.timestamp
    runtime_str = format_time_hms(runtime)
    
    # Build header text
    header_text = Text()
    
    # Title
    header_text.append("🎯 ADAN v1.0 - BTC/USDT MONITOR", style="bold cyan")
    header_text.append("\n")
    
    # Portfolio metrics
    header_text.append("Portfolio: ", style="dim white")
    pnl_color = get_pnl_color(portfolio.total_pnl_pct)
    header_text.append(
        f"{format_usd(portfolio.total_value_usd)} ({format_percentage(portfolio.total_pnl_pct)})",
        style=f"bold {pnl_color}"
    )
    
    header_text.append(" │ ", style="dim white")
    header_text.append("Positions: ", style="dim white")
    header_text.append(f"{portfolio.position_count}", style="bold cyan")
    
    header_text.append(" │ ", style="dim white")
    header_text.append("Win Rate: ", style="dim white")
    header_text.append(f"{portfolio.win_rate:.1f}%", style="bold green")
    
    header_text.append("\n")
    
    # Capital and runtime
    header_text.append("Capital: ", style="dim white")
    header_text.append(f"{format_usd(portfolio.available_capital_usd)}", style="bold yellow")
    
    header_text.append(" │ ", style="dim white")
    header_text.append("P&L: ", style="dim white")
    pnl_color = get_pnl_color(portfolio.total_pnl_pct)
    header_text.append(
        f"{format_usd(portfolio.total_pnl_usd)}",
        style=f"bold {pnl_color}"
    )
    
    header_text.append(" │ ", style="dim white")
    header_text.append("Runtime: ", style="dim white")
    header_text.append(f"{runtime_str}", style="bold magenta")
    
    # Create panel
    return Panel(
        header_text,
        style="bold cyan",
        border_style="cyan",
        padding=(0, 1),
    )
