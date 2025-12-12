"""
Dashboard generator for ADAN Dashboard

Combines all section renderers into complete layout.
"""

from rich.layout import Layout
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .models import PortfolioState
from .data_collector import DataCollector
from .layout import get_optimal_layout, update_layout_content
from .sections import (
    render_header,
    render_decision_matrix,
    render_positions,
    render_closed_trades,
    render_performance,
    render_system_health,
)


def generate_dashboard(data_collector: DataCollector, console: Console) -> Layout:
    """
    Generate complete dashboard layout with all sections.
    
    Args:
        data_collector: Data collector instance
        console: Rich Console object
    
    Returns:
        Complete Rich Layout with all sections rendered
    """
    # Get optimal layout for terminal size
    layout = get_optimal_layout(console)
    
    try:
        # Collect all data
        portfolio = data_collector.get_portfolio_state()
        signal = data_collector.get_current_signal()
        market_context = data_collector.get_market_context()
        positions = data_collector.get_open_positions()
        trades = data_collector.get_closed_trades(limit=5)
        health_data = data_collector.get_system_health()
        
        # Render all sections
        sections = {}
        
        # Header
        sections["header"] = render_header(portfolio)
        
        # Decision matrix (only if we have signal and market context)
        if signal and market_context:
            sections["decision_matrix"] = render_decision_matrix(signal, market_context)
        else:
            sections["decision_matrix"] = Panel(
                Text("⏳ Waiting for signal and market data...", style="dim yellow"),
                title="[bold cyan]📊 DECISION MATRIX[/]",
                border_style="cyan",
            )
        
        # Positions
        sections["positions"] = render_positions(positions)
        
        # Trades
        sections["trades"] = render_closed_trades(trades)
        
        # Performance
        sections["performance"] = render_performance(trades)
        
        # System health
        sections["health"] = render_system_health(health_data)
        
        # Update layout with rendered sections
        update_layout_content(layout, sections)
        
    except Exception as e:
        # Error handling - show error message
        error_panel = Panel(
            Text(f"❌ Error generating dashboard: {str(e)}", style="bold red"),
            title="[bold red]ERROR[/]",
            border_style="red",
        )
        
        # Put error in header and clear other sections
        layout["header"].update(error_panel)
        
        # Clear other sections with placeholder
        placeholder = Panel(
            Text("Dashboard unavailable due to error", style="dim white"),
            border_style="dim white",
        )
        
        for section_name in ["decision_matrix", "positions", "trades", "performance", "health"]:
            if section_name in layout:
                layout[section_name].update(placeholder)
    
    return layout


def generate_dashboard_from_portfolio(portfolio: PortfolioState, console: Console) -> Layout:
    """
    Generate dashboard from a PortfolioState object (for testing).
    
    Args:
        portfolio: Portfolio state with all data
        console: Rich Console object
    
    Returns:
        Complete Rich Layout with all sections rendered
    """
    # Get optimal layout for terminal size
    layout = get_optimal_layout(console)
    
    try:
        # Render all sections from portfolio data
        sections = {}
        
        # Header
        sections["header"] = render_header(portfolio)
        
        # Decision matrix
        if portfolio.current_signal and portfolio.market_context:
            sections["decision_matrix"] = render_decision_matrix(
                portfolio.current_signal,
                portfolio.market_context
            )
        else:
            sections["decision_matrix"] = Panel(
                Text("⏳ Waiting for signal and market data...", style="dim yellow"),
                title="[bold cyan]📊 DECISION MATRIX[/]",
                border_style="cyan",
            )
        
        # Positions
        sections["positions"] = render_positions(portfolio.open_positions)
        
        # Trades
        sections["trades"] = render_closed_trades(portfolio.closed_trades)
        
        # Performance
        sections["performance"] = render_performance(portfolio.closed_trades)
        
        # System health
        sections["health"] = render_system_health(portfolio.system_health)
        
        # Update layout with rendered sections
        update_layout_content(layout, sections)
        
    except Exception as e:
        # Error handling
        error_panel = Panel(
            Text(f"❌ Error generating dashboard: {str(e)}", style="bold red"),
            title="[bold red]ERROR[/]",
            border_style="red",
        )
        
        layout["header"].update(error_panel)
    
    return layout
