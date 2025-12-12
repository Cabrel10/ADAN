"""
System Health section renderer for ADAN Dashboard

Displays API/feed/model/DB status, CPU/memory, and alerts.
"""

from typing import Dict, Any, List
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ..formatters import format_status_symbol


def render_system_health(health_data: Dict[str, Any]) -> Panel:
    """
    Render the system health section.
    
    Args:
        health_data: System health metrics dictionary
    
    Returns:
        Rich Panel containing system health information
    """
    # Create table
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Component", style="bold cyan", width=18)
    table.add_column("Status", style="bold white")
    
    # API status
    api_status = health_data.get("api_status", False)
    api_latency = health_data.get("api_latency_ms", 0)
    api_symbol = format_status_symbol(api_status)
    api_color = "green" if api_latency < 100 else "yellow" if api_latency < 500 else "red"
    table.add_row(
        "API",
        f"{api_symbol} [bold {api_color}]{api_latency}ms[/]"
    )
    
    # Data feed status
    feed_status = health_data.get("feed_status", False)
    feed_lag = health_data.get("feed_lag_ms", 0)
    feed_symbol = format_status_symbol(feed_status)
    feed_color = "green" if feed_lag < 200 else "yellow" if feed_lag < 1000 else "red"
    table.add_row(
        "Data Feed",
        f"{feed_symbol} [bold {feed_color}]{feed_lag}ms[/]"
    )
    
    # Model status
    model_status = health_data.get("model_status", False)
    model_latency = health_data.get("model_latency_ms", 0)
    model_symbol = format_status_symbol(model_status)
    model_color = "green" if model_latency < 150 else "yellow" if model_latency < 500 else "red"
    table.add_row(
        "Model",
        f"{model_symbol} [bold {model_color}]{model_latency}ms[/]"
    )
    
    # Database status
    db_status = health_data.get("db_status", False)
    db_symbol = format_status_symbol(db_status)
    table.add_row(
        "Database",
        f"{db_symbol} [bold green]OK[/]" if db_status else f"{db_symbol} [bold red]ERROR[/]"
    )
    
    # CPU usage
    cpu_percent = health_data.get("cpu_percent", 0.0)
    cpu_color = "green" if cpu_percent < 50 else "yellow" if cpu_percent < 80 else "red"
    table.add_row(
        "CPU",
        f"[bold {cpu_color}]{cpu_percent:.1f}%[/]"
    )
    
    # Memory usage
    memory_gb = health_data.get("memory_gb", 0.0)
    memory_total = health_data.get("memory_total_gb", 4.0)
    memory_pct = (memory_gb / memory_total * 100) if memory_total > 0 else 0.0
    memory_color = "green" if memory_pct < 50 else "yellow" if memory_pct < 80 else "red"
    table.add_row(
        "Memory",
        f"[bold {memory_color}]{memory_gb:.1f}GB / {memory_total:.1f}GB[/]"
    )
    
    # Threads
    threads = health_data.get("threads", 0)
    table.add_row(
        "Threads",
        f"[bold cyan]{threads}[/]"
    )
    
    # Uptime
    uptime_percent = health_data.get("uptime_percent", 0.0)
    uptime_color = "green" if uptime_percent > 99 else "yellow" if uptime_percent > 95 else "red"
    table.add_row(
        "Uptime",
        f"[bold {uptime_color}]{uptime_percent:.1f}%[/]"
    )
    
    # Alerts section
    alerts = health_data.get("alerts", [])
    
    if alerts:
        table.add_row("", "")  # Spacer
        table.add_row("[bold yellow]⚠️  ALERTS[/]", "")
        
        for alert in alerts[:5]:  # Show max 5 alerts
            severity = alert.get("severity", "INFO")
            message = alert.get("message", "Unknown alert")
            
            if severity == "CRITICAL":
                severity_color = "bold red"
                severity_icon = "🔴"
            elif severity == "WARNING":
                severity_color = "bold yellow"
                severity_icon = "🟡"
            else:
                severity_color = "bold cyan"
                severity_icon = "ℹ️"
            
            table.add_row(
                "",
                f"{severity_icon} [{severity_color}]{message}[/]"
            )
    
    # Create panel
    return Panel(
        table,
        title="[bold violet]⚙️  SYSTEM HEALTH[/]",
        border_style="violet",
        box=box.ROUNDED,
    )
