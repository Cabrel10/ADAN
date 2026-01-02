"""
Main dashboard application for ADAN BTC/USDT Terminal Dashboard

Provides the live dashboard with real-time updates.
"""

import time
import signal
import sys
from typing import Optional
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .data_collector import DataCollector
from .mock_collector import MockDataCollector
from .generator import generate_dashboard


class AdanBtcDashboard:
    """
    Main ADAN BTC/USDT Dashboard application.
    
    Provides real-time monitoring of trading activity with live updates.
    """
    
    def __init__(
        self,
        data_collector: Optional[DataCollector] = None,
        refresh_rate: float = 2.0,
        console: Optional[Console] = None,
    ):
        """
        Initialize the dashboard application.
        
        Args:
            data_collector: Data collector instance (uses MockDataCollector if None)
            refresh_rate: Refresh rate in seconds (default: 2.0)
            console: Rich Console instance (creates new if None)
        """
        self.data_collector = data_collector or MockDataCollector(seed=42)
        self.refresh_rate = refresh_rate
        self.console = console or Console()
        self.running = False
        self.start_time = datetime.now()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.console.print("\n[bold yellow]📊 Shutting down dashboard...[/]")
        self.running = False
    
    def _create_startup_panel(self) -> Panel:
        """Create startup panel"""
        startup_text = Text()
        startup_text.append("🎯 ADAN BTC/USDT Dashboard\n", style="bold cyan")
        startup_text.append("📊 Initializing...\n", style="bold yellow")
        startup_text.append(f"⏰ Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n", style="dim white")
        startup_text.append(f"🔄 Refresh Rate: {self.refresh_rate}s\n", style="dim white")
        startup_text.append("\n⌨️  Press Ctrl+C to exit", style="dim cyan")
        
        return Panel(
            startup_text,
            title="[bold magenta]🚀 ADAN DASHBOARD[/]",
            border_style="magenta",
        )
    
    def _create_error_panel(self, error: Exception) -> Panel:
        """Create error panel"""
        error_text = Text()
        error_text.append("❌ Dashboard Error\n\n", style="bold red")
        error_text.append(f"Error: {str(error)}\n", style="red")
        error_text.append(f"Type: {type(error).__name__}\n", style="dim red")
        error_text.append("\n🔄 Retrying in next refresh cycle...", style="dim yellow")
        
        return Panel(
            error_text,
            title="[bold red]ERROR[/]",
            border_style="red",
        )
    
    def run(self) -> None:
        """
        Run the dashboard with live updates.
        
        This method will run indefinitely until interrupted.
        """
        self.running = True
        
        # Show startup message
        self.console.print("\n[bold cyan]🎯 Starting ADAN BTC/USDT Dashboard...[/]")
        self.console.print(f"[dim]Refresh rate: {self.refresh_rate}s[/]")
        self.console.print(f"[dim]Data source: {type(self.data_collector).__name__}[/]")
        self.console.print("[dim]Press Ctrl+C to exit[/]\n")
        
        # Connect data collector
        try:
            if not self.data_collector.is_connected():
                self.console.print("[yellow]📡 Connecting to data sources...[/]")
                if not self.data_collector.connect():
                    self.console.print("[red]❌ Failed to connect to data sources[/]")
                    return
                self.console.print("[green]✅ Connected to data sources[/]")
        except Exception as e:
            self.console.print(f"[red]❌ Connection error: {e}[/]")
            return
        
        # Start live dashboard
        try:
            with Live(
                self._create_startup_panel(),
                console=self.console,
                screen=True,
                refresh_per_second=1 / self.refresh_rate,
            ) as live:
                
                iteration = 0
                
                while self.running:
                    try:
                        iteration += 1
                        
                        # Generate dashboard
                        dashboard = generate_dashboard(self.data_collector, self.console)
                        
                        # Update live display
                        live.update(dashboard)
                        
                        # Log iteration (every 10th iteration)
                        if iteration % 10 == 0:
                            runtime = datetime.now() - self.start_time
                            self.console.log(
                                f"📊 Dashboard iteration {iteration} | "
                                f"Runtime: {runtime} | "
                                f"Connected: {self.data_collector.is_connected()}"
                            )
                        
                        # Sleep for refresh rate
                        time.sleep(self.refresh_rate)
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        # Show error but continue running
                        error_panel = self._create_error_panel(e)
                        live.update(error_panel)
                        
                        self.console.log(f"❌ Dashboard error: {e}")
                        time.sleep(self.refresh_rate)
        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.console.print(f"[red]❌ Fatal error: {e}[/]")
        finally:
            # Cleanup
            try:
                if self.data_collector.is_connected():
                    self.data_collector.disconnect()
            except Exception:
                pass
            
            runtime = datetime.now() - self.start_time
            self.console.print(f"\n[bold green]✅ Dashboard stopped gracefully[/]")
            self.console.print(f"[dim]Total runtime: {runtime}[/]")
            self.console.print(f"[dim]Total iterations: {iteration}[/]\n")
    
    def run_once(self) -> None:
        """
        Generate and display dashboard once (for testing).
        """
        try:
            # Connect if needed
            if not self.data_collector.is_connected():
                self.data_collector.connect()
            
            # Generate and display dashboard
            dashboard = generate_dashboard(self.data_collector, self.console)
            self.console.print(dashboard)
            
        except Exception as e:
            error_panel = self._create_error_panel(e)
            self.console.print(error_panel)
        finally:
            # Cleanup
            try:
                if self.data_collector.is_connected():
                    self.data_collector.disconnect()
            except Exception:
                pass
