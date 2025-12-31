#!/usr/bin/env python3
"""
ADAN BTC/USDT Terminal Dashboard - Main Entry Point

Professional real-time monitoring interface for ADAN trading bot.

Usage:
    python scripts/adan_btc_dashboard.py [options]
    
Options:
    --refresh RATE  Refresh rate in seconds (default: 60.0)
    --once          Run once and exit (for testing)
    --mock          Use mock data collector (for testing only)
    --help          Show this help message
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from adan_trading_bot.dashboard import AdanBtcDashboard
from adan_trading_bot.dashboard.real_collector import RealDataCollector
from adan_trading_bot.dashboard.mock_collector import MockDataCollector


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ADAN BTC/USDT Terminal Dashboard - Real Market Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with REAL data from Binance testnet (default)
    python scripts/adan_btc_dashboard.py
    
    # Run with custom refresh rate (real data)
    python scripts/adan_btc_dashboard.py --refresh 30.0
    
    # Run once for testing (real data)
    python scripts/adan_btc_dashboard.py --once
    
    # Use mock data for testing (fallback only)
    python scripts/adan_btc_dashboard.py --mock
    
    # Run with mock data and custom refresh
    python scripts/adan_btc_dashboard.py --mock --refresh 5.0
"""
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data collector (for testing only). Default is REAL data from Binance testnet."
    )
    
    parser.add_argument(
        "--refresh",
        type=float,
        default=30.0,
        help="Refresh rate in seconds (default: 30.0 for real-time market data from Binance)"
    )
    
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (for testing)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mock data (default: 42)"
    )
    
    return parser.parse_args()


def create_data_collector(args):
    """Create appropriate data collector based on arguments"""
    if args.mock:
        # Use mock data collector for testing
        return MockDataCollector(seed=args.seed)
    else:
        # Use real data collector (default)
        return RealDataCollector()


def main():
    """Main entry point"""
    args = parse_arguments()
    console = Console()
    
    # Print header
    console.print("\n[bold cyan]🔴 ADAN BTC/USDT Dashboard - Real Market Data[/bold cyan]")
    console.print("[cyan]📡 Data Source: Binance Testnet (LIVE)[/cyan]\n")
    
    # Validate arguments
    if args.refresh <= 0:
        console.print("[red]❌ Refresh rate must be positive[/]")
        sys.exit(1)
    
    if args.refresh < 0.1:
        console.print("[yellow]⚠️  Very fast refresh rate may impact performance[/]")
    
    # Create data collector
    try:
        data_collector = create_data_collector(args)
        if args.mock:
            collector_type = "Mock Data (Testing)"
            console.print(f"[yellow]⚠️  Using {collector_type} Collector[/]")
        else:
            collector_type = "Real Data (Live Binance)"
            console.print(f"[green]✅ Using {collector_type} Collector[/]")
            console.print(f"[cyan]   Refresh Rate: {args.refresh}s[/cyan]")
    except Exception as e:
        console.print(f"[red]❌ Failed to create data collector: {e}[/]")
        sys.exit(1)
    
    # Create dashboard
    try:
        dashboard = AdanBtcDashboard(
            data_collector=data_collector,
            refresh_rate=args.refresh,
            console=console,
        )
    except Exception as e:
        console.print(f"[red]❌ Failed to create dashboard: {e}[/]")
        sys.exit(1)
    
    # Run dashboard
    try:
        if args.once:
            console.print("[cyan]📊 Running dashboard once...[/]\n")
            dashboard.run_once()
            console.print("\n[green]✅ Dashboard completed[/]")
        else:
            console.print("[cyan]📊 Starting live dashboard...[/]")
            console.print("[yellow]Press Ctrl+C to exit[/]\n")
            dashboard.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Dashboard stopped by user[/]")
    except Exception as e:
        console.print(f"\n[red]❌ Dashboard error: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
