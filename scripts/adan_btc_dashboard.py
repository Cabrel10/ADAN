#!/usr/bin/env python3
"""
ADAN BTC/USDT Terminal Dashboard - Main Entry Point

Professional real-time monitoring interface for ADAN trading bot.

Usage:
    python scripts/adan_btc_dashboard.py [options]
    
Options:
    --mock          Use mock data collector (default)
    --real          Use real data collector (requires ADAN system)
    --refresh RATE  Refresh rate in seconds (default: 2.0)
    --once          Run once and exit (for testing)
    --help          Show this help message
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from adan_trading_bot.dashboard import AdanBtcDashboard
from adan_trading_bot.dashboard.mock_collector import MockDataCollector


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ADAN BTC/USDT Terminal Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with mock data (default)
    python scripts/adan_btc_dashboard.py
    
    # Run with custom refresh rate
    python scripts/adan_btc_dashboard.py --refresh 1.0
    
    # Run once for testing
    python scripts/adan_btc_dashboard.py --once
    
    # Use real data collector (requires ADAN system)
    python scripts/adan_btc_dashboard.py --real
"""
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock data collector (default)"
    )
    
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real data collector (requires ADAN system)"
    )
    
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Refresh rate in seconds (default: 2.0)"
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
    if args.real:
        # Try to import and create real data collector
        try:
            # This would be implemented in Phase 5
            from adan_trading_bot.dashboard.real_collector import RealDataCollector
            return RealDataCollector()
        except ImportError:
            console = Console()
            console.print("[red]❌ Real data collector not available yet[/]")
            console.print("[yellow]📊 Falling back to mock data collector[/]")
            return MockDataCollector(seed=args.seed)
    else:
        # Use mock data collector
        return MockDataCollector(seed=args.seed)


def main():
    """Main entry point"""
    args = parse_arguments()
    console = Console()
    
    # Validate arguments
    if args.refresh <= 0:
        console.print("[red]❌ Refresh rate must be positive[/]")
        sys.exit(1)
    
    if args.refresh < 0.1:
        console.print("[yellow]⚠️  Very fast refresh rate may impact performance[/]")
    
    # Create data collector
    try:
        data_collector = create_data_collector(args)
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
        else:
            dashboard.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Dashboard interrupted by user[/]")
    except Exception as e:
        console.print(f"\n[red]❌ Dashboard error: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
