#!/usr/bin/env python3
"""
🎯 ADAN Terminal Dashboard - Paper Trading Monitor
Real-time trading status with vibrant aesthetics
"""

import json
import time
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box
from rich.style import Style

# ✅ JOUR 2: Importer le système unifié
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
try:
    from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
    UNIFIED_DB_AVAILABLE = True
except ImportError:
    UNIFIED_DB_AVAILABLE = False
    UnifiedMetricsDB = None

console = Console()


def create_sparkline(values, width=40):
    """Create a colorful ASCII sparkline with gradient"""
    if not values or len(values) < 2:
        return Text("─" * width, style="dim cyan")
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return Text("─" * width, style="dim cyan")
    
    # Normalize values
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    # Scale to width
    step = len(normalized) / width
    sampled = [normalized[int(i * step)] for i in range(width)]
    
    # Sparkline characters with color gradient
    chars = "  ▂▃▄▅▆▇█"
    spark = Text()
    
    for i, v in enumerate(sampled):
        char = chars[min(int(v * 8), 7)]
        # Color gradient from green (low) to orange to magenta (high)
        if v < 0.33:
            color = "green"
        elif v < 0.66:
            color = "yellow"
        else:
            color = "magenta"
        spark.append(char, style=color)
    
    return spark

STATUS_TTL = 90  # seconds before status.json considered STALE

def read_status(path: Path):
    try:
        st = json.loads(path.read_text())
        # stale check
        ts = st.get("timestamp", None)
        if ts is None:
            mtime = path.stat().st_mtime
            ts = mtime
        age = time.time() - float(ts)
        st["_stale"] = age > STATUS_TTL
        return st
    except Exception:
        return {"_stale": True}

# Alias for compatibility
load_status = read_status

def tail_lines(path: Path, max_bytes=200000, max_lines=300):
    try:
        with path.open("rb") as f:
            f.seek(0,2)
            size = f.tell()
            to_read = min(size, max_bytes)
            f.seek(max(0, size - to_read))
            data = f.read().decode(errors="ignore")
            lines = data.splitlines()
            return lines[-max_lines:]
    except Exception:
        return []

# --- log parsing helpers (best-effort) ---
EXEC_RE = re.compile(r"(✅|INFO).*?(SELL|BUY).*(Executed|Executed:|Executed:)\s*[:#]?\s*([0-9]+).*?([0-9]+\.[0-9]+)|(Executed:\s*([0-9]+))", re.IGNORECASE)
TRADE_LINE_RE = re.compile(r"(SELL|BUY).*(Executed|Executed:|Executed Order).*?([0-9]+)", re.IGNORECASE)
PRICE_RE = re.compile(r"(Price[:=]?\s*)([0-9]+\.[0-9]+)")
OHLC_RE = re.compile(r"O[:=]?\s*([0-9]+\.[0-9]+).*H[:=]?\s*([0-9]+\.[0-9]+).*L[:=]?\s*([0-9]+\.[0-9]+).*C[:=]?\s*([0-9]+\.[0-9]+)", re.IGNORECASE)
CANDLE_RE = re.compile(r"O:([0-9]+\.[0-9]+)\s+H:([0-9]+\.[0-9]+)\s+L:([0-9]+\.[0-9]+)\s+C:([0-9]+\.[0-9]+)", re.IGNORECASE)
TRADE_EXEC_RE = re.compile(r"(?:SELL|BUY).*Executed[:\s]*([0-9]+).*?[:\s]*([0-9]+\.[0-9]+)", re.IGNORECASE)
ORDER_RE = re.compile(r"Order[:#]?\s*#?(\d+)", re.IGNORECASE)
QTY_RE = re.compile(r"Qty[:=]?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
TP_SL_RE = re.compile(r"TP[:=]?\s*([0-9]+\.[0-9]+)|SL[:=]?\s*([0-9]+\.[0-9]+)|TP/SL[:=]?\s*([0-9]+\.[0-9]+)\/([0-9]+\.[0-9]+)", re.IGNORECASE)
POSITION_OPEN_RE = re.compile(r"(POSITION.*OPENED|POSITION OUVERTE|OPEN POSITION)", re.IGNORECASE)
POSITION_CLOSE_RE = re.compile(r"(POSITION.*CLOSED|POSITION FERMÉE|CLOSE POSITION)", re.IGNORECASE)
WAIT_RE = re.compile(r"Waiting\s+([0-9]+)s", re.IGNORECASE)
LATENCY_RE = re.compile(r"Latency[:\s]*([0-9]+)\s*ms", re.IGNORECASE)
EXCHANGE_RE = re.compile(r"(binance[-\s]*testnet|binance|bybit|ftx|kraken)", re.IGNORECASE)
CLOSES_RE = re.compile(r"close[:=]?\s*([0-9]+\.[0-9]+)", re.IGNORECASE)

def parse_logs_to_status(lines: list[str]) -> dict:
    status = {}
    recent_trades = []
    price_series = []
    last_price = None
    for ln in reversed(lines):  # scan backward to get recent first
        # price
        m = PRICE_RE.search(ln)

def parse_logs_for_data(log_path, data):
    """Parse logs to extract trading data as fallback"""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        # Parse recent lines for market data, trades, etc.
        for line in reversed(lines[-100:]):  # Last 100 lines
            # Extract price
            if 'price' in line.lower() and not data.get('market_data', {}).get('price'):
                match = re.search(r'price[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
                if match:
                    data.setdefault('market_data', {})['price'] = float(match.group(1))
            
            # Extract trades
            if '🔵 BUY' in line or '🔴 SELL' in line:
                match = re.search(r'(🔵 BUY|🔴 SELL).*?(\d+\.?\d*)\s+@\s+(\d+\.?\d*)', line)
                if match:
                    data.setdefault('recent_trades', []).append({
                        'side': 'BUY' if '🔵' in match.group(1) else 'SELL',
                        'qty': float(match.group(2)),
                        'price': float(match.group(3))
                    })
            
            # Extract position
            if 'position' in line.lower() and 'size' in line.lower():
                match = re.search(r'size[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
                if match:
                    data.setdefault('position', {})['size'] = float(match.group(1))
                    
            # Extract latency
            if 'latency' in line.lower():
                match = re.search(r'latency[:\s]+(\d+)', line, re.IGNORECASE)
                if match:
                    data.setdefault('system', {})['latency_ms'] = int(match.group(1))
                    
            # Extract exchange
            if 'exchange' in line.lower() or 'binance' in line.lower():
                if 'binance' in line.lower():
                    data.setdefault('system', {})['exchange'] = 'Binance Testnet'
                    
    except Exception as e:
        console.print(f"[yellow]⚠ Log parsing error: {e}[/]")
    
    return data


def generate_dashboard(data):
    """Generate the vibrant dashboard layout"""
    layout = Layout()
    
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=12)
    )
    
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # HEADER - Vibrant market info
    # ═══════════════════════════════════════════════════════════════════
    market = data.get('market_data', {})
    symbol = market.get('symbol', 'N/A')
    price = float(market.get('price', 0.0))
    timeframe = market.get('timeframe', '5m')
    
    header_text = Text()
    header_text.append("🎯 ", style="bold magenta")
    header_text.append(f"{symbol}", style="bold cyan")
    header_text.append(" @ ", style="dim white")
    header_text.append(f"{timeframe}", style="bold yellow")
    header_text.append(" │ ", style="magenta")
    header_text.append("💰 ", style="bold yellow")
    header_text.append(f"${price:,.2f}", style="bold green")
    
    layout["header"].update(Panel(header_text, box=box.DOUBLE, style="bold magenta", border_style="magenta"))
    
    # ═══════════════════════════════════════════════════════════════════
    # ACCOUNT INFO - Green/Orange theme
    # ═══════════════════════════════════════════════════════════════════
    account = data.get('account', {})
    # Use virtual balance for "Balance" display to match bot logic, or real?
    # User said: "Dashboard affiche balance ($29) et pnl (toujours $0)" -> "Problème actuel"
    # User wants: "Real Equity", "USDT", "BTC", "PnL (réel)"
    
    # We will show Virtual Balance as "Trading Capital" and Real Balance as "Real Assets"
    
    usdt_virtual = float(account.get('usdt_virtual', account.get('balance', 0.0)))
    initial_balance = float(account.get('initial_balance', 0.0))
    
    # Real metrics
    real_equity = float(account.get('real_equity', 0.0))
    usdt_real = float(account.get('usdt_balance', 0.0))
    btc_real = float(account.get('btc_balance', 0.0))
    pnl_abs = float(account.get('pnl_absolute', 0.0))
    pnl_pct = float(account.get('pnl_percent', 0.0))
    
    stale_flag = data.get("_stale", False)
    
    account_table = Table(show_header=False, box=None, padding=(0, 2), style="on #001a00")
    account_table.add_column("Metric", style="bold yellow", width=16)
    account_table.add_column("Value", style="bold white")
    
    # Virtual Capital (What the bot uses)
    account_table.add_row("💵 Virtual Cap", f"[bold green]${usdt_virtual:.2f}[/]")
    
    # PnL (Real)
    pnl_color = "bold green" if pnl_abs >= 0 else "bold red"
    pnl_icon = "📈" if pnl_abs >= 0 else "📉"
    account_table.add_row(f"{pnl_icon} PnL (Real)", f"[{pnl_color}]${pnl_abs:.2f} ({pnl_pct:+.2f}%)[/]")
    
    # Real Metrics
    if real_equity > 0:
        account_table.add_row("💎 Real Equity", f"[bold cyan]${real_equity:,.2f}[/]")
    if usdt_real > 0 or btc_real > 0:
        account_table.add_row("💵 Real USDT", f"[dim]${usdt_real:.2f}[/]")
        account_table.add_row("₿ Real BTC", f"[dim]{btc_real:.8f}[/]")
        
    # Stale status
    status_age = int(time.time() - data.get("timestamp", time.time()))
    status_color = "bold red" if stale_flag else "dim white"
    status_text = "STALE" if stale_flag else f"{status_age}s ago"
    account_table.add_row("⏱ Status", f"[{status_color}]{status_text}[/]")

    # We can remove the separate Real Metrics table since we integrated it or keep it?
    # The user's patch B puts everything in account_table.
    # I will stick to the user's preference of having a "Real Metrics" block if possible, 
    # but the user's patch B example shows merging them into account_table.
    # "Colle ce snippet... account_table = Table.grid()... add_row('Real Equity')..."
    # So I will merge them.
    
    # Remove the old Real Metrics table creation to avoid duplicates or errors
    real_table = Table(show_header=False, box=None, padding=(0, 2), style="on #0a001a")
    # Empty real table to satisfy layout update if I don't remove the panel
    # But I should probably remove the panel from layout if I merge.
    # User's patch B: "# then display account_table in panel"
    # User's patch B doesn't show the layout construction.
    # I'll keep the layout as is (3 panels on left) but make the second panel empty or reuse it.
    # Actually, I'll just put the detailed breakdown in the second panel.
    
    real_table.add_column("Metric", style="bold magenta", width=16)
    real_table.add_column("Value", style="bold white")
    real_table.add_row("USDT Balance", f"${usdt_real:,.2f}")
    real_table.add_row("BTC Balance", f"{btc_real:.8f}")
    real_table.add_row("Real Equity", f"${real_equity:,.2f}")
    # ═══════════════════════════════════════════════════════════════════
    # POSITION - Orange/Violet theme
    # ═══════════════════════════════════════════════════════════════════
    position = data.get('position', {})
    pos_size = float(position.get('size', 0.0))
    pos_entry = float(position.get('entry_price', 0.0))
    pos_side = position.get('side', 'NONE')
    
    pos_table = Table(show_header=False, box=None, padding=(0, 2), style="on #1a0040")
    pos_table.add_column("Metric", style="bold magenta", width=16)
    pos_table.add_column("Value", style="bold white")
    
    if pos_size > 0:
        side_color = "bold green" if pos_side == "LONG" else "bold red"
        side_icon = "🚀" if pos_side == "LONG" else "🔻"
        pos_table.add_row(f"{side_icon} Type", f"[{side_color}]{pos_side}[/]")
        pos_table.add_row("⚖️  Taille", f"[bold cyan]{pos_size:.4f}[/]")
        pos_table.add_row("🎯 Entrée", f"[bold yellow]${pos_entry:.2f}[/]")
        if price > 0 and pos_entry > 0:
            pnl_pct_pos = ((price - pos_entry) / pos_entry * 100) if pos_side == "LONG" else ((pos_entry - price) / pos_entry * 100)
            pnl_color_pos = "bold green" if pnl_pct_pos >= 0 else "bold red"
            pos_table.add_row("💎 Variation", f"[{pnl_color_pos}]{pnl_pct_pos:+.2f}%[/]")
    else:
        pos_table.add_row("⭕ Status", "[dim yellow]Pas de position[/]")
        pos_table.add_row("🎲 Mode", "[dim cyan]Recherche signal...[/]")
    
    # Combine left panels
    left_content = Table.grid()
    left_content.add_row(Panel(account_table, title="[bold green]💰 COMPTE[/]", border_style="bold green", box=box.ROUNDED))
    left_content.add_row(Panel(real_table, title="[bold magenta]🔎 REAL METRICS[/]", border_style="bold magenta", box=box.ROUNDED))
    left_content.add_row(Panel(pos_table, title="[bold magenta]📍 POSITION[/]", border_style="bold magenta", box=box.ROUNDED))

    
    layout["left"].update(left_content)
    
    # ═══════════════════════════════════════════════════════════════════
    # RECENT TRADES - Green/Red with Orange highlights
    # ═══════════════════════════════════════════════════════════════════
    trades = data.get('recent_trades', [])
    trade_table = Table(show_header=True, box=box.ROUNDED, style="on #002000")
    trade_table.add_column("⏰ Heure", style="dim cyan", width=10)
    trade_table.add_column("📊 Type", justify="center", width=12)
    trade_table.add_column("📦 Qté", justify="right", style="bold white", width=10)
    trade_table.add_column("💵 Prix", justify="right", style="bold yellow", width=12)
    trade_table.add_column("💎 PnL", justify="right", width=12)
    
    for trade in trades[-5:]:  # Last 5 trades
        side = trade.get('side', 'HOLD')
        qty = trade.get('qty', 0.0)
        price_val = trade.get('price', 0.0)
        pnl = trade.get('pnl', 0.0)
        ts = trade.get('timestamp', '')
        
        # Parse timestamp
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            time_str = dt.strftime('%H:%M:%S')
        except:
            time_str = ts[:8] if len(ts) > 8 else ts
        
        if side == "BUY":
            side_style = "bold green"
            side_text = "🟢 BUY"
        elif side == "SELL":
            side_style = "bold red"
            side_text = "🔴 SELL"
        else:
            side_style = "dim white"
            side_text = "⚪ HOLD"
        
        pnl_color = "bold green" if pnl >= 0 else "bold red"
        pnl_icon = "⬆" if pnl >= 0 else "⬇"
        
        trade_table.add_row(
            time_str,
            f"[{side_style}]{side_text}[/]",
            f"[bold cyan]{qty:.4f}[/]",
            f"${price_val:.2f}",
            f"[{pnl_color}]{pnl_icon} {pnl:+.2f}[/]"
        )
    
    if not trades:
        trade_table.add_row(
            "[dim]--:--:--[/]", 
            "[dim yellow]En attente[/]", 
            "[dim]─[/]", 
            "[dim]─[/]", 
            "[dim]─[/]"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # SYSTEM STATUS - Violet/Purple theme
    # ═══════════════════════════════════════════════════════════════════
    system = data.get('system', {})
    pid = system.get('pid', 'N/A')
    latency = system.get('latency_ms', 0)
    mode = system.get('mode', 'paper')
    exchange = system.get('exchange', 'Binance Testnet')
    
    # Latency color coding with icons
    if latency < 500:
        lat_color = "bold green"
        lat_icon = "🚀"
    elif latency < 2000:
        lat_color = "bold yellow"
        lat_icon = "⚡"
    else:
        lat_color = "bold red"
        lat_icon = "🐌"
    
    sys_table = Table(show_header=False, box=None, padding=(0, 2), style="on #1a001a")
    sys_table.add_column("Metric", style="bold violet", width=16)
    sys_table.add_column("Value", style="bold white")
    
    sys_table.add_row("🆔 PID", f"[bold cyan]{pid}[/]")
    sys_table.add_row(f"{lat_icon} Latence", f"[{lat_color}]{latency}ms[/]")
    sys_table.add_row("🎮 Mode", f"[bold yellow]{mode.upper()}[/]")
    sys_table.add_row("🌐 Exchange", f"[bold magenta]{exchange}[/]")
    
    # ═══════════════════════════════════════════════════════════════════
    # SPARKLINE - Gradient visualization
    # ═══════════════════════════════════════════════════════════════════
    price_series = data.get('price_series', [])
    spark_visual = create_sparkline(price_series, width=45)
    
    # Combine right panels
    right_content = Table.grid()
    right_content.add_row(Panel(trade_table, title="[bold green]📈 TRADES RÉCENTS[/]", border_style="bold green", box=box.ROUNDED))
    right_content.add_row(Panel(sys_table, title="[bold violet]⚙️  SYSTÈME[/]", border_style="bold violet", box=box.ROUNDED))
    right_content.add_row(Panel(spark_visual, title="[bold cyan]📊 PRICE CHART[/]", border_style="bold cyan", box=box.ROUNDED))
    
    layout["right"].update(right_content)
    
    # ═══════════════════════════════════════════════════════════════════
    # FOOTER - Logs with color-coded levels
    # ═══════════════════════════════════════════════════════════════════
    logs = data.get('recent_logs', [])
    
    # Filter out noisy warnings
    filtered_logs = []
    for log in logs[-10:]:
        # Skip feature warnings and pandas warnings
        if any(skip in log for skip in ['WARNING - Feature', 'SettingWithCopyWarning', 'pandas.pydata.org']):
            continue
        filtered_logs.append(log)
    
    log_text = Text()
    for log_line in filtered_logs[-8:]:  # Last 8 filtered logs
        # Color code by level
        if 'ERROR' in log_line:
            log_text.append("🔴 ", style="bold red")
            log_text.append(log_line + "\n", style="red")
        elif 'WARNING' in log_line:
            log_text.append("⚠️  ", style="bold yellow")
            log_text.append(log_line + "\n", style="yellow")
        elif '🤖' in log_line or 'Prediction' in log_line:
            log_text.append("🤖 ", style="bold magenta")
            log_text.append(log_line + "\n", style="bold cyan")
        elif 'INFO' in log_line:
            log_text.append("ℹ️  ", style="bold cyan")
            log_text.append(log_line + "\n", style="cyan")
        else:
            log_text.append(log_line + "\n", style="dim white")
    
    if not filtered_logs:
        log_text.append("⏳ En attente de logs...\n", style="dim yellow")
    
    layout["footer"].update(Panel(log_text, title="[bold orange1]📝 LOGS RÉCENTS[/]", border_style="bold orange1", box=box.ROUNDED))
    
    return layout


def tail_log(log_path, n_lines=10):
    """Read last n lines from log file"""
    try:
        with open(log_path, 'r') as f:
            return list(deque(f, maxlen=n_lines))
    except FileNotFoundError:
        return []


def main():
    if len(sys.argv) < 2:
        console.print("[bold red]❌ Usage: python terminal_dashboard.py <status.json> [log_file.log][/]")
        sys.exit(1)
    
    status_path = Path(sys.argv[1])
    log_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Vibrant startup message
    console.print()
    console.print("[bold magenta]╔═══════════════════════════════════════════╗[/]")
    console.print("[bold magenta]║[/]  [bold cyan]🎯 ADAN PAPER TRADING DASHBOARD[/]      [bold magenta]║[/]")
    console.print("[bold magenta]╠═══════════════════════════════════════════╣[/]")
    console.print("[bold magenta]║[/]  [bold green]Status:[/] [bold yellow]Initializing...[/]               [bold magenta]║[/]")
    console.print("[bold magenta]╚═══════════════════════════════════════════╝[/]")
    console.print()
    time.sleep(1)
    
    try:
        # ✅ JOUR 2: Initialiser la base de données unifiée
        db = None
        if UNIFIED_DB_AVAILABLE and UnifiedMetricsDB:
            try:
                db = UnifiedMetricsDB()
            except Exception as e:
                console.print(f"[yellow]⚠️  Impossible de charger la base de données unifiée: {e}[/]")
        
        with Live(generate_dashboard({}), refresh_per_second=2, console=console, screen=True) as live:
            while True:
                data = load_status(status_path)
                
                # Fallback to log parsing if JSON is incomplete
                if log_path and not data.get('market_data'):
                    data = parse_logs_for_data(log_path, data)
                
                # Add recent logs
                if log_path:
                    data['recent_logs'] = tail_log(log_path, n_lines=20)
                
                # ✅ JOUR 2: Ajouter les données de la base de données unifiée
                if db:
                    try:
                        # Récupérer les derniers trades
                        recent_trades = db.get_trades(limit=5)
                        if recent_trades:
                            data['recent_trades'] = [
                                {
                                    'side': t['action'],
                                    'qty': t['quantity'],
                                    'price': t['price'],
                                    'pnl': t['pnl'],
                                    'timestamp': t['timestamp']
                                }
                                for t in recent_trades
                            ]
                        
                        # Récupérer les dernières métriques
                        sharpe_metrics = db.get_metrics('sharpe_ratio', limit=1)
                        if sharpe_metrics:
                            data['metrics'] = {
                                'sharpe': sharpe_metrics[0]['value']
                            }
                    except Exception as e:
                        pass  # Silently fail if DB read fails
                
                live.update(generate_dashboard(data))
                time.sleep(1)
    except KeyboardInterrupt:
        console.print()
        console.print("[bold yellow]👋 Dashboard arrêté gracieusement[/]")
        console.print()
    except Exception as e:
        console.print(f"[bold red]❌ Erreur fatale: {e}[/]")
        raise


if __name__ == "__main__":
    main()
