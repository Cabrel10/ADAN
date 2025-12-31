# ADAN BTC/USDT Terminal Dashboard - User Guide

## Overview

The ADAN BTC/USDT Terminal Dashboard is a real-time monitoring interface for the ADAN trading bot. It provides comprehensive visibility into trading positions, market signals, performance metrics, and system health.

## Quick Start

### Installation

```bash
# Install dependencies
pip install rich psutil

# Run the dashboard
python scripts/adan_btc_dashboard.py
```

### Basic Usage

```bash
# Run with default settings (mock data)
python scripts/adan_btc_dashboard.py

# Run with real data collector
python scripts/adan_btc_dashboard.py --collector real

# Run with custom refresh rate (milliseconds)
python scripts/adan_btc_dashboard.py --refresh 500

# Run with specific data source
python scripts/adan_btc_dashboard.py --source binance
```

## Dashboard Layout

The dashboard is organized into six main sections:

### 1. Header Section
Displays portfolio overview:
- **Total Portfolio Value**: Current total value in USDT
- **Available Capital**: Capital not engaged in positions
- **Position Count**: Number of open positions
- **Total P&L**: Realized + Unrealized profit/loss
- **Win Rate**: Percentage of winning trades
- **Runtime**: How long the bot has been running

### 2. Decision Matrix Section
Shows ADAN's current trading signal:
- **Signal**: BUY / SELL / HOLD
- **Confidence**: Signal confidence (0.0-1.0)
- **Horizon**: Trading timeframe (5m/1h/4h/1d)
- **Worker Votes**: Individual worker confidence scores
- **Decision Driver**: What triggered the signal (Trend/MeanReversion/Breakout)
- **Market Metrics**: Current volatility, RSI, ADX, latency, slippage

### 3. Active Positions Section
Lists all open positions:
- **Pair**: Trading pair (BTCUSDT)
- **Side**: LONG or SHORT
- **Size**: Position size in BTC
- **Entry/Current**: Entry and current prices
- **P&L**: Unrealized profit/loss in USD and %
- **Duration**: How long position has been open
- **SL/TP**: Stop Loss and Take Profit distances

### 4. Closed Trades Section
Shows last 5 closed trades in reverse chronological order:
- **Outcome**: ✅ (Win) / ❌ (Loss) / ⚠️ (Breakeven)
- **Duration**: How long trade was open
- **Size**: Position size in BTC
- **Entry/Exit**: Entry and exit prices
- **P&L**: Realized profit/loss
- **Reason**: Why trade was closed (TP/SL/Manual/Time)
- **Confidence**: Signal confidence at entry

### 5. Performance Analytics Section
Displays trading statistics:
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss
- **Trade Count**: Total number of closed trades
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline

### 6. System Health Section
Shows system status:
- **Component Status**: API, Feed, Model, Database (✅/❌)
- **System Resources**: CPU %, Memory %, Thread count, Uptime
- **Alerts**: Any system alerts with severity levels

## Color Coding

The dashboard uses professional color coding for quick visual feedback:

### P&L Colors
- 🟢 **Green**: Profitable (>2% gain)
- 🟡 **Yellow**: Small profit (0-2% gain)
- ⚪ **White**: Breakeven (-0.5% to 0%)
- 🟠 **Orange**: Small loss (-2% to -0.5%)
- 🔴 **Red**: Large loss (<-2%)

### Confidence Colors
- 🟢 **Green**: Very High (>0.8)
- 🟡 **Yellow**: High (0.6-0.8)
- ⚪ **White**: Moderate (0.4-0.6)
- 🟠 **Orange**: Low (0.2-0.4)
- 🔴 **Red**: Very Low (<0.2)

### Risk Colors
- 🟢 **Green**: Low (<1%)
- 🟡 **Yellow**: Medium (1-2%)
- 🟠 **Orange**: High (2-5%)
- 🔴 **Red**: Critical (>5%)

### Status Colors
- 🟢 **Green**: OK / Healthy
- 🟡 **Yellow**: Warning
- 🔴 **Red**: Error / Critical

## Command-Line Arguments

```bash
python scripts/adan_btc_dashboard.py [OPTIONS]

Options:
  --collector {mock|real}     Data collector type (default: mock)
  --refresh MILLISECONDS      Refresh interval in ms (default: 500)
  --source {binance|...}      Data source (default: binance)
  --log-level {DEBUG|INFO|WARNING|ERROR}  Logging level (default: INFO)
  --output FILE               Log output file (default: dashboard.log)
  --help                      Show help message
```

## Configuration

### Environment Variables

```bash
# Data source configuration
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"

# Dashboard configuration
export DASHBOARD_REFRESH_MS=500
export DASHBOARD_LOG_LEVEL=INFO
```

### Configuration File

Create `config/dashboard.yaml`:

```yaml
dashboard:
  refresh_interval_ms: 500
  max_positions_display: 10
  max_trades_display: 5
  
data_collector:
  type: real
  source: binance
  timeout_seconds: 10
  
logging:
  level: INFO
  file: logs/dashboard.log
  max_size_mb: 100
  backup_count: 5
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit dashboard |
| `r` | Refresh data immediately |
| `c` | Clear alerts |
| `l` | Toggle log view |
| `h` | Show help |

## Data Freshness

The dashboard monitors data freshness:
- **Green**: Data is fresh (<30 seconds old)
- **Yellow**: Data is stale (30-90 seconds old)
- **Red**: Data is very stale (>90 seconds old)

Stale data is indicated with a ⏱️ icon next to the timestamp.

## Performance Targets

The dashboard is optimized for performance:
- **Header render**: <20ms
- **Decision matrix render**: <30ms
- **Positions render**: <25ms
- **Closed trades render**: <25ms
- **Performance render**: <20ms
- **System health render**: <20ms
- **Full dashboard generation**: <100ms
- **Refresh cycle**: <500ms

## Troubleshooting

### Dashboard not updating
1. Check data collector connection: `--collector real`
2. Verify API credentials are set
3. Check network connectivity
4. Review logs: `tail -f logs/dashboard.log`

### High CPU usage
1. Increase refresh interval: `--refresh 1000`
2. Reduce number of displayed positions
3. Check for data collector issues

### Memory usage growing
1. Restart dashboard periodically
2. Check for data collector memory leaks
3. Reduce historical data retention

### Colors not displaying correctly
1. Ensure terminal supports 256 colors
2. Check terminal color settings
3. Try different terminal emulator

## Advanced Usage

### Custom Data Collector

Create a custom data collector:

```python
from src.adan_trading_bot.dashboard.data_collector import DataCollector

class CustomDataCollector(DataCollector):
    def connect(self):
        # Your connection logic
        pass
    
    def get_portfolio_state(self):
        # Return PortfolioState
        pass
    
    # Implement other required methods
```

### Extending the Dashboard

Add custom sections:

```python
from src.adan_trading_bot.dashboard.sections import render_header

def render_custom_section(data):
    # Your rendering logic
    return Panel(...)
```

### Integration with Monitoring Systems

Export metrics to monitoring systems:

```bash
# Prometheus metrics
python scripts/export_prometheus_metrics.py

# InfluxDB metrics
python scripts/export_influxdb_metrics.py
```

## Performance Optimization

### Reduce Refresh Rate
```bash
python scripts/adan_btc_dashboard.py --refresh 1000  # 1 second
```

### Limit Data Display
```bash
# In config/dashboard.yaml
dashboard:
  max_positions_display: 5
  max_trades_display: 3
```

### Use Mock Data for Testing
```bash
python scripts/adan_btc_dashboard.py --collector mock
```

## Logging

Dashboard logs are stored in `logs/dashboard.log`:

```bash
# View recent logs
tail -f logs/dashboard.log

# Search for errors
grep ERROR logs/dashboard.log

# Export logs
cp logs/dashboard.log logs/dashboard_backup_$(date +%Y%m%d).log
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `logs/dashboard.log`
3. Check system health section for alerts
4. Contact support with logs and configuration

## Version History

### v1.0.0 (Current)
- Initial release
- Real-time position monitoring
- Trading signal display
- Performance analytics
- System health monitoring
- Professional color coding
- Optimized rendering (<100ms)

## License

ADAN BTC/USDT Terminal Dashboard is part of the ADAN Trading Bot system.
