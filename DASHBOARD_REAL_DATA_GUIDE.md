# 🎯 ADAN Dashboard - Real Data Guide

## Overview

Le dashboard ADAN BTC/USDT affiche maintenant les **données réelles en temps réel** du système de trading. Plus de données mock par défaut !

## Quick Start

### Option 1: Start Everything (Recommended)
```bash
./scripts/launch_dashboard_with_monitor.sh
```

This will:
1. Start the paper trading monitor (collects real data)
2. Wait for initialization
3. Launch the dashboard (displays real data)

### Option 2: Start Dashboard Only
```bash
python scripts/adan_btc_dashboard.py
```

Requirements:
- Paper trading monitor must already be running
- State file must exist at `/mnt/new_data/t10_training/phase2_results/paper_trading_state.json`

### Option 3: Start Monitor Only
```bash
./scripts/launch_dashboard_with_monitor.sh --monitor-only
```

## Command Line Options

### Dashboard Options
```bash
python scripts/adan_btc_dashboard.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--refresh RATE` | Refresh rate in seconds | 60.0 |
| `--once` | Run once and exit (for testing) | - |
| `--mock` | Use mock data (testing only) | - |
| `--help` | Show help message | - |

### Examples

**Run with real data (default)**
```bash
python scripts/adan_btc_dashboard.py
```

**Run with 30-second refresh rate**
```bash
python scripts/adan_btc_dashboard.py --refresh 30.0
```

**Run once for testing**
```bash
python scripts/adan_btc_dashboard.py --once
```

**Use mock data for testing**
```bash
python scripts/adan_btc_dashboard.py --mock
```

**Combined options**
```bash
python scripts/adan_btc_dashboard.py --refresh 30.0 --once
```

## Data Architecture

### Real Data Flow
```
┌─────────────────────────────────────────────────────────┐
│ ADAN Trading System                                     │
│ - Exchange API (Binance Testnet)                        │
│ - Indicator Calculator (RSI, ADX, ATR)                  │
│ - Portfolio Manager                                     │
│ - Worker Ensemble (4 models)                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Paper Trading Monitor                                   │
│ scripts/paper_trading_monitor.py                        │
│ - Runs every 5 minutes (300s)                           │
│ - Collects market data                                  │
│ - Calculates indicators                                 │
│ - Generates trading signals                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│ State File (JSON)                                       │
│ /mnt/new_data/t10_training/phase2_results/              │
│ paper_trading_state.json                                │
│ - Portfolio state                                       │
│ - Market context                                        │
│ - Current signal                                        │
│ - System health                                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│ RealDataCollector                                       │
│ src/adan_trading_bot/dashboard/real_collector.py        │
│ - Reads fresh from state file                           │
│ - No caching (always current)                           │
│ - Handles missing files gracefully                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│ ADAN Dashboard                                          │
│ scripts/adan_btc_dashboard.py                           │
│ - Displays real-time data                               │
│ - Updates every 60 seconds (configurable)               │
│ - Shows portfolio, signals, market context              │
└─────────────────────────────────────────────────────────┘
```

## Dashboard Display

### Main Sections

#### 1. Header
```
🎯 ADAN v1.0 - BTC/USDT MONITOR
Portfolio: $29.00 (0.00%) │ Positions: 0 │ Win Rate: 0%
```

#### 2. Decision Matrix
```
Signal            HOLD
Confidence        0.00
Horizon           1h
Workers           [W1: HOLD, W2: HOLD, W3: HOLD, W4: HOLD]
Driver            Ensemble Consensus

Volatility        0.25%
RSI               57 (Neutral)
ADX               35 (Moderate)
Trend             Trending
Regime            Trending
```

#### 3. Market Context
```
Price             $90,206.62
ATR               $44.11 (0.049%)
Volume Change     -17.7%
```

#### 4. Portfolio Status
```
Open Positions    0
Closed Trades     0
Total P&L         $0.00 (0.00%)
```

#### 5. System Health
```
API Status        ✅ OK
Feed Status       ✅ OK
Model Status      ✅ OK
Database Status   ✅ OK
```

## Data Freshness

### Update Intervals
- **Monitor**: Updates every 5 minutes (300 seconds)
- **Dashboard**: Refreshes every 60 seconds (configurable)
- **State File**: Updated by monitor after each analysis

### Synchronization
```
Time    Monitor              Dashboard
00:00   Analyze market       Display data
00:05   Update state file    Refresh display
00:10   Analyze market       Display data
00:15   Update state file    Refresh display
...
```

## Troubleshooting

### Dashboard Shows Empty Data
**Problem**: Dashboard displays 0 values for all metrics

**Solution**:
1. Check if paper_trading_monitor is running:
   ```bash
   ps aux | grep paper_trading_monitor
   ```

2. Check if state file exists:
   ```bash
   ls -lah /mnt/new_data/t10_training/phase2_results/paper_trading_state.json
   ```

3. Start the monitor:
   ```bash
   python scripts/paper_trading_monitor.py
   ```

### Dashboard Shows Old Data
**Problem**: Dashboard displays stale data (not updating)

**Solution**:
1. Check state file modification time:
   ```bash
   stat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json
   ```

2. Restart the monitor:
   ```bash
   pkill -f paper_trading_monitor
   python scripts/paper_trading_monitor.py
   ```

### Connection Error
**Problem**: "Failed to connect to data sources"

**Solution**:
1. Verify state file directory exists:
   ```bash
   mkdir -p /mnt/new_data/t10_training/phase2_results
   ```

2. Check file permissions:
   ```bash
   chmod 755 /mnt/new_data/t10_training/phase2_results
   ```

3. Restart dashboard:
   ```bash
   python scripts/adan_btc_dashboard.py
   ```

## Performance

### Resource Usage
- **CPU**: ~5-10% (light)
- **Memory**: ~100-150 MB
- **Disk I/O**: Minimal (reads state file every refresh)

### Optimization Tips
1. **Increase refresh rate** for less frequent updates:
   ```bash
   python scripts/adan_btc_dashboard.py --refresh 120.0
   ```

2. **Decrease refresh rate** for more frequent updates:
   ```bash
   python scripts/adan_btc_dashboard.py --refresh 30.0
   ```

3. **Use mock data** for testing (faster, no I/O):
   ```bash
   python scripts/adan_btc_dashboard.py --mock
   ```

## Integration with Other Components

### Paper Trading Monitor
- **Location**: `scripts/paper_trading_monitor.py`
- **Purpose**: Collects real trading data and updates state file
- **Frequency**: Every 5 minutes
- **Output**: `/mnt/new_data/t10_training/phase2_results/paper_trading_state.json`

### Real Data Collector
- **Location**: `src/adan_trading_bot/dashboard/real_collector.py`
- **Purpose**: Reads state file and provides data to dashboard
- **Caching**: None (always fresh)
- **Error Handling**: Graceful fallback to empty state

### Dashboard App
- **Location**: `src/adan_trading_bot/dashboard/app.py`
- **Purpose**: Main dashboard application
- **Features**: Live updates, signal handling, error recovery

## Advanced Usage

### Running Multiple Dashboards
```bash
# Terminal 1: Monitor
python scripts/paper_trading_monitor.py

# Terminal 2: Dashboard 1 (60s refresh)
python scripts/adan_btc_dashboard.py --refresh 60.0

# Terminal 3: Dashboard 2 (30s refresh)
python scripts/adan_btc_dashboard.py --refresh 30.0
```

### Testing with Mock Data
```bash
# Terminal 1: Monitor (real data)
python scripts/paper_trading_monitor.py

# Terminal 2: Dashboard (mock data for testing)
python scripts/adan_btc_dashboard.py --mock
```

### Continuous Monitoring
```bash
# Run dashboard in background
nohup python scripts/adan_btc_dashboard.py > dashboard.log 2>&1 &

# Monitor the log
tail -f dashboard.log
```

## Configuration

### State File Location
Edit `src/adan_trading_bot/dashboard/real_collector.py`:
```python
state_file = Path("/mnt/new_data/t10_training/phase2_results/paper_trading_state.json")
```

### Default Refresh Rate
Edit `scripts/adan_btc_dashboard.py`:
```python
parser.add_argument(
    "--refresh",
    type=float,
    default=60.0,  # Change this value
    help="Refresh rate in seconds"
)
```

## FAQ

**Q: Can I run the dashboard without the monitor?**
A: Yes, but you'll see empty data. The monitor must be running to populate the state file.

**Q: How often is the data updated?**
A: The monitor updates every 5 minutes. The dashboard refreshes every 60 seconds (configurable).

**Q: Can I use the dashboard with mock data?**
A: Yes, use the `--mock` flag for testing purposes.

**Q: What if the state file is missing?**
A: The dashboard will show empty data but won't crash. It will recover when the file is created.

**Q: Can I change the refresh rate?**
A: Yes, use the `--refresh` option with any value in seconds.

**Q: Is the data cached?**
A: No, the RealDataCollector reads fresh from disk on every update.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs: `tail -f dashboard.log`
3. Check the state file: `cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json`
4. Verify the monitor is running: `ps aux | grep paper_trading_monitor`

---
**Last Updated**: 2025-12-14
**Version**: 1.0
**Status**: ✅ Production Ready
