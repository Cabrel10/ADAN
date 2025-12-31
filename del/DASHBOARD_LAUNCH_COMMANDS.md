# Dashboard Launch Commands

## Current Status

The dashboard is ready to be launched, but it's currently using mock data. After implementing the spec, it will display real market data.

## Launch Dashboard (Current - Mock Data)

```bash
# Launch with mock data (current state)
python scripts/adan_btc_dashboard.py --mock

# Launch with real data (after implementation)
python scripts/adan_btc_dashboard.py

# Launch with custom refresh rate
python scripts/adan_btc_dashboard.py --refresh 30.0

# Launch once and exit (for testing)
python scripts/adan_btc_dashboard.py --once
```

## What's Needed for Real Data

1. **State File**: `/mnt/new_data/t10_training/phase2_results/paper_trading_state.json`
   - Must be created by paper trading monitor
   - Contains portfolio, signal, market, and system data

2. **RealDataCollector**: Must be enhanced with:
   - Network latency measurement
   - State file reading with retry logic
   - Market data freshness tracking
   - Signal display with worker votes
   - Portfolio state validation
   - System health metrics

3. **Tests**: Must pass all property-based tests:
   - Real data connection
   - Market data freshness
   - Network metrics accuracy
   - Signal consistency
   - Portfolio state validity
   - System health bounds

## Monitoring the Dashboard

### Check if State File Exists
```bash
ls -la /mnt/new_data/t10_training/phase2_results/paper_trading_state.json
```

### Check State File Content
```bash
cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | python -m json.tool
```

### Monitor State File Updates
```bash
watch -n 1 'cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | python -m json.tool | head -50'
```

## Troubleshooting

### Dashboard Shows "Waiting for state file..."
- Check if state file exists at `/mnt/new_data/t10_training/phase2_results/paper_trading_state.json`
- Check if paper trading monitor is running
- Check file permissions

### Dashboard Shows Static Data
- Verify RealDataCollector is being used (not MockDataCollector)
- Check if state file is being updated
- Check dashboard refresh rate

### Network Metrics Show 0
- Verify network latency measurement is implemented
- Check if API and WebSocket connections are active
- Verify system health metrics are being collected

## After Implementation

Once the spec is implemented, the dashboard will:

1. **Display Real Market Data**
   - Current BTC price from Binance
   - Real RSI, ADX, volatility
   - Real market regime (bullish/bearish/ranging)

2. **Show Trading Signals**
   - Current signal (BUY/SELL/HOLD)
   - Signal confidence (0-100%)
   - Individual worker votes
   - Decision driver

3. **Display Portfolio State**
   - Total portfolio value
   - Available capital
   - Open positions with P&L
   - Last 5 closed trades

4. **Monitor System Health**
   - API latency (ms)
   - WebSocket feed lag (ms)
   - CPU usage (%)
   - Memory usage (GB)
   - System uptime (%)

5. **Update in Real-Time**
   - Refresh every 60 seconds (configurable)
   - Fresh data from state file
   - Network metrics measured on each refresh

## Next: Implementation

Ready to implement the spec? Start with Task 1 in `.kiro/specs/dashboard-real-data-fix/tasks.md`
