# Quick Commands Reference

## Monitor Management

### Start Monitor
```bash
nohup python scripts/paper_trading_monitor.py \
  --api_key "YOUR_API_KEY" \
  --api_secret "YOUR_API_SECRET" \
  > paper_trading.log 2>&1 &
```

### Stop Monitor
```bash
pkill -f paper_trading_monitor.py
```

### View Monitor Logs (Real-time)
```bash
tail -f paper_trading.log
```

### View Recent Trades
```bash
tail -100 paper_trading.log | grep "Trade Exécuté"
```

### View Signals
```bash
tail -100 paper_trading.log | grep "Ensemble:"
```

### View TP/SL Events
```bash
tail -100 paper_trading.log | grep -E "TP atteint|SL atteint"
```

## Dashboard Management

### Start Dashboard (Real-time)
```bash
python scripts/adan_btc_dashboard.py --real --refresh 60.0
```

### Start Dashboard (Mock Data)
```bash
python scripts/adan_btc_dashboard.py --mock --refresh 2.0
```

### Start Dashboard (Background)
```bash
nohup python scripts/adan_btc_dashboard.py --real --refresh 60.0 > dashboard.log 2>&1 &
```

### Stop Dashboard
```bash
pkill -f adan_btc_dashboard.py
```

## State File Management

### View Current State
```bash
cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .
```

### View Portfolio Only
```bash
cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .portfolio
```

### View Current Signal
```bash
cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .signal
```

### View Market Data
```bash
cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .market
```

### View Active Positions
```bash
cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .portfolio.positions
```

### Watch State File (Updates Every 10s)
```bash
watch -n 1 'cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .'
```

## System Monitoring

### Check Monitor Status
```bash
ps aux | grep paper_trading_monitor | grep -v grep
```

### Check Dashboard Status
```bash
ps aux | grep adan_btc_dashboard | grep -v grep
```

### Check Both
```bash
ps aux | grep -E "paper_trading_monitor|adan_btc_dashboard" | grep -v grep
```

### Kill All ADAN Processes
```bash
pkill -9 -f "paper_trading_monitor.py|adan_btc_dashboard.py"
```

## Diagnostics

### Test Monitor Connection
```bash
python scripts/verify_data_pipeline.py
```

### Test Dashboard Connection
```bash
python scripts/adan_btc_dashboard.py --real --once
```

### Check Models
```bash
ls -lh /mnt/new_data/t10_training/checkpoints/final/*.zip
```

### Check Config
```bash
cat config/config.yaml | head -20
```

## Performance Monitoring

### Monitor CPU/Memory Usage
```bash
watch -n 1 'ps aux | grep -E "paper_trading_monitor|adan_btc_dashboard" | grep -v grep'
```

### View Monitor Startup Time
```bash
head -20 paper_trading.log | grep "INFO"
```

### Count Trades Executed
```bash
grep -c "Trade Exécuté" paper_trading.log
```

### Count Signals Generated
```bash
grep -c "Ensemble:" paper_trading.log
```

### View Latest 10 Signals
```bash
grep "Ensemble:" paper_trading.log | tail -10
```

## Troubleshooting

### Monitor Not Starting
```bash
# Check for errors
python scripts/paper_trading_monitor.py --api_key "KEY" --api_secret "SECRET"

# Check logs
tail -50 paper_trading.log
```

### Dashboard Not Showing Data
```bash
# Check state file exists
ls -lh /mnt/new_data/t10_training/phase2_results/paper_trading_state.json

# Check state file content
cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json

# Check dashboard logs
tail -50 dashboard.log
```

### No Signals Generated
```bash
# Check if analysis interval has passed (5 minutes)
tail -50 paper_trading.log | grep "ANALYSE"

# Check if position is open (blocks analysis)
cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .portfolio.positions
```

### Position Not Showing in Dashboard
```bash
# Check state file has positions
cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .portfolio.positions

# Check dashboard is reading fresh data
tail -20 dashboard.log | grep "Loaded state"
```

## Development

### Run Tests
```bash
python -m pytest tests/ -v
```

### Check Code Quality
```bash
python scripts/debug_indicators.py
python scripts/verify_data_pipeline.py
python scripts/verify_cnn_ppo.py
```

### View Training Config
```bash
cat /mnt/new_data/t10_training/phase2_results/paper_trading_config.json | jq .
```

### View Ensemble Config
```bash
cat /mnt/new_data/t10_training/phase2_results/adan_ensemble_config.json | jq .
```

## Useful Aliases

Add these to your `.bashrc` or `.zshrc`:

```bash
# Monitor commands
alias adan-monitor-start='nohup python scripts/paper_trading_monitor.py --api_key "YOUR_KEY" --api_secret "YOUR_SECRET" > paper_trading.log 2>&1 &'
alias adan-monitor-stop='pkill -f paper_trading_monitor.py'
alias adan-monitor-logs='tail -f paper_trading.log'
alias adan-monitor-status='ps aux | grep paper_trading_monitor | grep -v grep'

# Dashboard commands
alias adan-dashboard-start='nohup python scripts/adan_btc_dashboard.py --real --refresh 60.0 > dashboard.log 2>&1 &'
alias adan-dashboard-stop='pkill -f adan_btc_dashboard.py'
alias adan-dashboard-logs='tail -f dashboard.log'
alias adan-dashboard-status='ps aux | grep adan_btc_dashboard | grep -v grep'

# State file commands
alias adan-state='cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .'
alias adan-positions='cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .portfolio.positions'
alias adan-signal='cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .signal'
alias adan-market='cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .market'

# System commands
alias adan-status='ps aux | grep -E "paper_trading_monitor|adan_btc_dashboard" | grep -v grep'
alias adan-stop-all='pkill -9 -f "paper_trading_monitor.py|adan_btc_dashboard.py"'
```

## Key Metrics

### Monitor Performance
- Loop cycle: 10 seconds
- Analysis interval: 300 seconds (5 minutes)
- TP/SL check: 30 seconds
- State save: Every 10 seconds

### Dashboard Performance
- Refresh rate: 60 seconds
- Data source: JSON file (fresh read)
- Update latency: <1 second

### System Limits
- Capital: $29.00 (Micro tier)
- Max positions: 1 (single asset)
- Position size: 0.0003 BTC
- TP: 3% (fixed)
- SL: 2% (fixed)

## Important Files

| File | Purpose |
|------|---------|
| `scripts/paper_trading_monitor.py` | Main trading monitor |
| `scripts/adan_btc_dashboard.py` | Dashboard entry point |
| `src/adan_trading_bot/dashboard/real_collector.py` | Real data collector |
| `src/adan_trading_bot/dashboard/app.py` | Dashboard app |
| `/mnt/new_data/t10_training/phase2_results/paper_trading_state.json` | Shared state file |
| `paper_trading.log` | Monitor logs |
| `dashboard.log` | Dashboard logs |
| `config/config.yaml` | System configuration |

## Support

For issues or questions:
1. Check logs: `tail -f paper_trading.log`
2. Check state: `cat /mnt/new_data/t10_training/phase2_results/paper_trading_state.json | jq .`
3. Check status: `ps aux | grep -E "paper_trading_monitor|adan_btc_dashboard"`
4. Run diagnostics: `python scripts/verify_data_pipeline.py`
