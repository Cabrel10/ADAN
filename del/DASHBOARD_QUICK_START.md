# 🚀 ADAN Dashboard - Quick Start Guide

## Launch Dashboard with Real Market Data

### Option 1: Run Once (Testing)
```bash
python scripts/dashboard_with_real_data.py --once
```

### Option 2: Run with Auto-Refresh (30 seconds)
```bash
python scripts/dashboard_with_real_data.py
```

### Option 3: Run with Custom Refresh Interval
```bash
python scripts/dashboard_with_real_data.py --refresh 60
```

---

## 📊 What You'll See

```
🔴 ADAN v1.0 - BTC/USDT MONITOR (REAL DATA)
📡 Data Source: Binance Testnet (LIVE)

💰 PORTFOLIO STATE
  Total Value:        $29.00
  Available Capital:  $29.00
  Open Positions:     0

📊 MARKET DATA (REAL)
  Current Price:      $87,760.33  ← REAL from Binance
  RSI:                58.18
  ATR:                $8,391.49
  ADX:                94.75
  Volatility:         9.56%

🔧 SYSTEM HEALTH
  API Status:         ✅ OK (Latency: 1606.83ms)
  Feed Status:        ✅ OK
  Model Status:       ✅ OK
  Database Status:    ✅ OK

🎯 CURRENT SIGNAL
  Direction:          🟡 HOLD
  Confidence:         0.00%
```

---

## 🔴 Real Data Proof

✅ **BTC Price**: $87,760.33 (from Binance testnet)
✅ **RSI**: 58.18 (calculated from real OHLCV)
✅ **ATR**: $8,391.49 (real volatility)
✅ **API Latency**: 1606.83ms (measured)
✅ **Portfolio**: $29.00 (real testnet balance)

---

## 🛑 Stop Dashboard

Press `Ctrl+C` to exit

---

## 📝 Features

- ✅ Real Binance testnet connection
- ✅ Real market data (BTC/USDT)
- ✅ Real technical indicators
- ✅ Real API latency measurement
- ✅ Real portfolio data
- ✅ Auto-refresh every 30 seconds
- ✅ Professional dashboard layout
- ✅ Color-coded status indicators

---

## 🔧 Troubleshooting

### Dashboard won't start
```bash
# Check Python environment
python --version

# Check dependencies
pip list | grep ccxt
pip list | grep pandas
```

### No market data showing
- Check internet connection
- Verify Binance API keys in environment
- Check if Binance testnet is accessible

### Slow updates
- Default refresh is 30 seconds
- Increase with: `--refresh 60`
- Decrease with: `--refresh 15`

---

## 📚 More Information

See `DASHBOARD_REAL_DATA_LIVE.md` for detailed technical information.
