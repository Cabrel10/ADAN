# 🚀 ADAN Quick Start Guide

## ⚡ 30-Second Setup

```bash
# 1. Kill old process
pkill -f paper_trading_monitor.py

# 2. Start monitor
python scripts/paper_trading_monitor.py \
  --api_key "YOUR_KEY" \
  --api_secret "YOUR_SECRET" &

# 3. Watch logs
tail -f paper_trading.log
```

---

## 📋 Verification Commands

```bash
# Check all systems
python3 scripts/verify_data_pipeline.py

# Check indicators
python3 scripts/debug_indicators.py

# Check trading logic
python3 scripts/test_trade_execution.py

# Check models
python3 scripts/verify_cnn_ppo.py
```

---

## 🎯 What to Look For in Logs

### ✅ Good Signs
```
✅ Configuration loaded
✅ Models loaded (W1-W4)
✅ Normalizer initialized
✅ System ready
🔍 Market analysis
📊 Data fetched
🧠 Ensemble prediction
💸 Trade executed
```

### ❌ Bad Signs
```
❌ Config not found
❌ Models not found
❌ INCOMPATIBILITÉ
❌ Error
```

---

## 💰 Capital Tiers

| Tier | Range | Max Position | Max Trades |
|------|-------|--------------|-----------|
| Micro | $11-30 | 90% | 1 |
| Small | $30-100 | 65% | 2 |
| Medium | $100+ | 50% | 3 |

**Your Capital: $29.00 = Micro Tier**

---

## 📊 Trading Parameters

| Worker | TP | SL | Position |
|--------|----|----|----------|
| W1 | 3.21% | 2.53% | 11.21% |
| W2 | 5.00% | 2.50% | 25.00% |
| W3 | 18.00% | 10.00% | 50.00% |
| W4 | 2.00% | 1.20% | 20.00% |

---

## 🔄 Trading Cycle

1. **Every 5 minutes**: Analyze market
2. **If signal**: Execute trade with TP/SL
3. **Every 30 seconds**: Check TP/SL
4. **If TP/SL hit**: Close position
5. **Repeat**

---

## 🛑 Stop Monitor

```bash
pkill -f paper_trading_monitor.py
```

---

## 📁 Key Files

- `config/config.yaml` - Configuration
- `scripts/paper_trading_monitor.py` - Main monitor
- `paper_trading.log` - Logs
- `/mnt/new_data/t10_training/checkpoints/final/` - Models

---

## ✅ Status Check

```bash
# Is monitor running?
ps aux | grep paper_trading_monitor | grep -v grep

# Any errors?
grep "Error\|Exception" paper_trading.log | tail -5

# Any trades?
grep "Trade Exécuté" paper_trading.log | tail -5

# Compatibility OK?
grep "INCOMPATIBILITÉ" paper_trading.log
```

---

**Status: ✅ READY**
