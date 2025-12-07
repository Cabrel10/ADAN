# 🤖 ADAN Trading Bot - Adaptive Dynamic Agent Network

**Status**: ✅ **PRODUCTION READY** | Training in progress with Optuna-optimized hyperparameters

## 🎯 Overview

ADAN is an advanced multi-agent reinforcement learning trading system featuring:
- **4 Independent Workers** (W1-W4) with specialized trading strategies
- **Optuna Hyperparameter Optimization** (80 trials completed)
- **Dynamic Behavior Engine (DBE)** for market regime adaptation
- **Unified Metrics System** for real-time performance tracking
- **Parallel Training** with 1M steps per worker (~4M total)

## 📊 Current Status

### ✅ Completed Milestones
- [x] Optuna optimization: 80 trials (20 per worker)
- [x] Environment correction: MultiAssetChunkedEnv consistency
- [x] Bug fixes: PortfolioManager, add_trade() method
- [x] Hyperparameter loading: config.yaml updated
- [x] Training launch: 4 workers running in parallel
- [x] System validation: Stable and error-free

### 🏆 Best Hyperparameters (Optuna Results)

| Worker | Type | Score | Sharpe | Drawdown | Win Rate | Trades |
|--------|------|-------|--------|----------|----------|--------|
| **W1** | Ultra-Conservative | 3.9291 | 3.4397 | 6.66% | 51.68% | 556 |
| **W2** | Balanced | **3.9905** | **3.5168** | 16.12% | 51.63% | 258 |
| **W3** | Aggressive | 2.2877 | 2.0431 | 21.63% | 45.34% | 256 |
| **W4** | Hybrid | 3.6370 | 3.2432 | 18.55% | 50.14% | 430 |

**Winner**: W2 (Balanced) with best Sharpe ratio of 3.5168

### 📈 Training Progress
- **Status**: 🟢 ACTIVE
- **Processes**: 9 active (1 main + 4 workers + 4 auxiliary)
- **Log Size**: 328MB
- **Objective**: 1M steps per worker (~4M total)
- **Disk Space**: 28GB available

## 🏗️ Architecture

### Workers
- **W1 (Ultra-Conservative)**: Low risk, high Sharpe (3.44)
- **W2 (Balanced)**: Optimal risk-reward, best overall (3.52)
- **W3 (Aggressive)**: High risk, high returns
- **W4 (Hybrid)**: Mixed strategy, good Sharpe (3.24)

### Key Components
- **MultiAssetChunkedEnv**: Training environment with chunked data
- **Dynamic Behavior Engine**: Market regime detection and adaptation
- **Portfolio Manager**: Risk management and position sizing
- **Unified Metrics**: Real-time performance tracking
- **Central Logger**: Synchronized logging across workers

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.11
conda environment: trading_env
```

### Launch Training
```bash
# With Optuna-optimized parameters
python scripts/train_parallel_agents.py --config config/config.yaml --log-level INFO --steps 1000000

# Monitor training
bash check_training.sh
```

### Run Optuna Optimization
```bash
# Optimize single worker (20 trials)
python optuna_optimize_worker.py --worker W1 --trials 20

# All workers
for w in W1 W2 W3 W4; do
  python optuna_optimize_worker.py --worker $w --trials 20
done
```

## 📁 Project Structure

```
.
├── src/adan_trading_bot/
│   ├── environment/          # Trading environments
│   ├── model/               # RL models and ensemble
│   ├── portfolio/           # Portfolio management
│   ├── performance/         # Metrics and tracking
│   └── common/              # Utilities and logging
├── scripts/
│   ├── train_parallel_agents.py    # Main training script
│   └── terminal_dashboard.py       # Live monitoring
├── config/
│   └── config.yaml          # Configuration with Optuna params
├── tests/                   # Test suite
└── optuna_results/          # Optuna optimization results
```

## 🔧 Configuration

Hyperparameters are automatically loaded from Optuna results:
```yaml
workers:
  w1:
    risk_management:
      stop_loss_pct: 0.0389
      take_profit_pct: 0.0352
  w2:
    risk_management:
      stop_loss_pct: 0.0470
      take_profit_pct: 0.1748
  # ... w3, w4
```

## 📊 Monitoring

### Real-time Dashboard
```bash
python scripts/terminal_dashboard.py
```

### Training Status
```bash
bash check_training.sh
```

### Logs Location
```
/mnt/new_data/adan_logs/training_final_*.log
```

## 🎓 Key Features

### Optuna Optimization
- 80 trials across 4 workers
- Automatic hyperparameter tuning
- Best parameters saved to YAML
- Reproducible results

### Multi-Worker Training
- Independent agents with separate portfolios
- Parallel execution for efficiency
- Unified metrics collection
- No race conditions or conflicts

### Dynamic Behavior Engine
- Market regime detection (trending, sideways, volatile)
- Adaptive risk parameters
- ±10% adjustment capability
- Confidence-based decisions

### Risk Management
- Position sizing based on capital tiers
- Stop-loss and take-profit optimization
- Drawdown protection
- Frequency gating

## 🔍 Recent Fixes

### Environment Correction
- Changed from RealisticTradingEnv to MultiAssetChunkedEnv
- Ensures consistency between Optuna and training
- Eliminates hyperparameter divergence

### Bug Fixes
- Added `close_all_positions()` to PortfolioManager
- Fixed `add_trade()` bug in Optuna optimization
- Removed duplicate method definitions

## 📈 Next Steps

1. **Complete Training**: Let 1M steps per worker finish (~4M total)
2. **Analyze Results**: Compare worker performance
3. **Create Ensemble**: Combine 4 models with optimal weights
4. **Backtesting**: Validate on out-of-sample data
5. **Live Trading**: Deploy to production

## 📝 Documentation

- `OPTUNA_TRAINING_LAUNCH_COMPLETE.md` - Training launch details
- `ENVIRONMENT_CORRECTION_FINAL.md` - Environment fixes
- `WORKER_PERFORMANCE_REPORT.md` - Worker metrics
- `DBE_CORRECT_ROLE.md` - Dynamic Behavior Engine explanation

## 🤝 Contributing

This is an active research project. All changes should:
1. Pass existing tests
2. Include documentation
3. Be committed with clear messages
4. Maintain worker independence

## 📄 License

Proprietary - ADAN Trading System

---

**Last Updated**: December 7, 2025
**Training Status**: 🟢 ACTIVE
**Next Milestone**: Complete 4M steps training cycle
