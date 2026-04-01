# ADAN -- Autonomous Digital Asset Navigator (SOTA 2026)

**Version**: 2.0.0 | **Status**: Production-Ready | **Last updated**: April 2026

## 1. Overview

ADAN is a **state-of-the-art multi-agent crypto-trading system** built on
Proximal Policy Optimization (PPO) with a Cross-Attention Hierarchical
architecture.  Key innovations:

| Component | Description |
|-----------|-------------|
| **ContextualTemporalFusionExtractor** | Cross-Attention over 3 timeframes (5 m, 1 h, 4 h) with residual blocks |
| **FiLM Meta-RL** | Feature-wise Linear Modulation conditions policy layers on a 12-dim context vector |
| **Time2Vec** | Learnable cyclical embeddings (sin/cos of hour, weekday, day-of-month) injected into context |
| **Hidden Markov Model (HMM)** | Regime detection (4 states: Bull, Bear, Chop, Breakout) feeds the context vector |
| **Symlog Reward** | `sign(r) * log(1 + |r|)` compresses extreme PnL signals for stable training |
| **Capital Tier Supremacy** | Five tiers (Micro -> Enterprise) dynamically control exposure (90 % -> 15 %) and risk |
| **Anti-Spam Hysteresis (OMEGA-4E)** | Bidirectional exposure threshold prevents order-spam near boundaries |
| **Population-Based Training (PBT)** | Ray Tune auto-evolves lr, entropy, gamma across 4 worker profiles |

## 2. Architecture

```
                        +------------------+
                        |  12-dim Context  |
                        | (6 Market + 6    |
                        |  Time2Vec/HMM)   |
                        +--------+---------+
                                 |
                          FiLM modulation
                                 v
  5m branch ---> [Conv1D + ResBlock] --+
  1h branch ---> [Conv1D + ResBlock] --+--> Cross-Attention --> MLP Policy --> Box(25,)
  4h branch ---> [Conv1D + ResBlock] --+                         |
  portfolio  ---> [Linear]             |                    Symlog Reward
                                       |
                              VecNormalize (clip_obs=10)
```

**Context vector (12 dims)**:
`[Volatility, Trend, ADX, Regime, Drawdown, Candle_Progress, sin_hour, cos_hour, sin_weekday, cos_weekday, sin_dom, cos_dom]`

## 3. Risk Management -- Capital Tiers

Capital Tiers are the **sole authority** for exposure and risk-per-trade.
Workers carry only timeframe and trade-duration parameters.

| Tier | Balance Range | Exposure | Risk/Trade |
|------|--------------|----------|------------|
| **Micro Capital** | $11 -- $30 | 70 -- 90 % | 4.0 % |
| **Small Capital** | $30 -- $100 | 35 -- 75 % | 2.0 % |
| **Medium Capital** | $100 -- $500 | 45 -- 60 % | 2.25 % |
| **High Capital** | $500 -- $2 000 | 20 -- 35 % | 2.75 % |
| **Enterprise** | $2 000+ | 5 -- 15 % | 3.0 % |

Dynamic exposure scales from 90 % (Micro) down to 15 % (Enterprise) with
bidirectional hysteresis preventing order-spam at tier boundaries.

## 4. Worker Profiles (OMEGA)

| Profile | Worker | Timeframe | n_steps | batch |
|---------|--------|-----------|---------|-------|
| **Scalper** | w1 | 5 m | 512 | 64 |
| **Intraday** | w2 | 1 h | 512 | 64 |
| **Swing** | w3 | 4 h | 512 | 64 |
| **Position** | w4 | 4 h | 1024 | 128 |

## 5. Project Structure

```
ADAN/
+-- config/                    # Central configuration
|   +-- config.yaml           # Master config (Capital Tiers, agent, data)
|   +-- workers.yaml          # Worker timeframe/duration settings
|   +-- trading.yaml          # Trading rules
|   +-- feature_extractor_config.yaml
+-- src/adan_trading_bot/      # Core source
|   +-- agent/                # Feature extractors (FiLM, Time2Vec, HMM)
|   +-- environment/          # MultiAssetChunkedEnv, StateBuilder
|   +-- data_processing/      # ChunkedDataLoader, DataValidator
|   +-- portfolio/            # PortfolioManager
|   +-- risk_management/      # DBE (Dynamic Boundary Enforcement)
|   +-- common/               # ConfigLoader, utilities
+-- scripts/                   # Executable scripts
|   +-- train_parallel_agents.py   # Production PBT training (Ray Tune)
|   +-- paper_trading_monitor.py   # Live paper trading (VecNormalize inference)
|   +-- backtest_engine.py         # Backtest with ensemble support
|   +-- generate_colab_dataset.py  # Master Clock dataset generator
+-- tests/                     # pytest suite
+-- models/                    # Saved models & VecNormalize stats
|   +-- rl_agents/
|       +-- ppo_adan_simple.zip
|       +-- vecnormalize.pkl  # CRITICAL for inference
+-- data/processed/indicators/ # Parquet datasets (train/test/val)
+-- results/                   # Backtest & paper-trading reports
+-- requirements.txt
+-- setup.py / pyproject.toml
```

## 6. Quickstart

### 6.1 Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### 6.2 Generate aligned dataset

```bash
# Lightweight (Colab-friendly)
ADAN_CANDLES=2000 python scripts/generate_colab_dataset.py

# Full dataset
ADAN_CANDLES=50000 python scripts/generate_colab_dataset.py --no-testnet
```

The generator uses the **5 m Master Clock**: higher timeframes (1 h, 4 h) are
resampled and forward-filled onto the 5 m `DatetimeIndex`, eliminating the
time-travel data alignment bug.

### 6.3 Training (PBT -- 8-core)

```bash
python scripts/train_parallel_agents.py \
    --config config/config.yaml \
    --profiles scalper intraday swing position \
    --steps 2000000 \
    --num-cpus 8 \
    --num-samples 4
```

> **Note**: `SubprocVecEnv` is disabled by default (`--no-subproc`) to avoid
> Ray/fork conflicts.  Pass `--use-subproc` only on systems without Ray.

### 6.4 Backtest

```bash
# Single model
python scripts/backtest_engine.py --model models/rl_agents/ppo_adan_simple.zip --steps 2000

# Ensemble (w1-w4)
python scripts/backtest_engine.py --ensemble --steps 5000
```

### 6.5 Paper Trading (Binance Testnet)

```bash
python scripts/paper_trading_monitor.py --config config/config.yaml --duration 360
```

The monitor now correctly loads `VecNormalize.load("models/rl_agents/vecnormalize.pkl")`
with `training=False` and `norm_reward=False`, ensuring the observation
distribution matches training exactly.

### 6.6 Google Colab

Upload `data/processed/indicators/` to Colab, install deps, and run:

```python
from scripts.generate_colab_dataset import generate_dataset
generate_dataset(output_dir="data/processed/indicators", candles=5000)
```

## 7. Key Bug Fixes (April 2026)

| Bug | Fix |
|-----|-----|
| **Time-travel data alignment** | 5 m Master Clock; 1 h/4 h reindexed with ffill |
| **VecNormalize inference** | `VecNormalize.load()` + `training=False` in paper trading & backtest |
| **ConfigLoader env-var crash** | `_resolve_part` returns placeholder instead of raising `ValueError` |
| **Ray object_store_memory** | Removed obsolete `_system_config` from `ray.init()` |
| **Ray/SubprocVecEnv fork** | Default `use_subproc=False`; `DummyVecEnv` avoids deadlocks |
| **Timezone-aware comparison** | `_tz_naive()` applied in `DataValidator` |
| **Hard-coded worker risk** | Workers cleaned; Capital Tiers are sole risk authority |

## 8. Governance Rules

1. **Capital Tiers** are the sole source of risk parameters -- never override in worker configs.
2. **VecNormalize**: always load training stats (`vecnormalize.pkl`) for inference.
3. **Master Clock**: all multi-timeframe data must be reindexed onto the 5 m grid.
4. **`train_simple_ppo.py`** is a disposable sandbox script; production uses `train_parallel_agents.py`.

## 9. Testing

```bash
# Full test suite
pytest tests/ -v

# Quick environment smoke-test
python -c "
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader
cfg = ConfigLoader.load_config('config/config.yaml')
dl = ChunkedDataLoader(cfg)
data = dl.load_all_data()
env = MultiAssetChunkedEnv(data=data, config=cfg, worker_config=cfg['workers']['w1'])
obs, _ = env.reset()
print('context_vector shape:', obs['context_vector'].shape)
"
```

---

**Maintained by**: ADAN AI Developer | **License**: Proprietary
