# ADAN Trading Bot - Colab Quick Setup

## 🚀 Quick Start (5 minutes)

### Step 1: Clone the repository
```bash
cd /content
git clone https://github.com/Cabrel10/ADAN0.git bot
cd bot
```

### Step 2: Install dependencies
```bash
pip install -q -r requirements-colab.txt
```

### Step 3: Run training
```bash
python scripts/train_parallel_agents.py \
  --config-path config/config.yaml \
  --checkpoint-dir checkpoints \
  --resume
```

## 📊 Configuration

- **Workers**: 4 parallel (DummyVecEnv - no pickle issues)
- **Timesteps**: 500,000 per instance
- **Batch size**: 64
- **Learning rate**: 0.0003 (adaptive)
- **GPU**: Automatically detected and optimized

## 🎯 Expected Results

- **Training time**: 6-8 hours
- **Sharpe ratio**: > 2.0 after 300k steps
- **Max drawdown**: < 20%
- **Win rate**: > 50%

## 📁 Key Files

- `scripts/train_parallel_agents.py` - Main training script
- `scripts/optimize_hyperparams.py` - Hyperparameter optimization
- `config/config.yaml` - Configuration file
- `src/adan_trading_bot/` - Source code

## ⚠️ Important Notes

1. **No pickle errors**: Using DummyVecEnv everywhere (Colab + Local)
2. **4 workers always**: Consistent parallelism across environments
3. **GPU optimized**: TF32 enabled, cache cleared
4. **Clean codebase**: Only essential scripts included

## 🔧 Troubleshooting

### Import errors
```bash
# Verify imports
python -c "from adan_trading_bot.common.config_loader import ConfigLoader; print('OK')"
```

### Memory issues
- Reduce `timesteps_per_instance` in config.yaml
- Use smaller `batch_size`

### Slow training
- Check GPU usage: `nvidia-smi`
- Verify 4 workers are running
- Check logs in `logs/` directory

## 📞 Support

All critical parameters are in `config/config.yaml`. Modify with caution!
