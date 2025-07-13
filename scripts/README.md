# Scripts Directory

This directory contains various utility scripts for data processing, model training, evaluation, and other operational tasks within the ADAN project.

## Important Files:

- `convert_real_data.py`: Processes raw market data, calculates technical indicators, and saves them in a unified format.
- `merge_processed_data.py`: Merges processed data from multiple assets and timeframes into single, comprehensive datasets for training and evaluation.
- `train_rl_agent.py`: The main script for training the reinforcement learning agent.
- `evaluate_final.py`: Evaluates the performance of a trained agent.
- `fetch_data_ccxt.py`: Script for fetching historical market data using CCXT.
- `monitor_training.py`: Provides tools for monitoring the training process.
- `process_data.py`: (Potentially deprecated or a wrapper) Handles data processing steps.
- `paper_trading_agent.py`: Script for running the agent in a simulated paper trading environment.

## Usage:

Most scripts can be run from the project root. Refer to the `Utilisation` section in the main `README.md` for quick start examples.

Example for data processing:
```bash
python scripts/convert_real_data.py --config config/data_config_cpu.yaml
python scripts/merge_processed_data.py --config config/data_config_cpu.yaml
```

Example for training:
```bash
python scripts/train_rl_agent.py --config config/main_config.yaml
```

## Related Documentation:

- [Main README.md](../../README.md)
- [Data Processing Pipeline (src/adan_trading_bot/data_processing/)](../src/adan_trading_bot/data_processing/README.md)
- [Training Module (src/adan_trading_bot/training/)](../src/adan_trading_bot/training/README.md)
