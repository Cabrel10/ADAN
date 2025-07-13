# Models Directory

This directory stores all trained machine learning models, including reinforcement learning agents and feature encoders. It is organized to keep different types of models separate and easily accessible.

## Important Subdirectories:

- `baselines/`: Contains baseline models or pre-trained models for comparison.
- `encoders/`: Stores trained feature encoders, which might be used for dimensionality reduction or feature transformation.
- `rl_agents/`: Contains the trained reinforcement learning agents (e.g., PPO models) saved during or after the training process.

## Important Files:

- `final_model.zip`: The final trained model after the completion of a training run.
- `best_model/`: Directory containing the best performing model saved during training, typically based on evaluation metrics.
- `checkpoints/`: Contains periodic checkpoints of the trained models, allowing for training resumption or analysis of training progress.

## Usage:

Models from this directory are loaded for evaluation, backtesting, paper trading, or live trading. The `scripts/train_rl_agent.py` script saves models here, and other scripts (e.g., `scripts/evaluate_final.py`) load them.

## Related Documentation:

- [Main README.md](../../README.md)
- [Training Module (src/adan_trading_bot/training/)](../src/adan_trading_bot/training/README.md)