# Source Code Directory

This directory contains the core source code for the ADAN trading bot, organized into logical modules for data processing, environment simulation, agent implementation, and common utilities.

## Important Subdirectories:

- `adan_trading_bot/`: The main Python package for the ADAN project.
  - `agent/`: Contains the implementation of the reinforcement learning agent, including the PPO agent and feature extractors.
  - `common/`: Provides common utilities, constants, and logging configurations used across the project.
  - `data_processing/`: Handles all aspects of data loading, feature engineering, and data normalization.
  - `environment/`: Defines the Gymnasium-compatible trading environment, including components for portfolio management, order handling, and reward calculation.
  - `evaluation/`: Contains modules for evaluating the performance of trained agents.
  - `exchange_api/`: Provides connectors for interacting with cryptocurrency exchanges (e.g., CCXT).
  - `live_trading/`: Modules specifically designed for live trading operations, including safety management.
  - `training/`: Contains the training loop, callbacks, and other utilities related to model training.

## Related Documentation:

- [Main README.md](../../README.md)
