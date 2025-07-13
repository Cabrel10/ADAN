# Agent Module

This module contains the implementation of the reinforcement learning agent, including the PPO agent and feature extractors.

## Important Files:

- `ppo_agent.py`: Defines the Proximal Policy Optimization (PPO) agent, including its policy network and learning algorithm.
- `feature_extractors.py`: Implements custom feature extractors, such as Convolutional Neural Networks (CNNs), used to process raw observations into meaningful features for the agent.

## Usage:

The PPO agent defined here is used by the `trainer.py` script for training. Feature extractors are integrated into the agent's policy network.

## Related Documentation:

- [Main README.md](../../../../README.md)
- [Training Module (src/adan_trading_bot/training/)](../training/README.md)