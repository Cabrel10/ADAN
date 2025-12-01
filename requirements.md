# ADAN 2.0 Requirements

## 1. Data Pipeline Integrity (The Foundation)
- **Strict Separation**: Data MUST be split into Train (2021-2023) and Test (2024-Now).
- **No Leakage**: Test data must NEVER be used for training or global statistic calculation (e.g., normalization means/stds).
- **Completeness**: All indicators (ATR_20, ATR_50, ADX, etc.) must be pre-calculated.
- **Quality**: No NaNs, no infinite values, no holes in the data.
- **Verification**: Automated checks must confirm zero overlap between Train and Test sets.

## 2. Realistic Trading Environment (The Rhythm)
- **Inheritance**: `LiveTradingEnv` must inherit logic from the training environment to ensure consistency.
- **Constraints**:
    - **Minimum Notional**: Enforce Binance's minimum order size (e.g., $5 or $10).
    - **Transaction Costs**: Include realistic fees and slippage.
- **Order Management**:
    - Support for Market, Limit, and Stop-Loss orders.
    - Simulation of "Partial Fills" for larger volumes.

## 3. Observation Normalization (The Fishbowl)
- **VecNormalize**: All observations must be normalized using `VecNormalize`.
- **Persistence**: Normalization statistics (running mean/std) must be saved (`vecnormalize.pkl`) and loaded for inference/testing.
- **Raw Input**: The environment should provide raw portfolio states; normalization happens in the wrapper.
- **Stationarity**: Ensure inputs to the neural network are within a stable range (roughly -5 to +5) to prevent saturation.

## 4. Multi-Expert Fusion Pipeline
- **Dynamic Fusion**: Implement a mechanism to weight experts dynamically based on recent performance (or a meta-model).
- **Modularity**: Experts should be trainable independently before fusion.
- **State Persistence**: The fusion state must be part of the checkpoint to allow resuming training.

## 5. Stability & Safety
- **Reward Clipping**: Rewards must be clipped (e.g., [-10, 10]) to prevent gradient explosions from anomalies.
- **Reproducibility**: All random seeds (Python, NumPy, PyTorch, Gym) must be fixed via a `SeedManager`.
