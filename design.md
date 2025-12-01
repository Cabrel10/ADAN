# ADAN 2.0 Design Document

## Architecture Overview

The ADAN 2.0 architecture is designed to eliminate "cheating" (data leakage), ensure realistic simulation, and provide robust normalization.

### Core Components

1.  **SeedManager**
    *   **Responsibility**: Initialize all random number generators (RNGs) at the very beginning of execution.
    *   **Scope**: Python `random`, `numpy.random`, `torch.manual_seed`, `gym.utils.seeding`.

2.  **DataManager (Enhanced)**
    *   **Responsibility**: Load and serve strictly separated data.
    *   **Logic**:
        *   `load_data(mode='train')` -> Loads 2021-2023 parquet.
        *   `load_data(mode='test')` -> Loads 2024-Now parquet.
    *   **Validation**: Asserts no overlap in timestamps.

3.  **RealisticTradingEnv (inherits from BaseEnv)**
    *   **Responsibility**: Simulate the market mechanics faithfully.
    *   **Sub-components**:
        *   **OrderManager**: Handles order validation (min notional), execution simulation (slippage, partial fills), and fee calculation.
        *   **PortfolioManager**: Tracks balances and positions (already exists, needs refinement).
    *   **State**: Returns *raw* observations (prices, balances). Normalization is delegated.

4.  **Normalization Layer (The "Fishbowl")**
    *   **Implementation**: `VecNormalize` (from Stable Baselines3 or custom equivalent).
    *   **Workflow**:
        *   **Training**: Updates running mean/std of observations. Clips rewards.
        *   **Saving**: Serializes `vecnormalize.pkl` alongside the model checkpoint.
        *   **Inference/Testing**: Loads `vecnormalize.pkl` and sets `training=False` (freezes stats).

5.  **StableRewardCalculator**
    *   **Responsibility**: Compute rewards based on PnL and risk.
    *   **Safety**: Clips rewards to a sane range (e.g., [-10, 10]) to prevent destabilizing the policy.

6.  **CheckpointState**
    *   **Structure**:
        ```python
        {
            'model_state_dict': ...,
            'optimizer_state_dict': ...,
            'vecnormalize_state': ...,  # Crucial
            'fusion_state': ...,        # For multi-expert
            'epoch': ...,
            'rng_states': ...
        }
        ```

## Data Flow

1.  **Preprocessing**: Raw OHLCV -> `prepare_data_v2.py` -> Clean Parquet (Train/Test).
2.  **Training Loop**:
    *   `SeedManager` init.
    *   `DataManager` loads Train Parquet.
    *   `RealisticTradingEnv` instantiated.
    *   `VecNormalize` wraps Env.
    *   Agent interacts -> `VecNormalize` updates stats -> Model learns.
    *   Periodic Checkpoint -> Saves Model + `VecNormalize` stats.
3.  **Inference/Backtest**:
    *   `DataManager` loads Test Parquet.
    *   Load Model + `VecNormalize` stats.
    *   `VecNormalize` set to `norm_reward=False`, `training=False`.
    *   Run Episode.
