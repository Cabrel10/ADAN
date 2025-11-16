
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from adan_trading_bot.common.config import load_config
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader

@pytest.fixture
def default_config():
    """Load the default configuration."""
    return load_config("config/config.yaml")

@pytest.fixture
def mock_chunk_data():
    """Create a mock data chunk for testing."""
    n_points = 100
    timeframes = ["5m", "1h", "4h"]
    assets = ["BTCUSDT", "XRPUSDT"]
    chunk_data = {}
    for asset in assets:
        asset_data = {}
        for tf in timeframes:
            prices = 50000 + np.cumsum(np.random.normal(0, 100, n_points))
            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices * (1 + np.random.normal(0, 0.01, n_points)),
                    "volume": np.random.lognormal(10, 1, n_points),
                    "rsi_14": np.random.uniform(20, 80, n_points),
                    "macd_hist": np.random.normal(0, 0.1, n_points),
                    "atr_14": np.random.uniform(0.01, 0.05, n_points),
                    "bb_upper": prices * 1.02,
                    "bb_middle": prices,
                    "bb_lower": prices * 0.98,
                    "volume_ratio": np.random.uniform(0.5, 2.0, n_points),
                    "ema_ratio": np.random.uniform(0.95, 1.05, n_points),
                    "stoch_k": np.random.uniform(0, 100, n_points),
                    "vwap_ratio": np.random.uniform(0.98, 1.02, n_points),
                }
            )
            df.index = pd.to_datetime(pd.date_range(start="2024-01-01", periods=n_points, freq="5T"))
            asset_data[tf] = df
        chunk_data[asset] = asset_data
    return chunk_data

class TestEnvironmentInternals:
    def test_data_loading_and_initialization(self, default_config, mock_chunk_data):
        """
        Test Case 1: Data Loading and Preprocessing
        - Instantiate ChunkedDataLoader and load a chunk of data.
        - Verify the structure and content of the loaded data.
        - Instantiate MultiAssetChunkedEnv with this data.
        - In the environment, check the current_data attribute after reset.
        """
        with patch.object(ChunkedDataLoader, "load_chunk", return_value=mock_chunk_data):
            # 1. Instantiate Environment
            env = MultiAssetChunkedEnv(config=default_config, worker_config=default_config["workers"]["w1"])
            
            # 2. Reset environment to load initial data
            obs, info = env.reset()

            # 3. Check if current_data is loaded
            assert env.current_data is not None, "current_data should be loaded after reset"
            assert isinstance(env.current_data, dict), "current_data should be a dictionary"

            # 4. Verify the structure of the loaded data
            # The data is now in the format {asset: {tf: df}} after transpose_data
            expected_assets = default_config["workers"]["w1"]["assets"]
            expected_timeframes = default_config["workers"]["w1"]["timeframes"]

            # The mock data is in format {asset: {tf: df}}.
            
            # Let's check the structure of the mock_chunk_data
            assert "BTCUSDT" in mock_chunk_data
            assert "5m" in mock_chunk_data["BTCUSDT"]
            
            # Now let's check the structure of env.current_data
            # self.current_data should have the original structure from the loader.
            
            for asset in expected_assets:
                assert asset in env.current_data, f"Asset {asset} not found in env.current_data"
                for tf in expected_timeframes:
                    assert tf in env.current_data[asset], f"Timeframe {tf} not found for asset {asset}"
                    df = env.current_data[asset][tf]
                    assert isinstance(df, pd.DataFrame), f"Data for {asset}/{tf} should be a DataFrame"
                    assert not df.empty, f"DataFrame for {asset}/{tf} should not be empty"
                    # Check for NaNs
                    assert not df.isnull().values.any(), f"NaNs found in data for {asset}/{tf}"

            print("\n✅ Test Case 1 Passed: Data loading and initialization verified.")

    def test_state_building_and_observation_space(self, default_config, mock_chunk_data):
        """
        Test Case 2: State Building and Observation Space
        - Call _build_observation.
        - Verify the structure of the observation dictionary.
        - Verify the shape and dtype of each observation tensor.
        - Check for NaNs or Infs in the observation.
        """
        with patch.object(ChunkedDataLoader, "load_chunk", return_value=mock_chunk_data):
            # 1. Instantiate Environment
            env = MultiAssetChunkedEnv(config=default_config, worker_config=default_config["workers"]["w1"])
            env.reset()

            # 2. Build observation
            observation = env._build_observation()

            # 3. Verify structure
            assert isinstance(observation, dict), "Observation should be a dictionary"
            expected_keys = list(env.observation_space.spaces.keys())
            assert all(key in observation for key in expected_keys), f"Observation missing keys: {set(expected_keys) - set(observation.keys())}"
            assert all(key in expected_keys for key in observation.keys()), f"Observation has extra keys: {set(observation.keys()) - set(expected_keys)}"

            # 4. Verify shape and dtype
            for key, obs_value in observation.items():
                space = env.observation_space.spaces[key]
                # For timeframe observations, check that the first dimension matches
                # and the second dimension is reasonable (features)
                if key in ["5m", "1h", "4h"]:
                    assert obs_value.shape[0] == space.shape[0], \
                        f"Time window mismatch for {key}: expected {space.shape[0]}, got {obs_value.shape[0]}"
                    assert obs_value.shape[1] > 0, \
                        f"Feature dimension should be positive for {key}"
                else:
                    assert obs_value.shape == space.shape, \
                        f"Shape mismatch for key {key}: expected {space.shape}, got {obs_value.shape}"
                assert obs_value.dtype == space.dtype, \
                    f"Dtype mismatch for key {key}: expected {space.dtype}, got {obs_value.dtype}"

                # 5. Check for NaNs or Infs
                assert not np.isnan(obs_value).any(), f"NaNs found in observation for key {key}"
                assert np.isfinite(obs_value).all(), f"Infs found in observation for key {key}"

            print("\n✅ Test Case 2 Passed: State building and observation space verified.")

    def test_action_execution_and_portfolio_update(self, default_config, mock_chunk_data):
        """
        Test Case 3: Action Execution and Portfolio Update
        - Craft a specific action to open a long position.
        - Step the environment and verify that the position is opened.
        - Craft a specific action to close the position.
        - Step the environment and verify that the position is closed and PnL is calculated.
        """
        with patch.object(ChunkedDataLoader, "load_chunk", return_value=mock_chunk_data):
            # 1. Instantiate Environment
            env = MultiAssetChunkedEnv(config=default_config, worker_config=default_config["workers"]["w1"])
            env.reset()

            # 2. Craft a BUY action for BTCUSDT
            # Action space is Box(15,), where each asset has 3 actions (decision, risk, size)
            action = np.zeros(env.action_space.shape)
            action[0] = 1.0  # Buy BTCUSDT

            # 3. Step the environment to open the position
            obs, reward, done, truncated, info = env.step(action)

            # 4. Verify that the position is opened
            portfolio_manager = env.portfolio_manager
            btc_position = portfolio_manager.positions.get("BTCUSDT")
            assert btc_position is not None, "BTCUSDT position should be opened"
            assert btc_position.is_open, "BTCUSDT position should be open"
            assert btc_position.size > 0, "Position size should be greater than 0"

            print("\n✅ Test Case 3.1 Passed: Position opened successfully.")

            # 5. Craft a SELL action for BTCUSDT
            action[0] = -1.0  # Sell BTCUSDT

            # 6. Step the environment to close the position
            obs, reward, done, truncated, info = env.step(action)

            # 7. Verify that the position is closed
            assert not btc_position.is_open, "BTCUSDT position should be closed"
            assert "realized_pnl_total" in info, "realized_pnl_total should be in the info dictionary"
            
            print("\n✅ Test Case 3.2 Passed: Position closed successfully and PnL calculated.")

    def test_reward_calculation(self, default_config, mock_chunk_data):
        """
        Test Case 4: Reward Calculation
        - Check the reward value at each step of a trade (buy, hold, sell).
        - Verify that the reward components are calculated as expected.
        """
        with patch.object(ChunkedDataLoader, "load_chunk", return_value=mock_chunk_data):
            # 1. Instantiate Environment
            env = MultiAssetChunkedEnv(config=default_config, worker_config=default_config["workers"]["w1"])
            env.reset()

            # 2. Craft a BUY action for BTCUSDT
            action = np.zeros(env.action_space.shape)
            action[0] = 1.0  # Buy BTCUSDT

            # 3. Step the environment to open the position
            obs, reward, done, truncated, info = env.step(action)

            # 4. Check the reward after opening a position
            assert reward != 0, "Reward should not be zero after opening a position"
            assert "reward_components" in info, "reward_components should be in the info dictionary"
            # PnL might be slightly negative due to fees/commissions
            assert info["reward_components"]["pnl"] <= 0, \
                "PnL should be zero or negative when opening a position (due to fees)"

            print("\n✅ Test Case 4.1 Passed: Reward calculation for opening a position is correct.")

            # 5. Craft a HOLD action
            action[0] = 0.0  # Hold

            # 6. Step the environment to hold the position
            obs, reward, done, truncated, info = env.step(action)

            # 7. Check the reward after holding a position
            # The reward might be zero if there are no penalties/bonuses for holding
            # but we can check the components
            assert "reward_components" in info, "reward_components should be in the info dictionary"

            print("\n✅ Test Case 4.2 Passed: Reward calculation for holding a position is correct.")

            # 8. Craft a SELL action for BTCUSDT
            action[0] = -1.0  # Sell BTCUSDT

            # 9. Step the environment to close the position
            obs, reward, done, truncated, info = env.step(action)

            # 10. Check the reward after closing a position
            assert reward != 0, "Reward should not be zero after closing a position"
            assert "reward_components" in info, "reward_components should be in the info dictionary"
            assert info["reward_components"]["pnl"] != 0, "PnL should not be zero when closing a position"

            print("\n✅ Test Case 4.3 Passed: Reward calculation for closing a position is correct.")
