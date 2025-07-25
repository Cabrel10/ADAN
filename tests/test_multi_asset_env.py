import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.environment.state_builder import TimeframeConfig
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Mock Classes for Dependencies ---
class MockChunkedDataLoader:
    """A mock data loader that returns predictable data chunks."""
    def __init__(self, data_chunks, assets_list, features_by_timeframe):
        self.data_chunks = data_chunks
        self.total_chunks = len(data_chunks)
        self.assets_list = assets_list
        self.features_by_timeframe = features_by_timeframe
        logger.info(f"MockChunkedDataLoader initialized with {self.total_chunks} chunks.")

    def load_chunk(self, chunk_index):
        if chunk_index < self.total_chunks:
            logger.info(f"MockChunkedDataLoader: Loading chunk {chunk_index}")
            return self.data_chunks[chunk_index]
        raise IndexError(f"Chunk index {chunk_index} out of bounds.")

class MockObservationValidator:
    """A mock observation validator."""
    def validate_observation(self, observation):
        return True

# --- Helper for creating dummy data ---
def create_dummy_data(num_steps, assets, timeframes, features_per_timeframe):
    data = {}
    for asset in assets:
        data[asset] = {}
        for tf in timeframes:
            df_columns = features_per_timeframe[tf]
            df = pd.DataFrame(
                np.random.rand(num_steps, len(df_columns)),
                columns=df_columns,
                index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_steps, freq='min'))
            )
            # Ensure OHLCV are somewhat realistic
            df['OPEN'] = 100 + np.cumsum(np.random.randn(num_steps))
            df['HIGH'] = df['OPEN'] + np.random.rand(num_steps)
            df['LOW'] = df['OPEN'] - np.random.rand(num_steps)
            df['CLOSE'] = df['OPEN'] + (np.random.rand(num_steps) - 0.5)
            df['VOLUME'] = 1000 + np.random.rand(num_steps) * 100
            data[asset][tf] = df
    return data

# --- Test Fixtures ---
@pytest.fixture
def mock_config():
    return {
        "data": {
            "assets": ["BTC", "ETH"],
            "timeframes": ["5m", "1h"],
            "chunked_loader": {"chunk_size": 100},
            "memory_optimizations": {"aggressive_cleanup": True, "force_gc": True, "memory_monitoring": False}
        },
        "environment": {
            "initial_balance": 10000.0,
            "trading_fees": 0.001,
            "max_steps": 200,
            "assets": ["BTC", "ETH"],
            "mode": "backtest",
            "base_currency": "USDT",
            "state": {"window_size": 30},
            "warmup_steps": 30,
            "reward_weights": {
                'base_weight': 1.0,
                'transaction_cost_weight': 1.0,
                'concentration_weight': 0.5,
                'smoothness_weight': 0.2
            }
        },
        "portfolio": {},
        "trading_rules": {
            "stop_loss": 0.02,
            "take_profit": 0.04
        },
        "capital_tiers": [
            {"name": "Micro Capital", "min_capital": 0.0, "max_capital": 1000000.0, "max_position_size_pct": 100, "leverage": 1.0, "risk_per_trade_pct": 1.0, "max_drawdown_pct": 5.0}
        ],
        "dbe": {}
    }

@pytest.fixture
def dummy_data_chunks(mock_config):
    assets = mock_config['data']['assets']
    timeframes = mock_config['data']['timeframes']
    features_per_timeframe = {
        "5m": ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'CCI_20_0.015', 'ROC_9', 'MFI_14', 'EMA_5', 'EMA_20', 'SUPERT_14_2.0', 'SUPERTd_14_2.0', 'SUPERTl_14_2.0', 'SUPERTs_14_2.0', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2', 'PSARr_0.02_0.2', 'ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'VWAP_D', 'OBV'],
        "1h": ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'CCI_20_0.015', 'MFI_14', 'EMA_50', 'EMA_100', 'SMA_200', 'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2', 'PSARr_0.02_0.2', 'ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'OBV', 'VWAP_D']
    }
    
    # Create 2 chunks of 100 steps each
    chunk1 = create_dummy_data(100, assets, timeframes, features_per_timeframe)
    chunk2 = create_dummy_data(100, assets, timeframes, features_per_timeframe)
    return [chunk1, chunk2]

@pytest.fixture
def mock_data_loader(dummy_data_chunks, mock_config):
    assets = mock_config['data']['assets']
    timeframes = mock_config['data']['timeframes']
    features_per_timeframe = {
        "5m": ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'CCI_20_0.015', 'ROC_9', 'MFI_14', 'EMA_5', 'EMA_20', 'SUPERT_14_2.0', 'SUPERTd_14_2.0', 'SUPERTl_14_2.0', 'SUPERTs_14_2.0', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2', 'PSARr_0.02_0.2', 'ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'VWAP_D', 'OBV'],
        "1h": ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'CCI_20_0.015', 'MFI_14', 'EMA_50', 'EMA_100', 'SMA_200', 'ISA_9', 'PSARl_0.02_0.2', 'ATRr_14', 'BBL_20_2.0', 'OBV', 'VWAP_D']
    }
    return MockChunkedDataLoader(dummy_data_chunks, assets, features_per_timeframe)

@pytest.fixture
def multi_asset_env(mock_config, mock_data_loader):
    env = MultiAssetChunkedEnv(config=mock_config, data_loader_instance=mock_data_loader)
    return env

# --- Tests ---
def test_env_initialization(multi_asset_env):
    logger.info("Running test_env_initialization")
    assert multi_asset_env is not None
    assert multi_asset_env.action_space.shape == (len(multi_asset_env.assets),)
    assert multi_asset_env.observation_space.shape[0] == multi_asset_env.state_builder.get_observation_shape()
    logger.info("test_env_initialization passed.")

def test_env_reset(multi_asset_env):
    logger.info("Running test_env_reset")
    observation, info = multi_asset_env.reset()
    assert observation is not None
    assert isinstance(observation, np.ndarray)
    assert observation.shape == multi_asset_env.observation_space.shape
    assert multi_asset_env.current_step == 0 # After warm-up, current_step should be 0
    assert multi_asset_env.portfolio.total_capital == multi_asset_env.config['environment']['initial_balance']
    logger.info("test_env_reset passed.")

def test_env_step(multi_asset_env):
    logger.info("Running test_env_step")
    # Reset environment first
    multi_asset_env.reset()

    # Perform a step with a dummy action (e.g., buy BTC, hold ETH)
    action = np.array([0.5, 0.0], dtype=np.float32) # Action for BTC, ETH
    
    initial_capital = multi_asset_env.portfolio.total_capital

    observation, reward, terminated, truncated, info = multi_asset_env.step(action)

    assert observation is not None
    assert isinstance(observation, np.ndarray)
    assert observation.shape == multi_asset_env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Check if step counter increased
    assert multi_asset_env.current_step == 1
    
    # Check if portfolio value changed (due to trade/fees/market movement)
    # This is a very basic check, more detailed checks would require mocking prices
    assert multi_asset_env.portfolio.total_capital != initial_capital
    
    logger.info("test_env_step passed.")

def test_multiple_steps_and_chunk_loading(multi_asset_env):
    logger.info("Running test_multiple_steps_and_chunk_loading")
    multi_asset_env.reset()
    
    # Step through enough times to trigger a chunk load
    # Each chunk is 100 steps, warmup is 30. So, 100 - 30 = 70 steps in first chunk after warmup
    # Then next chunk loads. Let's step 80 times to ensure next chunk is loaded.
    num_steps_to_take = 80
    
    initial_chunk_idx = multi_asset_env.current_chunk_idx
    initial_step_in_chunk = multi_asset_env.step_in_chunk
    
    for i in range(num_steps_to_take):
        action = np.array([0.1, -0.1], dtype=np.float32) # Small buy/sell
        observation, reward, terminated, truncated, info = multi_asset_env.step(action)
        
        assert observation is not None
        assert not terminated # Should not terminate early
        assert not truncated # Should not truncate early
        
        if i == (multi_asset_env.data_loader.data_chunks[0]['BTC']['5m'].shape[0] - multi_asset_env.warmup_steps - 1):
            # This is the step right before the chunk transition
            assert multi_asset_env.current_chunk_idx == initial_chunk_idx
            logger.info(f"Approaching chunk transition at step {i+1}")
        
        if i == (multi_asset_env.data_loader.data_chunks[0]['BTC']['5m'].shape[0] - multi_asset_env.warmup_steps):
            # This is the step where the chunk transition should occur
            assert multi_asset_env.current_chunk_idx == initial_chunk_idx + 1
            assert multi_asset_env.step_in_chunk == 0 # Should reset to 0 for new chunk
            logger.info(f"Chunk transition occurred at step {i+1}")

    assert multi_asset_env.current_step == num_steps_to_take
    assert multi_asset_env.current_chunk_idx == initial_chunk_idx + 1 # Should have moved to next chunk
    assert not multi_asset_env.done # Should not be done yet
    logger.info("test_multiple_steps_and_chunk_loading passed.")

def test_env_termination_at_end_of_data(multi_asset_env):
    logger.info("Running test_env_termination_at_end_of_data")
    multi_asset_env.reset()
    
    # Step through almost all data
    # Total steps available = (total_chunks * chunk_size) - warmup_steps
    total_data_steps = multi_asset_env.data_loader.total_chunks * multi_asset_env.data_loader.data_chunks[0]['BTC']['5m'].shape[0]
    steps_to_take = total_data_steps - multi_asset_env.warmup_steps + 5 # Go slightly beyond end
    
    terminated = False
    truncated = False
    for i in range(steps_to_take):
        action = np.array([0.0, 0.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = multi_asset_env.step(action)
        if terminated or truncated:
            logger.info(f"Environment terminated/truncated at step {i+1}")
            break
            
    assert terminated or truncated
    assert multi_asset_env.done
    logger.info("test_env_termination_at_end_of_data passed.")

def test_get_applicable_tier(multi_asset_env, mock_config):
    logger.info("Running test_get_applicable_tier")
    # Test with capital within a tier
    tier = multi_asset_env.get_applicable_tier(5000.0)
    assert tier['name'] == "Micro Capital"
    
    # Test with capital at min_capital boundary
    tier = multi_asset_env.get_applicable_tier(0.0)
    assert tier['name'] == "Micro Capital"
    
    # Test with capital at max_capital boundary (should still be in tier)
    # Note: The mock config has a single tier with max_capital 1M, so any value below that is in.
    tier = multi_asset_env.get_applicable_tier(999999.0)
    assert tier['name'] == "Micro Capital"
    
    # Add another tier to mock_config for more robust testing
    mock_config['capital_tiers'].append({
        "name": "Medium Capital", 
        "min_capital": 1000000.0, 
        "max_capital": 5000000.0, 
        "max_position_size_pct": 50, 
        "leverage": 2.0, 
        "risk_per_trade_pct": 2.0, 
        "max_drawdown_pct": 3.0
    })
    # Re-initialize env to pick up new config (or directly update tiers if possible)
    # For simplicity, we'll just test with the updated config directly if possible
    # Or create a new env instance for this specific test
    env_with_tiers = MultiAssetChunkedEnv(config=mock_config, data_loader_instance=mock_data_loader)
    tier = env_with_tiers.get_applicable_tier(1500000.0)
    assert tier['name'] == "Medium Capital"
    
    logger.info("test_get_applicable_tier passed.")

def test_validate_config(multi_asset_env, mock_config):
    logger.info("Running test_validate_config")
    # Test with a valid config (should not raise error)
    try:
        multi_asset_env._validate_config()
        assert True # No error raised
    except ValueError as e:
        pytest.fail(f"_validate_config raised unexpected error: {e}")
        
    # Test with missing section
    invalid_config = mock_config.copy()
    del invalid_config['data']
    with pytest.raises(ValueError, match="Missing config section: data"):
        MultiAssetChunkedEnv(config=invalid_config, data_loader_instance=mock_data_loader)
        
    # Test with invalid capital_tiers type
    invalid_config = mock_config.copy()
    invalid_config['capital_tiers'] = "not a list"
    with pytest.raises(ValueError, match="capital_tiers must be a list of tier configurations"):
        MultiAssetChunkedEnv(config=invalid_config, data_loader_instance=mock_data_loader)
        
    # Test with missing field in tier
    invalid_config = mock_config.copy()
    invalid_config['capital_tiers'] = [{'name': 'Test', 'min_capital': 0.0}]
    with pytest.raises(ValueError, match="Tier at index 0 is missing required field: max_capital"):
        MultiAssetChunkedEnv(config=invalid_config, data_loader_instance=mock_data_loader)
        
    logger.info("test_validate_config passed.")

# Run tests if script is executed directly
if __name__ == "__main__":
    pytest.main([__file__])