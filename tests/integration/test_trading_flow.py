import unittest
import logging
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
from adan_trading_bot.common.config_loader import ConfigLoader

class TestTradingFlow(unittest.TestCase):
    """
    Integration Test A: Environment -> Portfolio Pipeline
    Verifies that a forced action in the environment correctly updates the portfolio.
    """
    
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TestTradingFlow")
        
        # Load real config but mock heavy components if needed
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config('config/config.yaml')
        
        # Override config for testing
        self.config['initial_capital'] = 10000.0
        self.config['trading_rules']['min_notional_usdt'] = 10.0
        
        # Mock ChunkedDataLoader to avoid needing real parquet files
        self.loader_patcher = patch('adan_trading_bot.environment.multi_asset_chunked_env.ChunkedDataLoader')
        self.MockLoaderClass = self.loader_patcher.start()
        
        # Setup mock data structure
        # 2 assets, 100 steps
        self.mock_data = {
            "BTCUSDT": {
                "5m": MagicMock(),
                "1h": MagicMock(),
                "4h": MagicMock()
            },
            "ETHUSDT": {
                "5m": MagicMock(),
                "1h": MagicMock(),
                "4h": MagicMock()
            }
        }
        # Configure mock dataframes
        import pandas as pd
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5T')
        for asset in self.mock_data:
            for tf in self.mock_data[asset]:
                df = pd.DataFrame({
                    'open': [50000.0] * 100,
                    'high': [51000.0] * 100,
                    'low': [49000.0] * 100,
                    'close': [50000.0] * 100,
                    'volume': [1000.0] * 100,
                    'date': dates
                })
                # Add required columns for observation
                for col in ['rsi', 'macd', 'atr', 'adx', 'cci', 'bb_upper', 'bb_lower', 'obv']:
                    df[col] = 0.5
                
                self.mock_data[asset][tf] = df
        
        # Configure the mock loader instance
        self.mock_loader_instance = self.MockLoaderClass.return_value
        self.mock_loader_instance.load_chunk.return_value = self.mock_data
        self.mock_loader_instance.get_available_chunks.return_value = [0]
        self.mock_loader_instance.assets = ["BTCUSDT", "ETHUSDT"]
        
        # Initialize Environment
        self.env = RealisticTradingEnv(
            config=self.config,
            worker_id=0,
            live_mode=False,
            min_hold_steps=0, # Disable hold for easier testing
            cooldown_steps=0, # Disable cooldown
            daily_trade_limit=100
        )
        
        # After initialization, inject current_chunk_data to provide timestamp info
        self.env.current_chunk_data = self.mock_data
        self.env.current_chunk = 0
        self.env.current_step = 0
        self.env.reset()

    def tearDown(self):
        self.loader_patcher.stop()

    def test_pipeline_env_to_portfolio(self):
        """
        The 'Smoke Test': Inject a hard buy action and verify portfolio update.
        """
        initial_cash = self.env.portfolio_manager.cash
        self.logger.info(f"Initial Cash: {initial_cash}")
        
        # Verify we have no positions
        self.assertEqual(len(self.env.portfolio_manager.positions), 0)
        
        # Create a HARD BUY action for the first asset (BTCUSDT)
        # Action space is usually [action, stop_loss, take_profit] per asset
        # 1.0 = Strong Buy
        action = np.zeros(self.env.action_space.shape)
        action[0] = 1.0  # Buy BTC
        action[1] = 0.05 # SL 5%
        action[2] = 0.10 # TP 10%
        
        self.logger.info("Injecting Hard Buy Action...")
        
        # Execute step
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Verify Portfolio Update
        final_cash = self.env.portfolio_manager.cash
        positions = self.env.portfolio_manager.positions
        
        self.logger.info(f"Final Cash: {final_cash}")
        self.logger.info(f"Positions: {positions.keys()}")
        
        # Checks
        # 1. Cash should have decreased (trade cost + fees)
        self.assertLess(final_cash, initial_cash, "Cash did not decrease after buy!")
        
        # 2. Position should exist
        self.assertIn("BTCUSDT", positions, "BTCUSDT position not found in portfolio!")
        self.assertTrue(positions["BTCUSDT"].is_open, "Position is not marked as open!")
        
        # 3. Check trade count in info
        self.assertGreater(info.get('trades_executed', 0), 0, "Info does not report executed trades!")
        
        self.logger.info("✅ Pipeline Test Passed: Action -> Trade -> Portfolio Update verified.")

if __name__ == '__main__':
    unittest.main()
