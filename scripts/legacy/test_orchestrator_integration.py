#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration test for TrainingOrchestrator with MultiAssetChunkedEnv.
"""
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.adan_trading_bot.training.training_orchestrator import TrainingOrchestrator
from src.adan_trading_bot.common.custom_logger import log_info, log_error

# Set up logging - we'll use the module-level log_* functions directly

def create_test_data(output_dir: str, num_assets: int = 2, num_rows: int = 1000) -> str:
    """Create test data for the environment."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a subdirectory for each asset
    for i in range(1, num_assets + 1):
        asset_dir = os.path.join(output_dir, f'ASSET_{i}')
        os.makedirs(asset_dir, exist_ok=True)
        
        # Generate random price data for the asset
        timestamps = pd.date_range(
            start='2023-01-01', periods=num_rows, freq='1min'
        )
        
        base_prices = np.random.uniform(90, 110, num_rows).cumsum()
        asset_data = {
            'timestamp': timestamps,
            'open': base_prices,
            'high': base_prices + 5,
            'low': base_prices - 5,
            'close': base_prices + 1,
            'volume': np.random.randint(100, 1000, num_rows),
        }
        
        # Save to train.parquet
        df = pd.DataFrame(asset_data)
        output_path = os.path.join(asset_dir, 'train.parquet')
        df.to_parquet(output_path, index=False)
        log_info(f"Created test data at {output_path}")
    
    return output_dir

def get_test_config(test_data_path: str) -> Dict[str, Any]:
    """Create a test configuration for the orchestrator."""
    return {
        "num_environments": 2,
        "curriculum_learning": False,
        "shared_experience_buffer": True,
        "replay_buffer_size": 10000,
        "environment_config": {
            "data": {
                "assets": ["ASSET_1", "ASSET_2"],
                "data_path": test_data_path,
                "data_dir": test_data_path,
                "timeframe": "1min",
                "train_start_date": "2023-01-01",
                "train_end_date": "2023-01-10",
                "validation_start_date": "2023-01-11",
                "validation_end_date": "2023-01-15",
                "test_start_date": "2023-01-16",
                "test_end_date": "2023-01-20",
                "features": ["open", "high", "low", "close", "volume"],
                "target": "close",
                "window_size": 50,
                "train_test_split": 0.8,
                "normalize": True,
                "shuffle": True,
            },
            "environment": {
                "data_path": test_data_path,
                "window_size": 50,
                "frame_bound": (50, 1000),
                "assets": ["ASSET_1", "ASSET_2"],
                "initial_balance": 10000.0,
                "trading_fee": 0.001,
                "max_position": 1.0,
                "reward_type": "sharpe",
                "normalize_obs": True,
                "render_mode": None,
            },
            # Portfolio configuration
            "initial_equity": 10000.0,
            "assets": ["ASSET_1", "ASSET_2"],  # Required by PortfolioManager
            
            # Trading rules
            "trading_rules": {
                "max_trade_size": 0.1,
                "max_position_size": 0.2,
                "max_daily_trades": 10,
                "max_daily_loss": 0.05,
                "futures_enabled": False,
                "leverage": 1,
                "commission_pct": 0.001,
                "min_trade_size": 0.0001,
                "min_notional_value": 10.0,
                "max_notional_value": 100000.0
            },
            
            # Risk management
            "risk_management": {
                "capital_tiers": {
                    "tier1": {"min_balance": 10000, "max_trade_size": 0.1, "max_position_size": 0.2},
                    "tier2": {"min_balance": 50000, "max_trade_size": 0.2, "max_position_size": 0.4},
                    "tier3": {"min_balance": 100000, "max_trade_size": 0.3, "max_position_size": 0.6}
                },
                "position_sizing": {
                    "concentration_limits": {
                        "max_single_asset": 0.3,
                        "max_sector": 0.6
                    }
                }
            },
            
            # Portfolio settings
            "portfolio": {
                "initial_balance": 10000.0,
                "max_position_size": 0.1,
                "max_trade_size": 0.05,
                "max_drawdown": 0.2,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.06,
                "max_trades_per_day": 10
            },

            "training": {
                "total_timesteps": 1000,
                "learning_rate": 0.0003,
                "n_steps": 256,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "target_kl": 0.01,
                "verbose": 1,
                "tensorboard_log": "./tensorboard_logs/",
                "policy_kwargs": {
                    "net_arch": {"pi": [64, 64], "vf": [64, 64]}
                },
            }
        }
    }

def test_training_orchestrator():
    """Test the TrainingOrchestrator with a simple configuration."""
    # Create test data directory structure
    base_dir = os.path.abspath("data/test_data")
    test_data_dir = os.path.join(base_dir, "final")
    
    # Create test data
    log_info(f"Creating test data in {test_data_dir}")
    create_test_data(test_data_dir, num_assets=2, num_rows=1000)
    
    # Get test config
    config = get_test_config(test_data_dir)
    
    # Initialize the orchestrator
    try:
        orchestrator = TrainingOrchestrator(
            config=config,
            agent_class=PPO,
            agent_config=config["environment_config"]["training"],
            test_mode_no_real_buffer=True
        )
        
        # Run training
        log_info("Starting training...")
        orchestrator.train_agent()
        log_info("Training completed successfully!")
        
        # Test saving/loading
        model_path = "models/test_model"
        orchestrator.save_agent(model_path)
        log_info(f"Model saved to {model_path}")
        
        return True
        
    except Exception as e:
        log_error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(base_dir):
            import shutil
            shutil.rmtree(base_dir)

if __name__ == "__main__":
    success = test_training_orchestrator()
    if success:
        log_info("Test completed successfully!")
        sys.exit(0)
    else:
        log_error("Test failed!")
        sys.exit(1)
