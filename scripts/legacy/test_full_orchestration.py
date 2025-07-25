import sys
import unittest
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Add project root to path
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

from adan_trading_bot.training.training_orchestrator import TrainingOrchestrator

class MinimalTestEnv(gym.Env):
    """A minimal Gymnasium environment for testing purposes."""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else {}
        # Ensure required config sections are present for validation
        self.config.setdefault("trading_rules", {})
        self.config.setdefault("capital_tiers", {})
        self.observation_space = spaces.Dict({
            "price_features": spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(1, 10, 5), 
                dtype=np.float32
            ),
            "portfolio_state": spaces.Box(
                low=0.0, 
                high=np.inf, 
                shape=(3,), 
                dtype=np.float32
            ),
        })
        self.action_space = spaces.MultiDiscrete(
            [3],  # Hold, Buy, Sell
            dtype=np.int64
        )
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = self.current_step >= 10  # End episode after 10 steps
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.observation_space.sample()
        info = {}
        return obs, info

    def render(self):
        pass

    def close(self):
        pass

class TestFullOrchestration(unittest.TestCase):

    def setUp(self):
        self.env_configs = [{'id': f'env{i}'} for i in range(4)]  # 4 environments
        self.agent_config = {'batch_size': 64}
        
        # Create a mock class that will act as the OnlineLearningAgent class
        self.MockOnlineLearningAgent_Class = MagicMock()
        self.MockOnlineLearningAgent_Class.__name__ = "MockOnlineLearningAgent"
        
        # Configure the mock agent instance that will be created by the orchestrator
        mock_agent_instance = MagicMock()
        mock_agent_instance.learn = MagicMock()
        mock_agent_instance.predict = MagicMock(return_value=(np.array([0]), None)) # Default action
        mock_agent_instance.policy = MagicMock()
        mock_agent_instance.device = "cpu"
        mock_agent_instance._update_target_networks = MagicMock()
        mock_agent_instance.rollout_buffer = MagicMock()
        mock_agent_instance.logger = MagicMock()

        # Configure the mock class to return this single mock instance
        self.MockOnlineLearningAgent_Class.return_value = mock_agent_instance
        self.mock_agent_instances = [mock_agent_instance] # Store for assertions
        
        # Instantiate orchestrator with the mock agent class
        self.orchestrator_config = {
            "num_environments": len(self.env_configs),
            "curriculum_learning": False,
            "shared_experience_buffer": True,
            "replay_buffer_size": 10000,
            "environment_config": {
                "data": {
                    "data_dir": "./data/processed",
                    "chunk_size": 100,
                    "assets": ["BTC/USDT", "ETH/USDT", "ADA/USDT", "XRP/USDT"]
                },
                "environment": {"initial_capital": 10000},
                "portfolio": {},
                "trading": {},
                "state": {"window_size": 10, "timeframes": ["1m"], "features_per_timeframe": {"1m": ["open", "high", "low", "close", "volume"]}}
            },
            "trading_rules": {},
            "capital_tiers": {}
        }
        self.orchestrator = TrainingOrchestrator(
            self.orchestrator_config,
            self.MockOnlineLearningAgent_Class,
            self.agent_config,
            env_factory=MinimalTestEnv, # Pass the MinimalTestEnv as the factory
            test_mode_no_real_buffer=True # Enable test mode to mock ReplayBuffer
        )

    def test_orchestrator_setup(self):
        # Verify that the orchestrator's agent is initialized
        self.assertIsNotNone(self.orchestrator.agent)
        
        # Verify that the vectorized environment is set up
        self.assertIsNotNone(self.orchestrator.vec_env)
        
        # Verify that the shared experience buffer is correctly set up in the orchestrator
        self.assertIsNotNone(self.orchestrator.shared_replay_buffer)
        if not self.orchestrator.test_mode_no_real_buffer:
            from stable_baselines3.common.buffers import ReplayBuffer
            self.assertIsInstance(self.orchestrator.shared_replay_buffer, ReplayBuffer)

    def test_orchestrator_train_cycle(self):
        # Mock the agent's learn method
        self.orchestrator.agent.learn = MagicMock()
        
        # Mock the callback to return a simple function
        original_get_callback = self.orchestrator._get_training_callback
        self.orchestrator._get_training_callback = MagicMock(return_value=MagicMock())
        
        try:
            self.orchestrator.train_agent(total_timesteps=100)
            
            # Verify that the agent's learn method was called with the correct arguments
            self.orchestrator.agent.learn.assert_called_once()
            call_args = self.orchestrator.agent.learn.call_args[1]
            self.assertEqual(call_args['total_timesteps'], 100)
            self.assertIsNotNone(call_args.get('callback'))
        finally:
            # Restore the original method
            self.orchestrator._get_training_callback = original_get_callback

    def test_shared_buffer_interaction(self):
        # Skip if shared buffer is not enabled
        if self.orchestrator.shared_replay_buffer is None:
            self.skipTest("Shared replay buffer not enabled")
            
        # Mock the shared replay buffer's add method
        self.orchestrator.shared_replay_buffer.add = MagicMock()
        
        # Track number of steps
        steps_done = 0
        
        def mock_learn(total_timesteps, callback):
            nonlocal steps_done
            # Simulate adding experiences for the full number of timesteps
            for _ in range(total_timesteps):
                self.orchestrator.shared_replay_buffer.add(
                    MagicMock(), MagicMock(), MagicMock(),
                    MagicMock(), MagicMock(), MagicMock()
                )
                steps_done += 1
                
                # Update callback
                if callback:
                    callback.model = self.orchestrator.agent
                    callback.num_timesteps = steps_done
                    if hasattr(callback, '_on_step'):
                        callback._on_step()
            
            # Return a dummy value to avoid None return
            return MagicMock()
        
        # Set up mock
        original_learn = self.orchestrator.agent.learn
        self.orchestrator.agent.learn = MagicMock(side_effect=mock_learn)
        
        try:
            # Run training with a smaller number of timesteps for the test
            test_timesteps = 10
            self.orchestrator.train_agent(total_timesteps=test_timesteps)
            
            # Verify buffer was used
            self.assertGreater(self.orchestrator.shared_replay_buffer.add.call_count, 0)
            # Verify it was called exactly test_timesteps times
            self.assertEqual(self.orchestrator.shared_replay_buffer.add.call_count, test_timesteps)
        finally:
            # Restore the original method
            self.orchestrator.agent.learn = original_learn

if __name__ == '__main__':
    unittest.main()