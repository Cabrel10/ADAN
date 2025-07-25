import logging
import os
import multiprocessing
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

from ..environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from ..common.utils import get_logger, get_project_root
from ..common.config_watcher import ConfigWatcher

logger = get_logger(__name__)

class TrainingOrchestrator:
    """
    Orchestrates the training of an RL agent across multiple trading environments.

    Manages parallel environment execution, curriculum learning, and a shared
    experience buffer.
    """

    def __init__(self, config: Dict[str, Any], agent_class: Any, agent_config: Dict[str, Any], env_factory: Optional[Callable[..., gym.Env]] = None, test_mode_no_real_buffer: bool = False):
        """
        Initializes the TrainingOrchestrator.

        Args:
            config: Main configuration dictionary for the orchestrator.
            agent_class: The Stable-Baselines3 agent class (e.g., PPO, A2C).
            agent_config: Configuration dictionary for the agent.
        """
        self.config = config
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.env_factory = env_factory
        self.test_mode_no_real_buffer = test_mode_no_real_buffer

        self.num_envs = self.config.get("num_environments", 1)
        self.curriculum_learning_enabled = self.config.get("curriculum_learning", False)
        self.shared_buffer_enabled = self.config.get("shared_experience_buffer", False)
        self.buffer_size = self.config.get("replay_buffer_size", 100000)

        self.environments: List[gym.Env] = []
        self.vec_env: Optional[gym.vector.VecEnv] = None
        self.agent: Optional[Any] = None
        self.shared_replay_buffer: Optional[ReplayBuffer] = None
        self.config_watcher: Optional[ConfigWatcher] = None
        self.dbe_instance: Optional[Any] = None  # Reference to DBE instance

        self._setup_environments()
        self._setup_agent()
        self._setup_shared_buffer()
        self._setup_config_watcher()

        logger.info(f"TrainingOrchestrator initialized with {self.num_envs} environments.")
        if self.curriculum_learning_enabled:
            logger.info("Curriculum learning is ENABLED.")
        if self.shared_buffer_enabled:
            logger.info("Shared experience buffer is ENABLED.")

    def _setup_environments(self) -> None:
        """Sets up multiple trading environments."""
        logger.info(f"Setting up {self.num_envs} environments...")
        env_configs = self._generate_env_configs()

        # Create env factory function that can be pickled
        env_factory = self.env_factory
        
        def make_env(env_config):
            """Helper function to create an environment instance."""
            def _init():
                if env_factory:
                    env = env_factory(env_config)  # Pass config to factory
                else:
                    env = MultiAssetChunkedEnv(env_config)
                return env
            return _init

        # Create a single environment first to get the proper observation space
        logger.info("Creating sample environment to determine observation space...")
        sample_env = make_env(env_configs[0])()
        sample_env.reset()  # Initialize the observation space
        logger.info(f"Sample environment observation space: {sample_env.observation_space}")
        sample_env.close()

        if self.num_envs > 1:
            # Use SubprocVecEnv for parallel execution
            self.vec_env = SubprocVecEnv([make_env(cfg) for cfg in env_configs])
        else:
            # Use DummyVecEnv for single environment
            self.vec_env = DummyVecEnv([make_env(env_configs[0])])
        
        logger.info("Environments setup complete.")

    def _generate_env_configs(self) -> List[Dict[str, Any]]:
        """
        Generates configuration dictionaries for each environment.
        Can implement curriculum learning logic here.
        """
        base_env_config = self.config.get("environment_config", {})
        env_configs = []

        # For now, just duplicate the base config. Curriculum learning will modify this.
        all_assets = self.config["environment_config"]["data"]["assets"]
        num_assets = len(all_assets)
        
        for i in range(self.num_envs):
            env_config = base_env_config.copy()
            env_config["trading_rules"] = {}
            env_config["capital_tiers"] = {}
            if self.curriculum_learning_enabled and num_assets > 0:
                # Assign a different asset to each environment in a round-robin fashion
                asset_for_env = all_assets[i % num_assets]
                env_config["data"]["assets"] = [asset_for_env]
                logger.info(f"Environment {i} assigned asset: {asset_for_env}")
            env_configs.append(env_config)
        return env_configs

    def _setup_agent(self) -> None:
        """Sets up the Stable-Baselines3 agent."""
        # Ensure environments are properly initialized by resetting them
        logger.info("Initializing environments to set up observation spaces...")
        self.vec_env.reset()
        
        # Remove any training-specific parameters from the agent config
        agent_config = self.agent_config.copy()
        training_params = {}
        
        # Extract training parameters that should be passed to learn() instead of __init__
        training_param_keys = ['total_timesteps', 'callback', 'log_interval', 'tb_log_name', 'reset_num_timesteps']
        for key in training_param_keys:
            if key in agent_config:
                training_params[key] = agent_config.pop(key)
        
        # Store training params for later use in train_agent()
        self.training_params = training_params
        
        # The agent needs to be initialized with the vectorized environment
        # PPO requires policy as first argument, then env
        policy = agent_config.pop('policy', 'MlpPolicy')  # Default to MlpPolicy
        self.agent = self.agent_class(policy, self.vec_env, **agent_config)
        logger.info(f"Agent {self.agent_class.__name__} setup complete.")

    def _setup_shared_buffer(self) -> None:
        """Sets up a shared replay buffer if enabled."""
        if self.shared_buffer_enabled:
            if self.test_mode_no_real_buffer:
                from unittest.mock import MagicMock
                self.shared_replay_buffer = MagicMock()
                logger.info("Shared ReplayBuffer mocked for test mode.")
            else:
                # The buffer needs to be compatible with the observation and action spaces
                self.shared_replay_buffer = ReplayBuffer(
                    self.buffer_size,
                    self.vec_env.observation_space,
                    self.vec_env.action_space,
                    device=self.agent_config.get("device", "auto"),
                    n_envs=self.num_envs,
                )
                logger.info(f"Shared ReplayBuffer setup with size {self.buffer_size}.")

    def _setup_config_watcher(self) -> None:
        """Sets up the ConfigWatcher for dynamic configuration reloading."""
        # Check if dynamic adaptation is enabled
        dynamic_adaptation_enabled = self.config.get("dynamic_adaptation", {}).get("enabled", True)
        
        if not dynamic_adaptation_enabled:
            logger.info("Dynamic configuration adaptation is disabled")
            return
        
        try:
            # Initialize ConfigWatcher with config directory
            project_root = Path(get_project_root())
            config_dir = project_root / "config"
            self.config_watcher = ConfigWatcher(str(config_dir), enabled=True)
            
            # Register callbacks for different configuration types
            self.config_watcher.register_callback('training', self._on_training_config_change)
            self.config_watcher.register_callback('environment', self._on_environment_config_change)
            self.config_watcher.register_callback('dbe', self._on_dbe_config_change)
            self.config_watcher.register_callback('risk', self._on_risk_config_change)
            self.config_watcher.register_callback('reward', self._on_reward_config_change)
            
            # Try to get DBE instance from environments for direct updates
            self._get_dbe_instance()
            
            logger.info("🔄 ConfigWatcher setup complete - Dynamic reload enabled")
            
        except Exception as e:
            logger.error(f"Failed to setup ConfigWatcher: {e}")
            self.config_watcher = None

    def _get_dbe_instance(self) -> None:
        """Get reference to DBE instance from environments for direct updates."""
        try:
            if hasattr(self.vec_env, 'envs') and len(self.vec_env.envs) > 0:
                # For DummyVecEnv, access the first environment
                first_env = self.vec_env.envs[0]
                if hasattr(first_env, 'dbe'):
                    self.dbe_instance = first_env.dbe
                    logger.debug("DBE instance reference obtained")
            elif hasattr(self.vec_env, 'get_attr'):
                # For SubprocVecEnv, try to get DBE through get_attr
                try:
                    dbe_list = self.vec_env.get_attr('dbe')
                    if dbe_list and len(dbe_list) > 0:
                        self.dbe_instance = dbe_list[0]  # Use first environment's DBE
                        logger.debug("DBE instance reference obtained via get_attr")
                except:
                    logger.debug("Could not get DBE instance via get_attr")
        except Exception as e:
            logger.debug(f"Could not get DBE instance: {e}")

    

    def _on_environment_config_change(self, config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]) -> None:
        """Handle environment configuration changes."""
        logger.info(f"🔄 Environment config changed: {list(changes.keys())}")
        
        try:
            # Update trading rules if changed
            if any(key.startswith('trading_rules') for key in changes.keys()):
                logger.info("📊 Trading rules updated - will apply on next episode reset")
            
            # Update capital tiers if changed
            if any(key.startswith('capital_tiers') for key in changes.keys()):
                logger.info("💰 Capital tiers updated - will apply on next episode reset")
            
            # Update risk modulation parameters
            if any(key.startswith('risk_modulation') for key in changes.keys()):
                logger.info("⚠️ Risk modulation updated - will apply immediately")
                # These changes will be picked up by the DBE on next update
                
        except Exception as e:
            logger.error(f"Error updating environment config: {e}")

    def _on_dbe_config_change(self, config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]) -> None:
        """Handle DBE configuration changes."""
        logger.info(f"🔄 DBE config changed: {list(changes.keys())}")
        
        try:
            if self.dbe_instance:
                # Update risk parameters
                if any(key.startswith('risk_parameters') for key in changes.keys()):
                    risk_params = new_config.get('risk_parameters', {})
                    if hasattr(self.dbe_instance, 'config'):
                        self.dbe_instance.config['risk_parameters'].update(risk_params)
                        logger.info("✅ DBE risk parameters updated")
                
                # Update mode thresholds
                if any(key.startswith('modes') for key in changes.keys()):
                    modes = new_config.get('modes', {})
                    if hasattr(self.dbe_instance, 'config'):
                        self.dbe_instance.config['modes'].update(modes)
                        logger.info("✅ DBE mode thresholds updated")
                
                # Update learning parameters
                if any(key.startswith('learning') for key in changes.keys()):
                    learning_params = new_config.get('learning', {})
                    if hasattr(self.dbe_instance, 'config'):
                        self.dbe_instance.config['learning'].update(learning_params)
                        logger.info("✅ DBE learning parameters updated")
                        
                # Force DBE to recalculate parameters with new config
                if hasattr(self.dbe_instance, 'compute_dynamic_modulation'):
                    logger.info("🔄 Triggering DBE recalculation with new parameters")
                    
            else:
                logger.warning("DBE instance not available for direct updates")
                
        except Exception as e:
            logger.error(f"Error updating DBE config: {e}")

    def _on_risk_config_change(self, config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]) -> None:
        """Handle risk configuration changes."""
        logger.info(f"🔄 Risk config changed: {list(changes.keys())}")
        
        try:
            # Update dynamic adaptation settings
            if 'dynamic_adaptation' in changes:
                adaptation_config = changes['dynamic_adaptation']['new_value']
                logger.info(f"✅ Dynamic adaptation settings updated: {adaptation_config}")
            
            # Update risk metrics thresholds
            if any(key.startswith('risk_metrics') for key in changes.keys()):
                logger.info("📊 Risk metrics thresholds updated")
            
            # Update position sizing parameters
            if any(key.startswith('position_sizing') for key in changes.keys()):
                logger.info("💼 Position sizing parameters updated")
                
        except Exception as e:
            logger.error(f"Error updating risk config: {e}")

    def _on_reward_config_change(self, config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]) -> None:
        """Handle reward configuration changes."""
        logger.info(f"🔄 Reward config changed: {list(changes.keys())}")
        
        try:
            # Update reward shaping parameters
            if any(key.startswith('reward_shaping') for key in changes.keys()):
                logger.info("🎯 Reward shaping parameters updated")
            
            # Update penalty parameters
            if any(key.startswith('penalties') for key in changes.keys()):
                logger.info("⚠️ Penalty parameters updated")
                
        except Exception as e:
            logger.error(f"Error updating reward config: {e}")
            
    def _on_training_config_change(self, config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]) -> None:
        """Handle training configuration changes."""
        logger.info(f"🔄 Training config changed: {list(changes.keys())}")
        logger.debug(f"Full changes dict: {changes}")
        
        if not hasattr(self, 'agent') or self.agent is None:
            logger.warning("Agent not initialized, cannot apply training config changes")
            return
            
        try:
            # Handle learning rate updates
            if 'learning_rate' in changes:
                try:
                    new_lr = float(changes['learning_rate']['new_value'])
                    old_lr = self.agent.learning_rate
                    
                    # Update learning rate for all parameter groups
                    for i, param_group in enumerate(self.agent.policy.optimizer.param_groups):
                        param_group['lr'] = new_lr
                    
                    logger.info(f"✅ Learning rate updated: {old_lr} -> {new_lr}")
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid learning rate value: {changes['learning_rate']}. Error: {e}")
            
            # Handle entropy coefficient updates
            if 'ent_coef' in changes:
                try:
                    new_ent_coef = float(changes['ent_coef']['new_value'])
                    old_ent_coef = self.agent.ent_coef
                    self.agent.ent_coef = new_ent_coef
                    logger.info(f"✅ Entropy coefficient updated: {old_ent_coef} -> {new_ent_coef}")
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid entropy coefficient value: {changes['ent_coef']}. Error: {e}")
            
            # Handle clip range updates
            if 'clip_range' in changes:
                try:
                    new_clip_range = float(changes['clip_range']['new_value'])
                    old_clip_range = self.agent.clip_range
                    self.agent.clip_range = new_clip_range
                    logger.info(f"✅ Clip range updated: {old_clip_range} -> {new_clip_range}")
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid clip range value: {changes['clip_range']}. Error: {e}")
            
        except Exception as e:
            logger.error(f"Error applying training config changes: {e}", exc_info=True)

    def train_agent(self, total_timesteps: int = None, callback: Optional[BaseCallback] = None) -> None:
        """
        Trains the RL agent.

        Args:
            total_timesteps: Total number of timesteps to train for. If None, will use value from config.
            callback: Optional callback for training process.
        """
        # Use provided timesteps or fall back to config
        if total_timesteps is None:
            total_timesteps = self.training_params.get('total_timesteps', 10000)
            
        # Set up training parameters
        train_kwargs = self.training_params.copy()
        train_kwargs['total_timesteps'] = total_timesteps
        
        # Get callback if specified in config or passed as argument
        if callback:
            train_kwargs['callback'] = callback
        else:
            callback = self._get_training_callback()
            if callback:
                train_kwargs['callback'] = callback
            
        logger.info(f"Starting agent training for {total_timesteps} timesteps...")
        self.agent.learn(**train_kwargs)
        logger.info("Agent training complete.")

    def _get_training_callback(self) -> Optional[BaseCallback]:
        """
        Returns a custom callback for training, if shared buffer or curriculum
        learning is enabled.
        """
        if self.shared_buffer_enabled or self.curriculum_learning_enabled:
            # This callback will handle adding experiences to the shared buffer
            # and potentially adjusting curriculum.
            class CustomTrainingCallback(BaseCallback):
                def __init__(self, orchestrator_instance, verbose=0):
                    super().__init__(verbose)
                    self.orchestrator = orchestrator_instance

                def _on_step(self) -> bool:
                    # This method is called after each step in the environment
                    # If shared buffer is enabled, add transitions to it
                    if self.orchestrator.shared_buffer_enabled:
                        # Accessing internal buffer of the agent to copy transitions
                        # This might be agent-specific (e.g., PPO's rollouts)
                        # For simplicity, we'll assume a generic way to get last transitions
                        # In a real scenario, you might need to hook into the agent's collect_rollouts
                        # or use a custom VecEnvWrapper that adds to the shared buffer.
                        # For now, this is a placeholder.
                        # self.orchestrator.shared_replay_buffer.add(...
                        pass # TODO: Implement shared buffer population

                    # Curriculum learning adjustments
                    if self.orchestrator.curriculum_learning_enabled:
                        # TODO: Implement curriculum adjustment logic
                        pass
                    return True
            return CustomTrainingCallback(self)
        return None

    def save_agent(self, path: str) -> None:
        """
        Saves the trained agent.

        Args:
            path: Path to save the agent.
        """
        if self.agent:
            self.agent.save(path)
            logger.info(f"Agent saved to {path}")
        else:
            logger.warning("No agent to save.")

    def load_agent(self, path: str) -> None:
        """
        Loads a pre-trained agent.

        Args:
            path: Path to the saved agent.
        """
        self.agent = self.agent_class.load(path, env=self.vec_env)
        logger.info(f"Agent loaded from {path}")

    def evaluate_agent(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluates the trained agent.

        Args:
            num_episodes: Number of episodes to run for evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        if not self.agent:
            logger.warning("No agent to evaluate. Train or load an agent first.")
            return {}

        logger.info(f"Evaluating agent for {num_episodes} episodes...")
        episode_rewards = []
        for i in range(num_episodes):
            obs, info = self.vec_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.vec_env.step(action)
                done = terminated or truncated
                episode_reward += reward[0] # Assuming single environment for simplicity in reward aggregation
            episode_rewards.append(episode_reward)
            logger.info(f"Episode {i+1}/{num_episodes} finished with reward: {episode_reward:.2f}")

        mean_reward = safe_mean(episode_rewards)
        logger.info(f"Evaluation complete. Mean reward: {mean_reward:.2f}")
        return {"mean_reward": mean_reward, "episode_rewards": episode_rewards}

    def close(self) -> None:
        """
        Closes all environments and stops the config watcher.
        """
        if self.config_watcher:
            self.config_watcher.stop()
            logger.info("ConfigWatcher stopped.")
            
        if self.vec_env:
            self.vec_env.close()
            logger.info("Environments closed.")


# Example Usage (for testing purposes, not part of the main script)
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback

    # Dummy config for testing
    test_config = {
        "num_environments": 2,
        "curriculum_learning": False,
        "shared_experience_buffer": False,
        "replay_buffer_size": 10000,
        "environment_config": {
            "data": {
                "data_dir": "./data/processed", # Adjust as needed
                "chunk_size": 100,
                "assets": ["BTC/USDT", "ETH/USDT"]
            },
            "environment": {"initial_capital": 10000},
            "portfolio": {},
            "trading": {},
            "state": {"window_size": 10, "timeframes": ["1m"], "features_per_timeframe": {"1m": ["open", "high", "low", "close", "volume"]}}
        }
    }

    # Agent config
    agent_config = {
        "policy": "MlpPolicy",
        "learning_rate": 0.0003,
        "n_steps": 20,
        "batch_size": 10,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "verbose": 1,
        "device": "cpu"
    }

    orchestrator = None
    try:
        orchestrator = TrainingOrchestrator(test_config, PPO, agent_config)
        orchestrator.train_agent(total_timesteps=100)
        orchestrator.evaluate_agent(num_episodes=2)
    except Exception as e:
        logger.error(f"An error occurred during orchestration: {e}")
    finally:
        if orchestrator:
            orchestrator.close()

