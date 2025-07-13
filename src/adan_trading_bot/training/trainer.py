#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training module for the ADAN trading bot with support for CNN feature extraction.
"""
import os
import time
import numpy as np
import torch as th
import torch.nn as nn
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

from ..common.utils import get_logger, load_config, create_directories
from ..data_processing.data_loader import load_merged_data, prepare_multi_timeframe_data
from ..environment.multi_asset_env import MultiAssetEnv
from ..models.feature_extractors import CustomCNNFeatureExtractor, get_feature_extractor

# Configure logger
logger = get_logger()

class CustomActorCriticPolicy(ActorCriticPolicy):
    """Custom policy with CNN feature extractor for 3D observations."""
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        # Custom feature extractor for 3D observations
        features_extractor_class = kwargs.pop('features_extractor_class', CustomCNNFeatureExtractor)
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch or [dict(pi=[256, 128], vf=[256, 128])],
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs
        )

class TrainingConfig:
    """Configuration class for model training."""
    
    def __init__(self, config: Dict[str, Any]):
        # Training parameters
        self.total_timesteps = config.get('total_timesteps', 1_000_000)
        self.n_steps = config.get('n_steps', 2048)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.ent_coef = config.get('ent_coef', 0.0)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Environment parameters
        self.n_envs = config.get('n_envs', 1)
        self.use_frame_stack = config.get('use_frame_stack', True)
        self.frame_stack = config.get('frame_stack', 4)
        self.normalize = config.get('normalize', True)
        
        # Model saving
        self.save_freq = config.get('save_freq', 10000)
        self.eval_freq = config.get('eval_freq', 10000)
        self.best_model_save_path = config.get('best_model_save_path', 'models/best')
        self.log_path = config.get('log_path', 'logs')
        self.tensorboard_log = config.get('tensorboard_log', 'logs/tensorboard')
        
        # Random seed for reproducibility
        self.seed = config.get('seed', 42)
        
        # Device (GPU if available, else CPU)
        self.device = 'cuda' if th.cuda.is_available() and config.get('use_gpu', True) else 'cpu'
        
        # Create necessary directories
        create_directories([self.best_model_save_path, self.log_path, self.tensorboard_log])
        
        # Set random seeds for reproducibility
        set_random_seed(self.seed)
        th.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        if th.cuda.is_available():
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False

def create_envs(
    train_data: Dict[str, pd.DataFrame],
    val_data: Dict[str, pd.DataFrame],
    config: TrainingConfig,
    env_config: Dict[str, Any]
) -> Tuple[GymEnv, GymEnv]:
    """
    Create training and evaluation environments.
    
    Args:
        train_data: Dictionary of training data DataFrames
        val_data: Dictionary of validation data DataFrames
        config: Training configuration
        env_config: Environment configuration
        
    Returns:
        Tuple of (train_env, eval_env)
    """
    def make_env(data: Dict[str, pd.DataFrame], is_training: bool = True) -> GymEnv:
        """Create a single environment."""
        def _init() -> MultiAssetEnv:
            env = MultiAssetEnv(
                data=data,
                config=env_config,
                is_training=is_training
            )
            return env
        
        env = _init()
        env = Monitor(env)
        
        # Wrap in DummyVecEnv if using multiple environments
        if config.n_envs > 1:
            env = DummyVecEnv([lambda: env] * config.n_envs)
        else:
            env = DummyVecEnv([lambda: env])
        
        # Frame stacking if enabled
        if config.use_frame_stack and len(env.observation_space.shape) >= 3:
            env = VecFrameStack(env, n_stack=config.frame_stack)
        
        # Normalize observations and rewards if enabled
        if config.normalize:
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        return env
    
    # Create training and evaluation environments
    train_env = make_env(train_data, is_training=True)
    eval_env = make_env(val_data, is_training=False) if val_data is not None else None
    
    return train_env, eval_env

def train_agent(
    config_path: str,
    custom_config: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[BaseCallback]] = None
) -> PPO:
    """
    Train a PPO agent with the given configuration.
    
    Args:
        config_path: Path to the configuration file
        custom_config: Optional dictionary to override config values
        callbacks: List of additional callbacks to use during training
        
    Returns:
        Trained PPO agent
    """
    # Load configuration
    config = load_config(config_path)
    if custom_config:
        config.update(custom_config)
    
    # Create training config
    train_config = TrainingConfig(config.get('training', {}))
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    train_data = load_merged_data(config['data_sources']['train_dir'])
    val_data = load_merged_data(config['data_sources']['val_dir']) if 'val_dir' in config['data_sources'] else None
    
    # Prepare multi-timeframe data
    train_data = prepare_multi_timeframe_data(train_data, config)
    if val_data is not None:
        val_data = prepare_multi_timeframe_data(val_data, config)
    
    # Create environments
    logger.info("Creating environments...")
    train_env, eval_env = create_envs(train_data, val_data, train_config, config.get('environment', {}))
    
    # Define policy kwargs for custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=512,
            channels=config['environment'].get('num_channels', 3),
            kernel_sizes=[3, 3, 3],
            use_batch_norm=True,
            dropout=0.1
        ),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=th.nn.ReLU,
        ortho_init=True
    )
    
    # Create agent
    logger.info("Initializing PPO agent...")
    agent = PPO(
        policy=CustomActorCriticPolicy,
        env=train_env,
        learning_rate=train_config.learning_rate,
        n_steps=train_config.n_steps,
        batch_size=train_config.batch_size,
        n_epochs=train_config.n_epochs,
        gamma=train_config.gamma,
        gae_lambda=train_config.gae_lambda,
        clip_range=train_config.clip_range,
        ent_coef=train_config.ent_coef,
        vf_coef=train_config.vf_coef,
        max_grad_norm=train_config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=train_config.tensorboard_log,
        verbose=1,
        device=train_config.device,
        seed=train_config.seed
    )
    
    # Setup callbacks
    if callbacks is None:
        callbacks = []
    
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(train_config.save_freq // train_config.n_envs, 1),
        save_path=train_config.best_model_save_path,
        name_prefix='model',
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Add evaluation callback if validation data is available
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=train_config.best_model_save_path,
            log_path=train_config.log_path,
            eval_freq=max(train_config.eval_freq // train_config.n_envs, 1),
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            warn=False
        )
        callbacks.append(eval_callback)
    
    # Create callback list
    callback = CallbackList(callbacks)
    
    # Train the agent
    logger.info(f"Starting training for {train_config.total_timesteps} timesteps...")
    start_time = time.time()
    
    try:
        agent.learn(
            total_timesteps=train_config.total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
            tb_log_name="ppo_cnn_training"
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds.")
    
    # Save the final model
    final_model_path = os.path.join(train_config.best_model_save_path, 'final_model')
    agent.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save the environment stats if normalization was used
    if train_config.normalize and hasattr(train_env, 'save'):
        vec_normalize_path = os.path.join(train_config.best_model_save_path, 'vec_normalize.pkl')
        train_env.save(vec_normalize_path)
        logger.info(f"VecNormalize stats saved to {vec_normalize_path}")
    
    # Close environments
    train_env.close()
    if eval_env is not None:
        eval_env.close()
    
    return agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the ADAN trading bot with PPO')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                      help='Path to the training configuration file')
    parser.add_argument('--model-path', type=str, default='models/ppo_cnn',
                      help='Path to save the trained model')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                      help='Number of timesteps to train for')
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU for training if available')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    custom_config = {
        'training': {
            'total_timesteps': args.timesteps,
            'use_gpu': args.gpu
        },
        'models_dir': args.model_path
    }
    
    # Start training
    train_agent(args.config, custom_config=custom_config)