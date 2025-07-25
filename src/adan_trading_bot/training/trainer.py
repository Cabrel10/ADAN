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
import gymnasium as gym
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Type
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

import adan_trading_bot.agent.ppo_agent as ppo_agent
from adan_trading_bot.common.utils import load_config, create_directories
import adan_trading_bot.common.utils as utils
from adan_trading_bot.data_processing.chunked_loader import ChunkedDataLoader
import adan_trading_bot.data_processing.chunked_loader as chunked_loader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
import adan_trading_bot.environment.multi_asset_chunked_env as multi_asset_env
import adan_trading_bot.models.feature_extractors as feature_extractors

# Configure logger
logger = utils.get_logger()

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
    config: TrainingConfig,
    env_config: Dict[str, Any]
) -> Tuple[GymEnv, GymEnv]:
    """
    Create training and evaluation environments.
    
    Args:
        config: Training configuration
        env_config: Environment configuration
        
    Returns:
        Tuple of (train_env, eval_env)
    """
    # Créer le data loader avec les bons paramètres
    data_loader = ChunkedDataLoader(
        data_dir='data/final',
        assets_list=env_config['data']['assets'],
        timeframes=env_config['data']['timeframes'],
        features_by_timeframe=env_config['data']['features_per_timeframe'],
        split='train',
        chunk_size=config.batch_size
    )
    
    # Ajouter les configurations nécessaires à l'environnement
    env_config.update({
        'data': {
            'assets': ['BTC', 'ETH', 'SOL', 'XRP', 'ADA'],
            'timeframes': ['5m', '1h', '4h'],
            'features_per_timeframe': {
                '5m': ['open', 'high', 'low', 'close', 'volume'],
                '1h': ['open', 'high', 'low', 'close', 'volume'],
                '4h': ['open', 'high', 'low', 'close', 'volume']
            }
        },
        'mode': 'train',
        'initial_balance': 10000.0,
        'trading_fees': 0.001,
        'state': {
            'window_size': 30,
            'include_portfolio_state': True,
            'observation_shape': (3, 30, 5)  # (timeframes, window_size, features_per_timeframe)
        }
    })
    
    # Créer les environnements d'entraînement et d'évaluation
    train_env = DummyVecEnv([
        lambda: Monitor(
            MultiAssetChunkedEnv(
                config=env_config
            )
        )
    ] * config.n_envs)

    # Créer l'environnement d'évaluation
    eval_env = DummyVecEnv([
        lambda: Monitor(
            MultiAssetChunkedEnv(
                config=env_config
            )
        )
    ])

    # Normaliser les observations si demandé
    if config.normalize:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        if eval_env is not None:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    # Ajouter le stacking de frames si demandé
    if config.use_frame_stack:
        train_env = VecFrameStack(train_env, n_stack=config.frame_stack)
        if eval_env is not None:
            eval_env = VecFrameStack(eval_env, n_stack=config.frame_stack)

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
        callbacks: Optional list of callbacks for training
    
    Returns:
        Trained PPO agent
    """
    # Load and merge configuration
    config = load_config(config_path)
    if custom_config:
        config.update(custom_config)
    
    # Create training configuration
    training_config = TrainingConfig(config)
    
    # Initialize environment configuration with data parameters
    env_config = config.get('environment', {})
    env_config.update({
        'data': {
            'assets': ['BTC', 'ETH', 'SOL', 'XRP', 'ADA'],
            'timeframes': ['5m', '1h', '4h'],
            'features_per_timeframe': {
                '5m': ['open', 'high', 'low', 'close', 'volume'],
                '1h': ['open', 'high', 'low', 'close', 'volume'],
                '4h': ['open', 'high', 'low', 'close', 'volume']
            }
        },
        'mode': 'train',
        'initial_balance': 10000.0,
        'trading_fees': 0.001,
        'trading_rules': {
            'futures_enabled': False,
            'leverage': 1,
            'commission_pct': 0.1,
            'futures_commission_pct': 0.02,
            'min_trade_size': 0.0001,
            'min_notional_value': 10.0,
            'max_notional_value': 100000.0
        },
        'risk_management': {
            'capital_tiers': [
                {'max_capital': 1000, 'max_position_size_pct': 5.0, 'max_drawdown_pct': 2.0},
                {'max_capital': 10000, 'max_position_size_pct': 3.0, 'max_drawdown_pct': 1.5},
                {'max_capital': 100000, 'max_position_size_pct': 2.0, 'max_drawdown_pct': 1.0}
            ],
            'position_sizing': {
                'max_risk_per_trade_pct': 1.0,
                'max_asset_allocation_pct': 20.0,
                'concentration_limits': {
                    'BTC': 30.0,
                    'ETH': 25.0,
                    'SOL': 20.0,
                    'XRP': 15.0,
                    'ADA': 10.0
                }
            }
        },
        'state': {
            'window_size': 30,
            'include_portfolio_state': True
        }
    })
    
    # Create environments
    train_env, eval_env = create_envs(training_config, env_config)
    
    # Define policy kwargs for custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=512,
            channels=3,
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            paddings=[1, 1, 1],
            activation_fn=th.nn.ReLU,
            use_batch_norm=True,
            dropout=0.1
        ),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=th.nn.ReLU,
        ortho_init=True
    )
    
    # Create PPO agent
    agent = create_ppo_agent(
        env=train_env,
        config=training_config,
        tensorboard_log=training_config.tensorboard_log,
        ent_coef=training_config.ent_coef,
        vf_coef=training_config.vf_coef,
        max_grad_norm=training_config.max_grad_norm,
        device=training_config.device,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # Create callbacks
    callback_list = []
    
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=train_config.save_freq,
        save_path=train_config.best_model_save_path,
        name_prefix='rl_model'
    )
    callback_list.append(checkpoint_callback)
    
    # Add evaluation callback if we have an evaluation environment
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=train_config.best_model_save_path,
            log_path=train_config.log_path,
            eval_freq=train_config.eval_freq,
            deterministic=True,
            render=False
        )
        callback_list.append(eval_callback)
    
    # Add any additional callbacks
    if callbacks:
        callback_list.extend(callbacks)
    
    # Train the agent
    model.learn(
        total_timesteps=train_config.total_timesteps,
        callback=CallbackList(callback_list)
    )
    
    # Clean up
    if train_env is not None:
        train_env.close()
    
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
    
    # Start training
    train_agent(
        config_path=args.config,
        custom_config={
            'total_timesteps': args.timesteps,
            'use_gpu': args.gpu,
            'models_dir': args.model_path
        }
    )