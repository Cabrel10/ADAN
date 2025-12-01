#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for ADAN 2.0 Multi-Expert System
====================================================================
50 trials per worker × 4 workers = 200 total trials

Architecture: RecurrentPPO (PPO + LSTM) + CNN (via feature extractor)
"""

import optuna
import yaml
import numpy as np
import torch
from pathlib import Path
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
import sys
import os

# Add src to path
sys.path.insert(0, 'src')
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Worker-specific constraints (strategic diversity)
WORKER_CONSTRAINTS = {
    'w1': {  # Scalper (high frequency)
        'daily_trade_limit': (20, 30),
        'min_hold_steps': (1, 3),
        'frequency_weight': (0.01, 0.05),
        'asset_cooldown_steps': (1, 2),
    },
    'w2': {  # Swing trader (medium hold)
        'daily_trade_limit': (10, 20),
        'min_hold_steps': (5, 10),
        'frequency_weight': (0.05, 0.15),
        'asset_cooldown_steps': (3, 5),
    },
    'w3': {  # Trend follower (conservative)
        'daily_trade_limit': (5, 15),
        'min_hold_steps': (8, 15),
        'drawdown_weight': (0.2, 0.4),
        'asset_cooldown_steps': (3, 5),
    },
    'w4': {  # Opportunist (adaptive)
        'daily_trade_limit': (15, 25),
        'min_hold_steps': (3, 7),
        'sharpe_weight': (0.2, 0.4),
        'asset_cooldown_steps': (2, 4),
    }
}


def suggest_hyperparameters(trial: optuna.Trial, worker_id: str) -> dict:
    """
    Suggest hyperparameters for PPO+CNN+LSTM architecture.
    """
    constraints = WORKER_CONSTRAINTS[worker_id]
    
    params = {
        # ===== PPO Core Parameters =====
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
        'vf_coef': trial.suggest_float('vf_coef', 0.25, 0.75),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
        
        # ===== Training Dynamics =====
        'n_steps': trial.suggest_int('n_steps', 1024, 4096, step=512),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        'n_epochs': trial.suggest_int('n_epochs', 5, 20),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
        
        # ===== Neural Network Architecture =====
        # PPO+CNN+LSTM confirmed - optimize layer sizes
        'net_arch_size': trial.suggest_categorical('net_arch_size', [128, 256, 384, 512]),
        'net_arch_layers': trial.suggest_int('net_arch_layers', 2, 4),
        'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [64, 128, 256]),
        # cnn_features is handled by the feature extractor in the env or policy
        
        # ===== Reward Function =====
        'pnl_normalization': trial.suggest_float('pnl_normalization', 25.0, 100.0),
        'sharpe_weight': trial.suggest_float(
            'sharpe_weight', 
            constraints.get('sharpe_weight', (0.1, 0.4))[0],
            constraints.get('sharpe_weight', (0.1, 0.4))[1]
        ),
        'drawdown_weight': trial.suggest_float(
            'drawdown_weight',
            constraints.get('drawdown_weight', (0.1, 0.4))[0],
            constraints.get('drawdown_weight', (0.1, 0.4))[1]
        ),
        'frequency_weight': trial.suggest_float(
            'frequency_weight',
            constraints.get('frequency_weight', (0.01, 0.1))[0],
            constraints.get('frequency_weight', (0.01, 0.1))[1]
        ),
        
        # ===== Trading Rules (Worker-Specific) =====
        'min_notional_usdt': trial.suggest_float('min_notional_usdt', 5.0, 15.0),
        'daily_trade_limit': trial.suggest_int(
            'daily_trade_limit',
            constraints['daily_trade_limit'][0],
            constraints['daily_trade_limit'][1]
        ),
        'asset_cooldown_steps': trial.suggest_int(
            'asset_cooldown_steps',
            constraints['asset_cooldown_steps'][0],
            constraints['asset_cooldown_steps'][1]
        ),
        'min_hold_steps': trial.suggest_int(
            'min_hold_steps',
            constraints['min_hold_steps'][0],
            constraints['min_hold_steps'][1]
        ),
        
        # ===== Risk Management =====
        'max_position_size': trial.suggest_float('max_position_size', 0.7, 0.95),
        'circuit_breaker_pct': trial.suggest_float('circuit_breaker_pct', 10.0, 20.0),
        'slippage_bps': trial.suggest_float('slippage_bps', 0.01, 0.05),
        'fee_bps': trial.suggest_float('fee_bps', 0.02, 0.06),
    }
    
    # Validation: Ensure reward weights sum <= 1.0 (PnL prioritized)
    reward_sum = (
        params['sharpe_weight'] + 
        params['drawdown_weight'] + 
        params['frequency_weight']
    )
    if reward_sum > 1.0:
        # Normalize while preserving ratios
        scale = 0.9 / reward_sum  # Leave 0.1 buffer
        params['sharpe_weight'] *= scale
        params['drawdown_weight'] *= scale
        params['frequency_weight'] *= scale
    
    return params


def objective(trial: optuna.Trial, worker_id: str, base_config: dict, n_training_steps: int = 25000) -> float:
    """
    Objective function for Optuna optimization.
    """
    logger.info(f"[{worker_id}] Trial {trial.number} started")
    
    # Suggest hyperparameters
    params = suggest_hyperparameters(trial, worker_id)
    
    # Update config with trial parameters
    config = base_config.copy()
    
    # FIX: Update config['agent']['ppo'] instead of config['ppo']
    if 'agent' in config and 'ppo' in config['agent']:
        config['agent']['ppo'].update({
            'learning_rate': params['learning_rate'],
            'clip_range': params['clip_range'],
            'ent_coef': params['ent_coef'],
            'vf_coef': params['vf_coef'],
            'gae_lambda': params['gae_lambda'],
            'n_steps': params['n_steps'],
            'batch_size': params['batch_size'],
            'n_epochs': params['n_epochs'],
            'max_grad_norm': params['max_grad_norm'],
        })
    
    # Prepare configs for RealisticTradingEnv
    reward_config = {
        'pnl_normalization': params['pnl_normalization'],
        'sharpe_weight': params['sharpe_weight'],
        'drawdown_weight': params['drawdown_weight'],
        'frequency_weight': params['frequency_weight'],
        'consistency_weight': 0.1  # Default
    }
    
    friction_config = {
        'slippage_bps': params['slippage_bps'],
        'fee_bps': params['fee_bps']
    }
    
    # Create environment
    def make_env():
        return RealisticTradingEnv(
            config=config,
            worker_config=config['workers'][worker_id],
            worker_id=int(worker_id[1]),  # Extract numeric ID
            enable_market_friction=True,
            reward_config=reward_config,
            friction_config=friction_config,
            min_hold_steps=params['min_hold_steps'],
            daily_trade_limit=params['daily_trade_limit'],
            cooldown_steps=params['asset_cooldown_steps'],
            min_notional=params['min_notional_usdt'],
            circuit_breaker_pct=params['circuit_breaker_pct'] / 100.0
        )
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # Construct policy_kwargs for RecurrentPPO
    # This ensures net_arch and lstm_hidden_size are actually used
    net_arch = [params['net_arch_size']] * params['net_arch_layers']
    
    policy_kwargs = {
        "net_arch": net_arch,
        "lstm_hidden_size": params['lstm_hidden_size'],
        "enable_critic_lstm": True,
        "optimizer_kwargs": {"eps": 1e-5}
    }
    
    # Train model using RecurrentPPO (MultiInputLstmPolicy)
    model = RecurrentPPO(
        'MultiInputLstmPolicy',
        env,
        learning_rate=params['learning_rate'],
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        gamma=params['gamma'],
        gae_lambda=params['gae_lambda'],
        clip_range=params['clip_range'],
        ent_coef=params['ent_coef'],
        vf_coef=params['vf_coef'],
        max_grad_norm=params['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=42 + trial.number,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    try:
        model.learn(total_timesteps=n_training_steps)
        
        # Evaluate on validation data
        obs = env.reset()
        # Reset LSTM states
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        
        all_rewards = []
        episode_returns = []
        episode_return = 0
        
        # Improved evaluation (5000 steps to allow episode completion)
        n_eval_steps = 5000
        for step_idx in range(n_eval_steps):
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            obs, reward, done, info = env.step(action)
            episode_starts = done
            episode_return += reward[0]
            all_rewards.append(reward[0])
            
            if done[0]:
                episode_returns.append(episode_return)
                episode_return = 0
                # obs is already reset by env
        
        # Calculate metric with better handling for incomplete episodes
        if len(all_rewards) > 0:
            mean_reward = np.mean(all_rewards)
            
            # If we have completed episodes, prefer that metric
            if len(episode_returns) > 0:
                mean_episode_return = np.mean(episode_returns)
                # Use Sharpe-like metric: mean return / std if possible
                if len(episode_returns) > 1:
                    std_return = np.std(episode_returns)
                    metric = mean_episode_return / (std_return + 1e-8)
                else:
                    metric = mean_episode_return
            else:
                # No completed episodes - use cumulative return so far
                metric = episode_return if episode_return != 0 else mean_reward
        else:
            metric = -999.0
        
        # Log detailed info
        n_episodes = len(episode_returns)
        mean_episode_return = np.mean(episode_returns) if n_episodes > 0 else episode_return
        logger.info(
            f"[{worker_id}] Trial {trial.number}: "
            f"Metric={metric:.3f}, Episodes={n_episodes}, "
            f"MeanEpReturn={mean_episode_return:.2f}, "
            f"MeanStepReward={mean_reward if len(all_rewards) > 0 else 0:.4f}"
        )
        
        # Report intermediate values for pruning
        trial.report(metric, step=n_training_steps)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return metric
        
    except Exception as e:
        logger.error(f"[{worker_id}] Trial {trial.number} failed: {e}")
        return -999.0
    
    finally:
        env.close()


def optimize_worker(worker_id: str, n_trials: int = 50):
    """
    Run Optuna optimization for a single worker.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Starting Optuna optimization for {worker_id}")
    logger.info(f"Trials: {n_trials} | Strategy: {WORKER_CONSTRAINTS[worker_id]}")
    logger.info(f"{'='*60}")
    
    # Load base config
    config_loader = ConfigLoader()
    base_config = config_loader.load_config('config/config.yaml')
    
    # Create study
    study = optuna.create_study(
        study_name=f"adan_2.0_{worker_id}",
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1000),
        storage=f'sqlite:///optuna_{worker_id}.db',
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, worker_id, base_config),
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True
    )
    
    # Save best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"\n{'='*60}")
    logger.info(f"[{worker_id}] Optimization Complete!")
    logger.info(f"Best Sharpe Ratio: {best_value:.3f}")
    logger.info(f"Best Parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"{'='*60}\n")
    
    # Save to YAML
    output_path = Path(f'config/optuna_best_{worker_id}.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    
    logger.info(f"Best parameters saved to {output_path}")
    
    return study


def main():
    """Run optimization for all 4 workers (200 trials total)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run 1 trial per worker for testing")
    args = parser.parse_args()
    
    workers = ['w1', 'w2', 'w3', 'w4']
    n_trials_per_worker = 1 if args.dry_run else 50
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# ADAN 2.0 Optuna Hyperparameter Optimization")
    logger.info(f"# Total Trials: {n_trials_per_worker * len(workers)}")
    logger.info(f"# Workers: {len(workers)} (RecurrentPPO architecture)")
    logger.info(f"# Strategy: Multi-Expert Diversification")
    if args.dry_run:
        logger.info("# MODE: DRY RUN (Testing)")
    logger.info(f"{'#'*70}\n")
    
    studies = {}
    for worker_id in workers:
        study = optimize_worker(worker_id, n_trials=n_trials_per_worker)
        studies[worker_id] = study
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("OPTIMIZATION SUMMARY (All Workers)")
    logger.info(f"{'='*70}")
    for worker_id, study in studies.items():
        logger.info(f"{worker_id}: Best Sharpe = {study.best_value:.3f}")
    logger.info(f"{'='*70}\n")
    
    logger.info("✅ All optimizations complete! Best configs saved to config/optuna_best_*.yaml")


if __name__ == '__main__':
    main()
