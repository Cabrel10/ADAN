#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization - CALIBRATED RANGES
======================================================
Ranges adjusted based on current config.yaml baseline and worker profiles.
Focused search to avoid exhaustive exploration.

Current baselines from config.yaml:
- learning_rate: (current unknown, using 3e-4 as baseline)
- clip_range: 0.2
- ent_coef: 0.01
- batch_size: 64
- n_steps: 2048
- gamma: ~0.99
- gae_lambda: ~0.95
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

sys.path.insert(0, 'src')
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Worker-specific constraints (from config.yaml profiles)
# CALIBRATED based on current successful configs
WORKER_CONSTRAINTS = {
    'w1': {  # Scalper (high frequency) - ALREADY VALIDATED
        'daily_trade_limit': (25, 30),  # Narrow range around current 30
        'min_hold_steps': (1, 3),
        'frequency_weight': (0.02, 0.05),  # Current baseline performing well
        'asset_cooldown_steps': (1, 2),
    },
    'w2': {  # Swing trader (medium hold)
        'daily_trade_limit': (15, 20),  # Focused on medium range
        'min_hold_steps': (6, 10),  # From config: min_hold_steps exists
        'frequency_weight': (0.08, 0.12),  # Tighter around current weights
        'asset_cooldown_steps': (3, 4),
    },
    'w3': {  # Trend follower (conservative)
        'daily_trade_limit': (10, 15),  # Lower frequency
        'min_hold_steps': (10, 15),
        'drawdown_weight': (0.25, 0.35),  # Narrow around current 0.3
        'asset_cooldown_steps': (3, 5),
    },
    'w4': {  # Opportunist (adaptive)
        'daily_trade_limit': (20, 25),
        'min_hold_steps': (5, 8),
        'sharpe_weight': (0.18, 0.25),  # Narrow around current 0.2
        'asset_cooldown_steps': (2, 3),
    }
}


def suggest_hyperparameters_calibrated(trial: optuna.Trial, worker_id: str) -> dict:
    """
    CALIBRATED hyperparameter search around known good baselines.
    Ranges are ~±30% of current config.yaml values.
    """
    constraints = WORKER_CONSTRAINTS[worker_id]
    
    params = {
        # ===== PPO Core - NARROW RANGES around baselines =====
        # Baseline LR unknown, using typical 3e-4 ±50%
        'learning_rate': trial.suggest_float('learning_rate', 2e-4, 5e-4, log=True),
        
        # Baseline clip_range: 0.2 → range [0.15, 0.25]
        'clip_range': trial.suggest_float('clip_range', 0.15, 0.25),
        
        # Baseline ent_coef: 0.01 → range [0.01, 0.05] (INCREASED to prevent model collapse)
        'ent_coef': trial.suggest_float('ent_coef', 0.01, 0.05, log=True),
        
        # Baseline vf_coef: 0.5 → range [0.4, 0.6]
        'vf_coef': trial.suggest_float('vf_coef', 0.4, 0.6),
        
        # Baseline gamma: ~0.99 → range [0.985, 0.995]
        'gamma': trial.suggest_float('gamma', 0.985, 0.995),
        
        # Baseline gae_lambda: ~0.95 → range [0.92, 0.97]
        'gae_lambda': trial.suggest_float('gae_lambda', 0.92, 0.97),
        
        # ===== Training Dynamics - REDUCED search =====
        # Baseline n_steps: 2048 → {1536, 2048, 2560}
        'n_steps': trial.suggest_categorical('n_steps', [1536, 2048, 2560]),
        
        # Baseline batch_size: 64 → {64, 128} (skip 256/512 - too slow)
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        
        # Baseline n_epochs: 10 → range [8, 12]
        'n_epochs': trial.suggest_int('n_epochs', 8, 12),
        
        # Baseline max_grad_norm: 0.5 → range [0.4, 0.7]
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.4, 0.7),
        
        # ===== Neural Network - FIXED to working architecture =====
        # Use proven architecture, only optimize size
        'net_arch_size': trial.suggest_categorical('net_arch_size', [256, 384]),  # Skip 128 (too small), 512 (too large)
        'net_arch_layers': 3,  # FIXED - 3 layers works well
        'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [128, 256]),  # Skip 64 (too small)
        
        # ===== Reward Function - TIGHT ranges around validated values =====
        # Dynamic normalization handles this, but allow small adjustment
        'pnl_normalization': trial.suggest_float('pnl_normalization', 90.0, 110.0),  # Near 100.0 baseline
        
        'sharpe_weight': trial.suggest_float(
            'sharpe_weight', 
            constraints.get('sharpe_weight', (0.15, 0.25))[0],
            constraints.get('sharpe_weight', (0.15, 0.25))[1]
        ),
        'drawdown_weight': trial.suggest_float(
            'drawdown_weight',
            constraints.get('drawdown_weight', (0.25, 0.35))[0],  # Baseline 0.3
            constraints.get('drawdown_weight', (0.25, 0.35))[1]
        ),
        'frequency_weight': trial.suggest_float(
            'frequency_weight',
            constraints.get('frequency_weight', (0.08, 0.12))[0],  # Baseline 0.1
            constraints.get('frequency_weight', (0.08, 0.12))[1]
        ),
        'invalid_sell_penalty_weight': trial.suggest_float('invalid_sell_penalty_weight', 0.03, 0.08),  # NEW - critical for sell bias
        
        # ===== Trading Rules - Worker-specific TIGHT ranges =====
        'min_notional_usdt': 11.0,  # FIXED - exchange minimum
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
        
        # ===== Risk Management - NARROW around current tiers =====
        'max_position_size': trial.suggest_float('max_position_size', 0.85, 0.95),  # Near 0.9
        'circuit_breaker_pct': 15.0,  # FIXED - validated at 15%
        'slippage_bps': trial.suggest_float('slippage_bps', 0.015, 0.025),  # Near 0.02
        'fee_bps': trial.suggest_float('fee_bps', 0.035, 0.045),  # Near 0.04
    }
    
    # Validation: Ensure reward weights sum <= 1.0
    reward_sum = (
        params['sharpe_weight'] + 
        params['drawdown_weight'] + 
        params['frequency_weight'] +
        params['invalid_sell_penalty_weight']
    )
    if reward_sum > 0.8:  # Leave room for PnL
        scale = 0.75 / reward_sum
        params['sharpe_weight'] *= scale
        params['drawdown_weight'] *= scale
        params['frequency_weight'] *= scale
        params['invalid_sell_penalty_weight'] *= scale
    
    return params


# Worker-specific success metrics (from user requirements)
WORKER_TARGETS = {
    'w1': {'metric': 'trades_per_day', 'target': 25, 'name': 'Scalper'},
    'w2': {'metric': 'sharpe_ratio', 'target': 1.8, 'name': 'Swing'},
    'w3': {'metric': 'profit_factor', 'target': 2.5, 'name': 'Trend'},
    'w4': {'metric': 'calmar_ratio', 'target': 3.0, 'name': 'Opportuniste'}
}


def calculate_sharpe_from_portfolio(portfolio_values: list) -> float:
    """Calculate Sharpe Ratio from portfolio value series."""
    if len(portfolio_values) < 2:
        return 0.0
    
    # Calculate returns
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
        
    # Annualize (assuming 5m steps -> 288 steps/day -> 252 trading days)
    # sqrt(288 * 252) approx sqrt(72576) approx 269.4
    annualization_factor = np.sqrt(288 * 252)
    sharpe = np.mean(returns) / np.std(returns) * annualization_factor
    return float(sharpe)


def calculate_calmar_from_portfolio(portfolio_values: list) -> float:
    """Calculate Calmar Ratio from portfolio value series."""
    if len(portfolio_values) < 2:
        return 0.0
        
    # Total return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    # Max Drawdown
    values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(values)
    drawdowns = (running_max - values) / running_max
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    if max_dd == 0:
        return 0.0
        
    # Annualize return? Calmar is usually Annual Return / Max DD
    # But for short eval, we use Total Return / Max DD as proxy
    return float(total_return / max_dd)


def calculate_worker_metric(worker_id: str, info_list: list, all_rewards: list, n_eval_steps: int, portfolio_values: list) -> float:
    """
    Calculate worker-specific metric based on targets.
    
    Returns normalized metric where 1.0 = target achieved.
    """
    target_config = WORKER_TARGETS[worker_id]
    metric_name = target_config['metric']
    target_value = target_config['target']
    
    if metric_name == 'trades_per_day':
        # Extract natural trades from last info
        if len(info_list) > 0 and 'frequency' in info_list[-1]:
            natural_trades = info_list[-1]['frequency'].get('counts', {}).get('daily_total', 0)
            # Normalize: eval was ~3000 steps, scale to full day equivalent
            # Assuming 5m timeframe: 288 steps per day (24h * 60min / 5min)
            trades_per_day_estimate = (natural_trades / n_eval_steps) * 288
            metric_value = trades_per_day_estimate / target_value  # 1.0 when target achieved
            logger.info(f"  [{worker_id}] Trades/day: {trades_per_day_estimate:.1f} (target: {target_value}, score: {metric_value:.2f})")
            return metric_value
        else:
            return 0.0
    
    elif metric_name == 'sharpe_ratio':
        # Calculate Sharpe from portfolio values (robust to incomplete episodes)
        sharpe = calculate_sharpe_from_portfolio(portfolio_values)
        metric_value = sharpe / target_value  # 1.0 when target achieved
        logger.info(f"  [{worker_id}] Sharpe: {sharpe:.2f} (target: {target_value}, score: {metric_value:.2f})")
        return max(metric_value, 0.0)  # Cap at 0 for negative Sharpe
    
    elif metric_name == 'profit_factor':
        # Calculate Profit Factor: sum(wins) / abs(sum(losses))
        winning_trades = [r for r in all_rewards if r > 0]
        losing_trades = [r for r in all_rewards if r < 0]
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            total_wins = sum(winning_trades)
            total_losses = abs(sum(losing_trades))
            profit_factor = total_wins / (total_losses + 1e-8)
            metric_value = profit_factor / target_value
            logger.info(f"  [{worker_id}] PF: {profit_factor:.2f} (target: {target_value}, score: {metric_value:.2f})")
            return metric_value
        else:
            return 0.0
    
    elif metric_name == 'calmar_ratio':
        # Calculate Calmar from portfolio values
        calmar = calculate_calmar_from_portfolio(portfolio_values)
        metric_value = calmar / target_value
        logger.info(f"  [{worker_id}] Calmar: {calmar:.2f} (target: {target_value}, score: {metric_value:.2f})")
        return max(metric_value, 0.0)
    
    return 0.0


def objective(trial: optuna.Trial, worker_id: str, base_config: dict, n_training_steps: int = 30000) -> float:
    """
    WORKER-SPECIFIC objective using targets from user requirements.
    """
    logger.info(f"[{worker_id}] Trial {trial.number} started - Target: {WORKER_TARGETS[worker_id]['metric']} > {WORKER_TARGETS[worker_id]['target']}")
    
    params = suggest_hyperparameters_calibrated(trial, worker_id)
    
    config = base_config.copy()
    
    # Update agent config
    if 'agent' not in config:
        config['agent'] = {}
    config['agent'].update({
        'learning_rate': params['learning_rate'],
        'clip_range': params['clip_range'],
        'ent_coef': params['ent_coef'],
        'vf_coef': params['vf_coef'],
        'gamma': params['gamma'],
        'gae_lambda': params['gae_lambda'],
        'n_steps': params['n_steps'],
        'batch_size': params['batch_size'],
        'n_epochs': params['n_epochs'],
        'max_grad_norm': params['max_grad_norm'],
    })
    
    # Prepare environment configs
    reward_config = {
        'pnl_normalization': params['pnl_normalization'],
        'sharpe_weight': params['sharpe_weight'],
        'drawdown_weight': params['drawdown_weight'],
        'frequency_weight': params['frequency_weight'],
        'consistency_weight': 0.1,
        'invalid_sell_penalty_weight': params['invalid_sell_penalty_weight']
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
    
    # Build policy
    net_arch = [params['net_arch_size']] * params['net_arch_layers']
    policy_kwargs = {
        "net_arch": net_arch,
        "lstm_hidden_size": params['lstm_hidden_size'],
        "enable_critic_lstm": True,
        "optimizer_kwargs": {"eps": 1e-5}
    }
    
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
        
        # Evaluate
        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        
        all_rewards = []
        episode_returns = []
        episode_return = 0
        info_list = []
        portfolio_values = []  # NEW: Collect portfolio values
        invalid_sells = 0
        natural_trades_pct = 0
        
        # Eval loop
        n_eval_steps = 3000
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
            info_list.append(info[0])
            
            # Collect portfolio value
            if 'portfolio_value' in info[0]:
                portfolio_values.append(info[0]['portfolio_value'])
            
            # Track invalid sells
            if 'invalid_sell_attempts' in info[0]:
                invalid_sells = info[0]['invalid_sell_attempts']
            
            if done[0]:
                episode_returns.append(episode_return)
                episode_return = 0
        
        # Calculate worker-specific metric using portfolio values
        worker_metric = calculate_worker_metric(worker_id, info_list, all_rewards, n_eval_steps, portfolio_values)
        
        # Calculate natural trades percentage (GLOBAL objective)
        if len(info_list) > 0 and 'frequency' in info_list[-1]:
            natural_trades = info_list[-1]['frequency'].get('counts', {}).get('daily_total', 0)
            # Estimate total trade attempts (natural + invalid sells)
            total_attempts = natural_trades + invalid_sells
            if total_attempts > 0:
                natural_trades_pct = (natural_trades / total_attempts) * 100
        
        # Penalty for low natural trades percentage (target: >80%)
        natural_trades_penalty = max(0, (80 - natural_trades_pct) / 80.0)  # 0 when >80%, 1.0 when 0%
        
        # Final metric: worker-specific metric - penalty for low natural trades
        final_metric = worker_metric - (natural_trades_penalty * 0.5)  # 50% weight on natural trades
        
        # Calculate auxiliary metrics for logging
        sharpe = calculate_sharpe_from_portfolio(portfolio_values)
        calmar = calculate_calmar_from_portfolio(portfolio_values)
        
        # Profit Factor
        winning_trades = [r for r in all_rewards if r > 0]
        losing_trades = [r for r in all_rewards if r < 0]
        profit_factor = 0.0
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            profit_factor = sum(winning_trades) / (abs(sum(losing_trades)) + 1e-8)
            
        start_val = portfolio_values[0] if portfolio_values else 0
        end_val = portfolio_values[-1] if portfolio_values else 0

        logger.info(f"""
🎯 TRIAL {trial.number} - Worker {worker_id}
├─ Portfolio: ${start_val:.2f} → ${end_val:.2f}
├─ Natural Trades: {natural_trades_pct:.1f}%
├─ Sharpe: {sharpe:.3f}
├─ Calmar: {calmar:.3f}
└─ Profit Factor: {profit_factor:.3f}
""")
        
        logger.info(
            f"[{worker_id}] Trial {trial.number} Results: "
            f"Worker Metric={worker_metric:.3f}, Natural Trades={natural_trades_pct:.1f}%, "
            f"Invalid Sells={invalid_sells}, Final Score={final_metric:.3f}"
        )
        
        # FIX C: Intelligent pruning - detect model collapse early
        if worker_metric < 0.01:  # < 1% of target (practically zero trades)
            logger.warning(f"[{worker_id}] Trial {trial.number} collapsed - pruning (metric={worker_metric:.3f})")
            raise optuna.TrialPruned()
        
        trial.report(final_metric, step=n_training_steps)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return final_metric
        
    except Exception as e:
        logger.error(f"[{worker_id}] Trial {trial.number} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return -999.0
    
    finally:
        env.close()


def optimize_worker(worker_id: str, n_trials: int = 30):
    """
    CALIBRATED: 30 trials per worker (vs 50) for faster results.
    """
    logger.info(f"{'='*60}")
    logger.info(f"CALIBRATED Optuna for {worker_id}")
    logger.info(f"Trials: {n_trials} | Focused ranges around config.yaml baseline")
    logger.info(f"{'='*60}")
    
    config_loader = ConfigLoader()
    base_config = config_loader.load_config('config/config.yaml')
    
    study = optuna.create_study(
        study_name=f"adan_2.0_{worker_id}_calibrated",
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=500),  # Faster pruning
        storage=f'sqlite:///optuna_{worker_id}_calibrated.db',
        load_if_exists=True
    )
    
    study.optimize(
        lambda trial: objective(trial, worker_id, base_config, n_training_steps=15000),  # Shorter training
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"\n{'='*60}")
    logger.info(f"[{worker_id}] CALIBRATED Optimization Complete!")
    logger.info(f"Best Metric: {best_value:.3f}")
    logger.info(f"Best Parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"{'='*60}\n")
    
    output_path = Path(f'config/optuna_best_{worker_id}_calibrated.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    
    logger.info(f"Best parameters saved to {output_path}")
    
    return study


def main():
    """Run CALIBRATED optimization (120 trials total vs 200)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run 1 trial for testing")
    parser.add_argument("--worker", type=str, help="Optimize single worker (w1/w2/w3/w4)")
    args = parser.parse_args()
    
    if args.worker:
        workers = [args.worker]
    else:
        workers = ['w1', 'w2', 'w3', 'w4']
    
    n_trials_per_worker = 1 if args.dry_run else 20
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# ADAN 2.0 CALIBRATED Optuna Optimization")
    logger.info(f"# Total Trials: {n_trials_per_worker * len(workers)}")
    logger.info(f"# Strategy: Focused search around config.yaml baseline")
    logger.info(f"# Training steps: 15k (vs 25k) for faster iteration")
    if args.dry_run:
        logger.info("# MODE: DRY RUN")
    logger.info(f"{'#'*70}\n")
    
    studies = {}
    for worker_id in workers:
        study = optimize_worker(worker_id, n_trials=n_trials_per_worker)
        studies[worker_id] = study
    
    logger.info(f"\n{'='*70}")
    logger.info("CALIBRATED OPTIMIZATION SUMMARY")
    logger.info(f"{'='*70}")
    for worker_id, study in studies.items():
        logger.info(f"{worker_id}: Best Metric = {study.best_value:.3f}")
    logger.info(f"{'='*70}\n")
    
    logger.info("✅ Calibrated optimization complete! Configs: config/optuna_best_*_calibrated.yaml")


if __name__ == '__main__':
    main()
