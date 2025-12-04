#!/usr/bin/env python3
"""
OPTUNA OPTIMIZATION - 4 WORKERS SÉQUENTIELS
Optimise hyperparamètres risk management + position sizing pour W1/W2/W3/W4

Usage:
    python optuna_optimize_worker.py --worker W1 --trials 20 --steps 3000
    python optuna_optimize_worker.py --worker W2 --trials 20 --steps 3000
    python optuna_optimize_worker.py --worker W3 --trials 20 --steps 3000
    python optuna_optimize_worker.py --worker W4 --trials 20 --steps 3000
"""
import sys
import argparse
import yaml
import numpy as np
import optuna
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Import PPO pour charger les modèles entraînés
try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("WARNING: stable_baselines3 not available, will use random actions")

# Chemins vers les modèles entraînés
MODEL_PATHS = {
    'W1': Path(__file__).parent / 'models' / 'rl_agents' / 'final' / 'w1_final.zip',
    'W2': Path(__file__).parent / 'models' / 'rl_agents' / 'final' / 'w2_final.zip',
    'W3': Path(__file__).parent / 'models' / 'rl_agents' / 'final' / 'w3_final.zip',
    'W4': Path(__file__).parent / 'models' / 'rl_agents' / 'final' / 'w4_final.zip',
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration spécifique par type de worker"""
    name: str
    target_trades_per_day: int
    param_ranges: Dict[str, Tuple[float, float]]
    constraints: Dict[str, float]
    
    
# CONFIGURATIONS DES 4 WORKERS
WORKER_CONFIGS = {
    'W1': WorkerConfig(
        name='Scalper',
        target_trades_per_day=13,  # 11-15 trades/jour
        param_ranges={
            'stop_loss_pct': (0.015, 0.040),      # 1.5% - 4.0%
            'take_profit_pct': (0.030, 0.080),    # 3.0% - 8.0%
            'risk_per_trade_pct': (0.010, 0.025), # 1.0% - 2.5%
            'position_size_pct': (0.10, 0.25),    # 10% - 25%
            'max_concurrent_positions': (3, 5),    # 3-5 positions
            'min_holding_period_steps': (5, 20),  # 5-20 steps
        },
        constraints={
            'min_trades': 5,        # Réduit pour éval courte
            'max_drawdown': 0.25,   # 25% max (assoupli)
            'min_win_rate': 0.0,    # 0% (agent aléatoire, win rate non pertinent)
            'min_sharpe': -5.0,     # Accepter Sharpe négatif (agent aléatoire)
        }
    ),
    'W2': WorkerConfig(
        name='Swing',
        target_trades_per_day=4,  # 3-5 trades/jour
        param_ranges={
            'stop_loss_pct': (0.030, 0.080),      # 3.0% - 8.0%
            'take_profit_pct': (0.060, 0.200),    # 6.0% - 20.0%
            'risk_per_trade_pct': (0.015, 0.035), # 1.5% - 3.5%
            'position_size_pct': (0.15, 0.35),    # 15% - 35%
            'max_concurrent_positions': (1, 2),    # 1-2 positions
            'min_holding_period_steps': (20, 60), # 20-60 steps
        },
        constraints={
            'min_trades': 3,
            'max_drawdown': 0.30,   # Assoupli
            'min_win_rate': 0.0,    # Agent aléatoire
            'min_sharpe': -5.0,     # Accepter négatif
        }
    ),
    'W3': WorkerConfig(
        name='Trend',
        target_trades_per_day=2,  # 1-3 trades/jour
        param_ranges={
            'stop_loss_pct': (0.050, 0.120),      # 5.0% - 12.0%
            'take_profit_pct': (0.100, 0.300),    # 10.0% - 30.0%
            'risk_per_trade_pct': (0.020, 0.040), # 2.0% - 4.0%
            'position_size_pct': (0.20, 0.40),    # 20% - 40%
            'max_concurrent_positions': (1, 2),    # 1-2 positions
            'min_holding_period_steps': (50, 150), # 50-150 steps
        },
        constraints={
            'min_trades': 2,
            'max_drawdown': 0.35,   # Assoupli
            'min_win_rate': 0.0,    # Agent aléatoire
            'min_sharpe': -5.0,     # Accepter négatif
        }
    ),
    'W4': WorkerConfig(
        name='Market Making',
        target_trades_per_day=20,  # 15-25 trades/jour
        param_ranges={
            'stop_loss_pct': (0.008, 0.025),      # 0.8% - 2.5%
            'take_profit_pct': (0.015, 0.050),    # 1.5% - 5.0%
            'risk_per_trade_pct': (0.005, 0.015), # 0.5% - 1.5%
            'position_size_pct': (0.05, 0.15),    # 5% - 15%
            'max_concurrent_positions': (5, 8),    # 5-8 positions
            'min_holding_period_steps': (3, 15),  # 3-15 steps
        },
        constraints={
            'min_trades': 5,        # Réduit
            'max_drawdown': 0.20,
            'min_win_rate': 0.0,    # Agent aléatoire
            'min_sharpe': -5.0,     # Accepter négatif
        }
    ),
}


def create_config_with_params(base_config: Dict, worker_config: WorkerConfig, params: Dict) -> Dict:
    """Crée une config avec les paramètres suggérés par Optuna - BYPASS DBE COMPLÈTEMENT"""
    import copy
    config = copy.deepcopy(base_config)
    
    # ========================================
    # 1. DÉSACTIVER LE DBE COMPLÈTEMENT
    # ========================================
    if 'dbe' not in config:
        config['dbe'] = {}
    config['dbe']['enabled'] = False
    config['dbe']['override_risk_params'] = False
    
    # ========================================
    # 2. INJECTER PARAMS OPTUNA DIRECTEMENT DANS TRADING_RULES
    # ========================================
    if 'trading_rules' not in config:
        config['trading_rules'] = {}
    if 'risk_management' not in config['trading_rules']:
        config['trading_rules']['risk_management'] = {}
    
    # Injecter SL/TP/Risk directement
    config['trading_rules']['risk_management'].update({
        'stop_loss_pct': params['stop_loss_pct'],
        'take_profit_pct': params['take_profit_pct'],
        'risk_per_trade_pct': params['risk_per_trade_pct'],
    })
    
    # ========================================
    # 3. POSITION SIZING
    # ========================================
    if 'position_sizing' not in config['trading_rules']:
        config['trading_rules']['position_sizing'] = {}
    
    config['trading_rules']['position_sizing'].update({
        'position_size_pct': params['position_size_pct'],
        'max_concurrent_positions': int(params['max_concurrent_positions']),
        'min_holding_period_steps': int(params.get('min_holding_period_steps', 10)),
    })
    
    # Aussi dans 'trading' pour compatibilité
    if 'trading' not in config:
        config['trading'] = {}
    config['trading'].update({
        'max_position_size_pct': params['position_size_pct'] * 100,
        'max_active_positions': int(params['max_concurrent_positions']),
    })
    
    # ========================================
    # 4. METTRE À JOUR CAPITAL TIERS AVEC PARAMS OPTUNA
    # ========================================
    if 'capital_tiers' in config.get('trading_rules', {}):
        for tier in config['trading_rules']['capital_tiers']:
            tier['stop_loss_pct'] = params['stop_loss_pct']
            tier['take_profit_pct'] = params['take_profit_pct']
            tier['risk_per_trade_pct'] = params['risk_per_trade_pct']
    
    # ========================================
    # 5. THRESHOLDS PLUS BAS POUR AGENT GAUSSIEN
    # ========================================
    if 'environment' not in config:
        config['environment'] = {}
    config['environment']['action_thresholds'] = {
        '5m': 0.20,   # Baissé de 0.30 pour permettre trades avec actions gaussiennes
        '1h': 0.25,
        '4h': 0.30,
    }
    
    # ========================================
    # 6. FORCE TRADES - Augmenté pour optimisation
    # ========================================
    if 'frequency' not in config['trading_rules']:
        config['trading_rules']['frequency'] = {}
    
    config['trading_rules']['frequency'].update({
        'force_trade_steps': {'5m': 50, '1h': 80, '4h': 150},
        'daily_forced_trades_limit': 50,  # Augmenté significativement
    })
    
    # ========================================
    # 7. LOGGING POUR VÉRIFIER
    # ========================================
    logger.info(f"[OPTUNA CONFIG] DBE disabled, params applied directly:")
    logger.info(f"  SL={params['stop_loss_pct']*100:.2f}%, TP={params['take_profit_pct']*100:.2f}%")
    logger.info(f"  PosSize={params['position_size_pct']*100:.1f}%, MaxPos={params['max_concurrent_positions']}")
    logger.info(f"  Thresholds: 5m=0.20, 1h=0.25, 4h=0.30")
    
    return config


def evaluate_environment(env: MultiAssetChunkedEnv, steps: int, seed: int = 42, 
                         worker_type: str = 'W2', optuna_params: Dict = None,
                         model = None) -> Dict[str, float]:
    """Évalue l'environnement avec le VRAI modèle PPO+CNN+LSTM.
    Les params Optuna sont injectés directement dans le portfolio, bypassing DBE."""
    np.random.seed(seed)
    
    obs = env.reset()
    # Handle tuple return from reset
    if isinstance(obs, tuple):
        obs = obs[0]
    initial_portfolio = env.portfolio.portfolio_value
    
    # ========================================
    # INJECTION PARAMS OPTUNA DANS PORTFOLIO (BYPASS DBE)
    # ========================================
    if optuna_params:
        try:
            if hasattr(env, 'portfolio'):
                env.portfolio.sl_pct = optuna_params.get('stop_loss_pct', 0.02)
                env.portfolio.tp_pct = optuna_params.get('take_profit_pct', 0.04)
                env.portfolio.pos_size_pct = optuna_params.get('position_size_pct', 0.20)
                env.portfolio.risk_per_trade = optuna_params.get('risk_per_trade_pct', 0.02)
                logger.debug(f"[OPTUNA] Injected params: SL={env.portfolio.sl_pct}, TP={env.portfolio.tp_pct}")
        except Exception as e:
            logger.warning(f"[OPTUNA] Could not inject params into portfolio: {e}")
    
    portfolio_values = [initial_portfolio]
    winning_trades = 0
    losing_trades = 0
    
    # Trackers pour compteurs cumulatifs
    prev_opened = 0
    prev_closed = 0
    total_opened = 0
    total_closed = 0
    
    for step in range(steps):
        # ========================================
        # UTILISER LE VRAI MODÈLE PPO AU LIEU D'ACTIONS ALÉATOIRES
        # ========================================
        if model is not None:
            try:
                action, _states = model.predict(obs, deterministic=False)
            except Exception as e:
                logger.warning(f"[MODEL_PREDICT] Error: {e}, using random action")
                action = np.random.normal(0, 0.1, env.action_space.shape)
        else:
            # Fallback: actions aléatoires si pas de modèle
            action = np.random.normal(0, 0.1, env.action_space.shape)
        
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
        
        current_portfolio = env.portfolio.portfolio_value
        portfolio_values.append(current_portfolio)
        
        # Compter trades via info['trades'] (compteur cumulatif fiable)
        if isinstance(info, dict):
            # Utiliser 'trades' qui est le compteur cumulatif total
            total_opened = info.get('trades', 0)
            
            # Compter fermetures via closed_positions list
            closed_this_step = len(info.get('closed_positions', []))
            total_closed += closed_this_step
            
            # Win/Loss tracking via closed_positions (ce sont des DICTS, pas des objets!)
            if 'closed_positions' in info and isinstance(info['closed_positions'], list):
                for pos in info['closed_positions']:
                    if isinstance(pos, dict):
                        pnl = pos.get('pnl', 0)
                        if pnl > 0:
                            winning_trades += 1
                        elif pnl < 0:
                            losing_trades += 1
                    elif hasattr(pos, 'realized_pnl'):
                        if pos.realized_pnl > 0:
                            winning_trades += 1
                        elif pos.realized_pnl < 0:
                            losing_trades += 1
        
        if done:
            break
    
    final_portfolio = portfolio_values[-1]
    
    # Calculate metrics
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[~np.isnan(returns)]  # Remove NaN
    
    if len(returns) < 2 or np.std(returns) == 0:
        sharpe = -999
    else:
        sharpe = np.sqrt(252 * 24) * np.mean(returns) / np.std(returns)  # Hourly bars
    
    # Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Win rate
    total_completed = winning_trades + losing_trades
    win_rate = winning_trades / total_completed if total_completed > 0 else 0.5  # Default 50% if no closes
    
    # Total return
    total_return = (final_portfolio - initial_portfolio) / initial_portfolio
    
    # Calmar Ratio = Annual Return / Max Drawdown
    calmar = total_return / max_drawdown if max_drawdown > 0.001 else 0.0
    
    return {
        'sharpe_ratio': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_opened,
        'trades_ratio': total_opened / (total_closed + 1),  # Avoid div by 0
        'final_portfolio': final_portfolio,
        'initial_portfolio': initial_portfolio,
        'calmar_ratio': calmar,
    }


def calculate_score(metrics: Dict[str, float], worker_config: WorkerConfig, steps: int) -> float:
    """Calcule le score final pour Optuna"""
    sharpe = metrics['sharpe_ratio']
    
    # Rejet immédiat si contraintes violées
    if sharpe == -999 or sharpe < worker_config.constraints['min_sharpe']:
        return -999
    
    if metrics['max_drawdown'] > worker_config.constraints['max_drawdown']:
        return -999
    
    if metrics['total_trades'] < worker_config.constraints['min_trades']:
        return -999
    
    if metrics['win_rate'] < worker_config.constraints['min_win_rate']:
        return -999
    
    # Score = Sharpe avec bonus/malus
    target_trades = worker_config.target_trades_per_day * (steps / 1440)  # steps → jours (1440 steps/jour = 24h × 60min)
    trade_ratio = min(metrics['total_trades'] / target_trades, 1.5)
    
    # Bonus pour atteindre la cible de trades
    bonus = 0.1 * trade_ratio
    
    # Malus léger pour drawdown (même si dans limite)
    dd_penalty = 0.1 * (metrics['max_drawdown'] / worker_config.constraints['max_drawdown'])
    
    final_score = sharpe * (1 + bonus) - dd_penalty
    
    return final_score


def objective(trial: optuna.Trial, base_config: Dict, worker_config: WorkerConfig, eval_steps: int, model=None) -> float:
    """Fonction objectif pour Optuna - utilise le VRAI modèle PPO"""
    
    # Suggest hyperparameters
    params = {}
    for param_name, (low, high) in worker_config.param_ranges.items():
        if param_name == 'max_concurrent_positions':
            params[param_name] = trial.suggest_int(param_name, int(low), int(high))
        elif param_name == 'min_holding_period_steps':
            params[param_name] = trial.suggest_int(param_name, int(low), int(high))
        else:
            params[param_name] = trial.suggest_float(param_name, low, high)
    
    # Create config with suggested params
    config = create_config_with_params(base_config, worker_config, params)
    
    try:
        # Create environment
        env = MultiAssetChunkedEnv(config=config)
        
        # Evaluate avec le VRAI modèle PPO
        worker_type = trial.study.study_name.split('_')[0] if '_' in trial.study.study_name else 'W1'
        metrics = evaluate_environment(env, eval_steps, worker_type=worker_type, optuna_params=params, model=model)
        
        # Calculate score
        score = calculate_score(metrics, worker_config, eval_steps)
        
        # Log trial results
        logger.info(f"Trial {trial.number}: Score={score:.4f}, Sharpe={metrics['sharpe_ratio']:.4f}, "
                   f"Trades={metrics['total_trades']}, DD={metrics['max_drawdown']:.2%}, "
                   f"WinRate={metrics['win_rate']:.2%}")
        
        # Store metrics in trial
        trial.set_user_attr('sharpe', metrics['sharpe_ratio'])
        trial.set_user_attr('trades', metrics['total_trades'])
        trial.set_user_attr('drawdown', metrics['max_drawdown'])
        trial.set_user_attr('win_rate', metrics['win_rate'])
        
        return score
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return -999


def main():
    parser = argparse.ArgumentParser(description='Optimize worker hyperparameters with Optuna')
    parser.add_argument('--worker', type=str, required=True, choices=['W1', 'W2', 'W3', 'W4'],
                       help='Worker to optimize')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of trials')
    parser.add_argument('--steps', type=int, default=3000,
                       help='Evaluation steps per trial')
    parser.add_argument('--output-dir', type=str, default='optuna_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load base config
    with open('config/config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Get worker config
    worker_config = WORKER_CONFIGS[args.worker]
    
    logger.info(f"=" * 80)
    logger.info(f"OPTUNA OPTIMIZATION - {args.worker} ({worker_config.name})")
    logger.info(f"=" * 80)
    logger.info(f"Trials: {args.trials}")
    logger.info(f"Evaluation steps: {args.steps}")
    logger.info(f"Target trades/day: {worker_config.target_trades_per_day}")
    logger.info(f"Constraints: {worker_config.constraints}")
    logger.info(f"")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create study
    study_name = f"{args.worker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_name = f"sqlite:///{output_dir}/{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=500, interval_steps=100)
    )
    
    # ========================================
    # CHARGER LE VRAI MODÈLE PPO+CNN+LSTM
    # ========================================
    model = None
    if PPO_AVAILABLE:
        model_path = MODEL_PATHS.get(args.worker)
        if model_path and model_path.exists():
            try:
                logger.info(f"")
                logger.info(f"🧠 LOADING TRAINED MODEL: {model_path}")
                model = PPO.load(str(model_path))
                logger.info(f"✅ Model loaded successfully!")
                logger.info(f"   Policy: {type(model.policy).__name__}")
                logger.info(f"")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                logger.warning(f"⚠️  Falling back to random actions")
                model = None
        else:
            logger.warning(f"⚠️  Model not found at {model_path}, using random actions")
    else:
        logger.warning(f"⚠️  stable_baselines3 not available, using random actions")
    
    # Optimize avec le vrai modèle
    study.optimize(
        lambda trial: objective(trial, base_config, worker_config, args.steps, model=model),
        n_trials=args.trials,
        show_progress_bar=True,
    )
    
    # Results
    logger.info(f"")
    logger.info(f"=" * 80)
    logger.info(f"OPTIMIZATION COMPLETE - {args.worker}")
    logger.info(f"=" * 80)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best score: {study.best_value:.4f}")
    logger.info(f"Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    
    logger.info(f"")
    logger.info(f"Best trial metrics:")
    logger.info(f"  Sharpe: {study.best_trial.user_attrs['sharpe']:.4f}")
    logger.info(f"  Trades: {study.best_trial.user_attrs['trades']}")
    logger.info(f"  Drawdown: {study.best_trial.user_attrs['drawdown']:.2%}")
    logger.info(f"  Win Rate: {study.best_trial.user_attrs['win_rate']:.2%}")
    
    # Save best params
    output_file = output_dir / f"{args.worker}_best_params.yaml"
    with open(output_file, 'w') as f:
        yaml.dump({
            'worker': args.worker,
            'score': float(study.best_value),
            'parameters': study.best_params,
            'metrics': {
                'sharpe': float(study.best_trial.user_attrs['sharpe']),
                'trades': int(study.best_trial.user_attrs['trades']),
                'drawdown': float(study.best_trial.user_attrs['drawdown']),
                'win_rate': float(study.best_trial.user_attrs['win_rate']),
            }
        }, f, default_flow_style=False)
    
    logger.info(f"")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Study database: {storage_name}")
    logger.info(f"=" * 80)


if __name__ == "__main__":
    main()
