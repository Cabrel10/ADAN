#!/usr/bin/env python3
"""
PHASE 2: OPTIMISATION HYPERPARAMÈTRES PPO
Optimise learning_rate, n_steps, batch_size, gamma, gae_lambda, etc.

Version corrigée avec métriques fiables.

Usage:
    python optuna_optimize_ppo.py --worker W1 --trials 20 --steps 5000
"""
import sys
import argparse
import yaml
import numpy as np
import optuna
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
import warnings
import traceback
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv,
)
from adan_trading_bot.optuna_evaluation import (
    evaluate_ppo_params_robust,
)

# Import PPO
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("ERROR: stable_baselines3 not available!")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PARAMÈTRES DE TRADING PHASE 1 (Fixés, utilisés comme base)
# ============================================================================

# Paramètres corrigés basés sur les meilleures valeurs Phase 1
BEST_TRADING_PARAMS = {
    'W1': {  # Scalper - EXCELLENT (Score 51.46)
        'stop_loss_pct': 0.0253,
        'take_profit_pct': 0.0321,
        'position_size_pct': 0.1121,
        'risk_per_trade_pct': 0.01,
        'max_concurrent_positions': 3,
        'min_holding_period_steps': 5,
    },
    'W2': {  # Swing Trader - MODÉRÉ (Amélioré)
        'stop_loss_pct': 0.025,       # 2.5% SL (entre W1 et W3)
        'take_profit_pct': 0.050,     # 5% TP (ratio 2.0)
        'position_size_pct': 0.25,    # 25% position (augmenté de 12%)
        'risk_per_trade_pct': 0.015,
        'max_concurrent_positions': 3,  # Augmenté de 2 à 3
        'min_holding_period_steps': 10,  # Réduit de 20 à 10 (plus de trades)
    },
    'W3': {  # Position Trader - AGRESSIF (RE-OPTIMISÉ - Recommandations appliquées)
        'stop_loss_pct': 0.10,        # 10% SL (agressif, augmenté de 8%)
        'take_profit_pct': 0.18,      # 18% TP (ratio 1.8, augmenté de 15%)
        'position_size_pct': 0.50,    # 50% position (très agressif, augmenté de 45%)
        'risk_per_trade_pct': 0.03,   # 3% risque par trade (augmenté de 2.5%)
        'max_concurrent_positions': 2,  # 2 positions (diversification)
        'min_holding_period_steps': 40,  # Réduit de 50 à 40 (plus d'opportunités)
    },
    'W4': {  # Day Trading - SHARPE OPTIMIZED (Complètement refondu)
        'stop_loss_pct': 0.012,       # 1.2% SL (serré pour Sharpe)
        'take_profit_pct': 0.020,     # 2.0% TP (ratio 1.67)
        'position_size_pct': 0.20,    # 20% position (variable)
        'risk_per_trade_pct': 0.012,  # 1.2% risque par trade
        'max_concurrent_positions': 4,  # 4 positions (diversification)
        'min_holding_period_steps': 5,  # 5 steps (court terme)
    },
}


class MetricsCallback(BaseCallback):
    """Callback pour collecter des métriques pendant l'entraînement"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.portfolio_values = []
        self.trades = []
        
    def _on_step(self) -> bool:
        # Récupérer la valeur du portfolio
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
            if hasattr(env, 'portfolio'):
                self.portfolio_values.append(env.portfolio.portfolio_value)
        return True


def load_base_config() -> Dict:
    """Charge la config de base"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_env_with_trading_params(base_config: Dict, worker: str) -> MultiAssetChunkedEnv:
    """Crée l'environnement avec les meilleurs trading params"""
    import copy
    config = copy.deepcopy(base_config)
    
    # Désactiver DBE COMPLÈTEMENT
    config['dbe'] = {
        'enabled': False,
        'override_risk_params': False,
    }
    
    # Utiliser les paramètres hardcodés (fiables)
    trading_params = BEST_TRADING_PARAMS[worker]
    
    if 'trading_rules' not in config:
        config['trading_rules'] = {}
    
    config['trading_rules']['risk_management'] = {
        'stop_loss_pct': trading_params['stop_loss_pct'],
        'take_profit_pct': trading_params['take_profit_pct'],
        'risk_per_trade_pct': trading_params['risk_per_trade_pct'],
    }
    
    config['trading_rules']['position_sizing'] = {
        'position_size_pct': trading_params['position_size_pct'],
        'max_concurrent_positions': trading_params['max_concurrent_positions'],
        'min_holding_period_steps': trading_params['min_holding_period_steps'],
    }
    
    # Force trade config plus agressive pour l'évaluation
    config['trading_rules']['frequency'] = {
        'force_trade_steps': {'5m': 15, '1h': 30, '4h': 60},
        'daily_max_total': 500,  # Augmenter limite journalière
        'daily_max_by_tf': {
            '5m': 200,
            '1h': 200,
            '4h': 200,
        },
    }
    
    # Augmenter le cap de trades forcés pour l'évaluation
    config['trading_rules']['daily_max_forced_trades'] = 500
    
    env = MultiAssetChunkedEnv(config=config)
    
    # INJECTION DIRECTE dans portfolio
    if hasattr(env, 'portfolio'):
        env.portfolio.sl_pct = trading_params['stop_loss_pct']
        env.portfolio.tp_pct = trading_params['take_profit_pct']
        env.portfolio.pos_size_pct = trading_params['position_size_pct']
        env.portfolio.risk_per_trade = trading_params['risk_per_trade_pct']
    
    return env


def suggest_ppo_params(trial: optuna.Trial) -> Dict:
    """Suggère les hyperparamètres PPO"""
    params = {}
    
    # Learning rate (loguniform)
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    
    # N steps (puissance de 2)
    n_steps_exp = trial.suggest_int('n_steps_exp', 9, 11)  # 512 à 2048
    params['n_steps'] = 2 ** n_steps_exp
    
    # Batch size
    batch_exp = trial.suggest_int('batch_exp', 5, 7)  # 32 à 128
    params['batch_size'] = min(2 ** batch_exp, params['n_steps'])
    
    # Autres hyperparamètres
    params['n_epochs'] = trial.suggest_int('n_epochs', 5, 15)
    params['gamma'] = trial.suggest_float('gamma', 0.97, 0.995)
    params['gae_lambda'] = trial.suggest_float('gae_lambda', 0.92, 0.99)
    params['clip_range'] = trial.suggest_float('clip_range', 0.15, 0.35)
    params['ent_coef'] = trial.suggest_float('ent_coef', 0.001, 0.02)
    params['vf_coef'] = trial.suggest_float('vf_coef', 0.3, 0.8)
    params['max_grad_norm'] = trial.suggest_float('max_grad_norm', 0.4, 0.8)
    
    return params


def calculate_metrics(portfolio_values: List[float], trades_info: List[Dict]) -> Dict[str, float]:
    """Calcule les métriques de performance de manière robuste"""
    
    pv = np.array(portfolio_values)
    
    if len(pv) < 10:
        return {
            'sharpe_ratio': -999,
            'max_drawdown': 1.0,
            'win_rate': 0.0,
            'total_return': -1.0,
            'total_trades': 0,
            'profit_factor': 0.0,
        }
    
    # Calcul des rendements
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-8)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 2:
        returns = np.zeros(10)
    
    # Sharpe Ratio (annualisé)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    if std_ret < 1e-10:
        # Pas de variance -> utiliser le return total comme proxy
        total_ret = (pv[-1] - pv[0]) / max(pv[0], 1e-8)
        sharpe = total_ret * 10  # Scale
    else:
        sharpe = np.sqrt(252 * 24 * 12) * mean_ret / std_ret  # 5min bars
    
    # Limiter Sharpe à des valeurs raisonnables
    sharpe = np.clip(sharpe, -50, 50)
    
    # Max Drawdown
    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / np.maximum(peak, 1e-8)
    max_dd = np.max(drawdown)
    
    # Total Return
    total_return = (pv[-1] - pv[0]) / max(pv[0], 1e-8)
    
    # Win/Loss depuis trades_info
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    for trade in trades_info:
        pnl = trade.get('pnl', 0)
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            losses += 1
            gross_loss += abs(pnl)
    
    total_trades = wins + losses
    win_rate = wins / max(total_trades, 1)
    profit_factor = gross_profit / max(gross_loss, 1e-8) if gross_loss > 0 else 10.0
    
    return {
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'total_return': float(total_return),
        'total_trades': int(total_trades),
        'profit_factor': float(min(profit_factor, 100)),
    }


def evaluate_ppo_params(
    env: MultiAssetChunkedEnv,
    ppo_params: Dict,
    training_steps: int,
    eval_steps: int = 2000,
) -> Dict[str, float]:
    """
    Wrapper pour la fonction robuste centralisée.

    Utilise evaluate_ppo_params_robust pour garantir une collecte
    fiable des métriques avec gardes-fous PPO.
    """
    return evaluate_ppo_params_robust(
        env=env,
        ppo_params=ppo_params,
        training_steps=training_steps,
        eval_steps=eval_steps,
    )


def compute_score(metrics: Dict[str, float]) -> float:
    """Calcule le score composite"""
    
    sharpe = metrics['sharpe_ratio']
    if sharpe == -999 or not np.isfinite(sharpe):
        return -999
    
    max_dd = metrics['max_drawdown']
    win_rate = metrics['win_rate']
    total_return = metrics['total_return']
    profit_factor = metrics['profit_factor']
    
    # Score composite
    # Pénaliser fortement le drawdown > 25%
    dd_penalty = max(0, 1 - max_dd * 2)
    
    # Bonus pour win rate > 50%
    wr_bonus = 1 + (win_rate - 0.5) * 0.5
    
    # Bonus pour profit factor
    pf_bonus = min(profit_factor / 2, 2)
    
    # Score final
    score = sharpe * dd_penalty * wr_bonus * pf_bonus
    
    # Bonus si return positif
    if total_return > 0:
        score *= (1 + total_return * 0.5)
    
    return float(np.clip(score, -999, 100))


def objective(
    trial: optuna.Trial,
    base_config: Dict,
    worker: str,
    training_steps: int,
    eval_steps: int
) -> float:
    """Fonction objectif Optuna"""

    try:
        # Suggérer hyperparamètres
        ppo_params = suggest_ppo_params(trial)
        # Créer environnement
        env = create_env_with_trading_params(base_config, worker)
        
        # Évaluer
        metrics = evaluate_ppo_params(env, ppo_params, training_steps, eval_steps)
        
        # Score
        score = compute_score(metrics)
        
        # Log
        logger.info(f"Trial {trial.number}: Score={score:.2f}, Sharpe={metrics['sharpe_ratio']:.2f}, "
                   f"DD={metrics['max_drawdown']:.1%}, WR={metrics['win_rate']:.1%}, "
                   f"Trades={metrics['total_trades']}")
        
        # Store
        trial.set_user_attr('sharpe', metrics['sharpe_ratio'])
        trial.set_user_attr('drawdown', metrics['max_drawdown'])
        trial.set_user_attr('win_rate', metrics['win_rate'])
        trial.set_user_attr('return', metrics['total_return'])
        trial.set_user_attr('trades', metrics['total_trades'])
        trial.set_user_attr('profit_factor', metrics['profit_factor'])
        
        return score
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()
        return -999


def run_optimization(worker: str, n_trials: int, training_steps: int, 
                     eval_steps: int, output_dir: Path) -> Dict:
    """Exécute l'optimisation pour un worker"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 2: PPO OPTIMIZATION - {worker}")
    logger.info(f"{'='*80}")
    logger.info(f"Trading params: {BEST_TRADING_PARAMS[worker]}")
    logger.info(f"Trials: {n_trials}, Train steps: {training_steps}, Eval steps: {eval_steps}")
    
    # Charger config
    base_config = load_base_config()
    
    # Créer study
    study_name = f"{worker}_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{output_dir}/{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3),
    )
    
    # Optimiser
    study.optimize(
        lambda trial: objective(trial, base_config, worker, training_steps, eval_steps),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    # Résultats
    best_trial = study.best_trial
    best_params = study.best_params
    
    # Convertir params
    ppo_params = {}
    for k, v in best_params.items():
        if k == 'n_steps_exp':
            ppo_params['n_steps'] = 2 ** v
        elif k == 'batch_exp':
            ppo_params['batch_size'] = 2 ** v
        else:
            ppo_params[k] = v
    
    result = {
        'worker': worker,
        'phase': 'Phase 2 - PPO Hyperparams',
        'score': float(study.best_value),
        'ppo_parameters': ppo_params,
        'trading_parameters': BEST_TRADING_PARAMS[worker],
        'metrics': {
            'sharpe': best_trial.user_attrs.get('sharpe'),
            'drawdown': best_trial.user_attrs.get('drawdown'),
            'win_rate': best_trial.user_attrs.get('win_rate'),
            'total_return': best_trial.user_attrs.get('return'),
            'trades': best_trial.user_attrs.get('trades'),
            'profit_factor': best_trial.user_attrs.get('profit_factor'),
        }
    }
    
    # Sauvegarder
    output_file = output_dir / f"{worker}_ppo_best_params.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 2 COMPLETE - {worker}")
    logger.info(f"{'='*80}")
    logger.info(f"Best score: {study.best_value:.4f}")
    logger.info(f"Best PPO params: {ppo_params}")
    logger.info(f"Saved to: {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Optuna PPO Hyperparameter Optimization')
    parser.add_argument('--worker', type=str, required=True, 
                       choices=['W1', 'W2', 'W3', 'W4', 'ALL'])
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--steps', type=int, default=5000, help='Training steps per trial')
    parser.add_argument('--eval-steps', type=int, default=2000, help='Evaluation steps')
    parser.add_argument('--output-dir', type=str, default='optuna_results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Workers à optimiser
    if args.worker == 'ALL':
        workers = ['W1', 'W2', 'W3', 'W4']
    else:
        workers = [args.worker]
    
    results = {}
    for worker in workers:
        result = run_optimization(
            worker=worker,
            n_trials=args.trials,
            training_steps=args.steps,
            eval_steps=args.eval_steps,
            output_dir=output_dir
        )
        results[worker] = result
    
    # Résumé final
    print("\n" + "="*80)
    print("PHASE 2 SUMMARY - ALL WORKERS")
    print("="*80)
    for w, r in results.items():
        print(f"{w}: Score={r['score']:.2f}, Sharpe={r['metrics']['sharpe']:.2f}, "
              f"DD={r['metrics']['drawdown']:.1%}")
    print("="*80)


if __name__ == "__main__":
    main()
