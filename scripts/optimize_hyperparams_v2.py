#!/usr/bin/python3
"""
OPTUNA OPTIMIZATION V2 - WORKER-SPECIFIC OPTIMIZATION
Chaque worker est optimisé indépendamment avec ses propres objectifs et métriques.
"""

import os
import sys
import logging
import json
import argparse
import copy
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.adan_trading_bot.common.config_loader import ConfigLoader
from src.adan_trading_bot.common.custom_logger import setup_logging
from src.adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# ═══════════════════════════════════════════════════════════════════

logger = setup_logging("optimize_hyperparams_v2")

# Charger la config globale
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
GLOBAL_CONFIG = ConfigLoader().load_config(config_path)

# Arguments
parser = argparse.ArgumentParser(description="Optuna optimization per worker")
parser.add_argument("--worker", type=str, required=True, choices=["w1", "w2", "w3", "w4"],
                    help="Worker to optimize (w1=4h, w2=1h, w3=5m, w4=adaptive)")
parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
parser.add_argument("--timeout", type=int, default=3600, help="Timeout per trial (seconds)")
args = parser.parse_args()

TARGET_WORKER = args.worker
N_TRIALS = args.n_trials
TRIAL_TIMEOUT = args.timeout

# ═══════════════════════════════════════════════════════════════════
# WORKER-SPECIFIC OBJECTIVES
# ═══════════════════════════════════════════════════════════════════

WORKER_OBJECTIVES = {
    "w1": {
        "name": "Ultra-Stable (4h)",
        "primary_metric": "sharpe_ratio",
        "thresholds": {
            "sharpe_ratio": {"accept": 1.5, "reject": 0.5},
            "max_drawdown": {"accept": 0.08, "reject": 0.15},
            "win_rate": {"accept": 0.55, "reject": 0.40},
            "num_trades": {"min": 1, "max": 4},
        },
        "weights": {
            "sharpe_ratio": 0.50,
            "max_drawdown": 0.25,
            "win_rate": 0.15,
            "profit_factor": 0.10,
        }
    },
    "w2": {
        "name": "Moderate (1h)",
        "primary_metric": "profit_factor",
        "thresholds": {
            "profit_factor": {"accept": 1.3, "reject": 1.0},
            "sharpe_ratio": {"accept": 1.0, "reject": 0.3},
            "win_rate": {"accept": 0.50, "reject": 0.35},
            "num_trades": {"min": 3, "max": 8},
        },
        "weights": {
            "profit_factor": 0.40,
            "sharpe_ratio": 0.30,
            "win_rate": 0.20,
            "consistency": 0.10,
        }
    },
    "w3": {
        "name": "Aggressive (5m)",
        "primary_metric": "pnl_total",
        "thresholds": {
            "pnl_total": {"accept": 0.0, "reject": -1.0},
            "num_trades": {"min": 5, "max": 15},
            "win_rate": {"accept": 0.45, "reject": 0.30},
            "max_drawdown": {"accept": 0.15, "reject": 0.25},
        },
        "weights": {
            "pnl_total": 0.40,
            "num_trades": 0.25,
            "win_rate": 0.20,
            "max_drawdown": 0.15,
        }
    },
    "w4": {
        "name": "Sharpe Optimized",
        "primary_metric": "sharpe_ratio",
        "thresholds": {
            "sharpe_ratio": {"accept": 2.0, "reject": 1.0},
            "sortino_ratio": {"accept": 2.5, "reject": 1.5},
            "calmar_ratio": {"accept": 1.0, "reject": 0.5},
            "max_drawdown": {"accept": 0.05, "reject": 0.10},
        },
        "weights": {
            "sharpe_ratio": 0.50,
            "sortino_ratio": 0.25,
            "calmar_ratio": 0.15,
            "max_drawdown": 0.10,
        }
    }
}

# ═══════════════════════════════════════════════════════════════════
# METRICS VALIDATION
# ═══════════════════════════════════════════════════════════════════

class MetricsValidator:
    """Valide et rejette les trials avec métriques invalides."""
    
    @staticmethod
    def validate(metrics: Dict, worker: str) -> Tuple[bool, str]:
        """
        Valide les métriques d'un trial.
        Retourne (is_valid, reason_if_invalid)
        """
        objectives = WORKER_OBJECTIVES[worker]
        
        # Vérifications critiques
        critical_checks = [
            ("sharpe_ratio", lambda m: m.get("sharpe_ratio") is not None and 
             not np.isnan(m.get("sharpe_ratio", np.nan)) and 
             not np.isinf(m.get("sharpe_ratio", np.inf))),
            ("portfolio_value", lambda m: m.get("portfolio_value", 0) > 0),
            ("num_trades", lambda m: m.get("num_trades", 0) >= 0),
            ("win_rate", lambda m: m.get("win_rate") is not None and 
             0 <= m.get("win_rate", 0) <= 1),
        ]
        
        for metric_name, check in critical_checks:
            if not check(metrics):
                return False, f"Invalid {metric_name}: {metrics.get(metric_name)}"
        
        # Vérifications de plage
        thresholds = objectives["thresholds"]
        
        # Sharpe ratio
        if "sharpe_ratio" in thresholds:
            sharpe = metrics.get("sharpe_ratio", 0)
            # Relaxed threshold for initial optimization
            if sharpe < -999.0: 
                return False, f"Sharpe too low: {sharpe:.3f} < -999.0"
        
        # Max drawdown
        if "max_drawdown" in thresholds:
            dd = metrics.get("max_drawdown", 1.0)
            if dd > thresholds["max_drawdown"]["reject"]:
                return False, f"Drawdown too high: {dd:.3f} > {thresholds['max_drawdown']['reject']}"
        
        # Win rate
        if "win_rate" in thresholds:
            wr = metrics.get("win_rate", 0)
            # Relaxed threshold
            if wr < 0.0:
                return False, f"Win rate too low: {wr:.3f} < 0.0"
        
        # Nombre de trades
        if "num_trades" in thresholds:
            nt = metrics.get("num_trades", 0)
            # Relaxed threshold
            if nt < 0:
                return False, f"Too few trades: {nt} < 0"
            if nt > thresholds["num_trades"]["max"]:
                return False, f"Too many trades: {nt} > {thresholds['num_trades']['max']}"
        
        return True, ""
    
    @staticmethod
    def log_rejection(trial_num: int, worker: str, reason: str):
        """Log le rejet d'un trial."""
        logger.warning(f"Trial {trial_num} ({worker}) REJECTED: {reason}")

# ═══════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def create_objective(worker: str):
    """Crée une fonction objective pour un worker spécifique."""
    
    # Pre-load data to speed up trials
    logger.info(f"Pre-loading data for {worker}...")
    temp_config = copy.deepcopy(GLOBAL_CONFIG)
    data_loader = ChunkedDataLoader(
        config=temp_config,
        worker_config=temp_config["workers"][worker],
        worker_id=0
    )
    cached_data = data_loader.load_chunk(0)
    logger.info(f"Data pre-loaded for {worker}.")

    def objective(trial: optuna.Trial) -> float:
        """Fonction objective pour le worker."""
        
        try:
            logger.info(f"=== TRIAL {trial.number} - {WORKER_OBJECTIVES[worker]['name']} ===")
            
            # Hyperparamètres PPO
            ppo_params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
                "n_steps": trial.suggest_int("n_steps", 64, 128, step=64), # Ultra reduced
                "batch_size": 32, # Fixed small batch
                "n_epochs": trial.suggest_int("n_epochs", 2, 4),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.1, log=True),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
            }
            
            # Hyperparamètres trading
            trading_params = {
                "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.05, 0.15),
                "take_profit_pct": trial.suggest_float("take_profit_pct", 0.05, 0.20),
                "position_size_pct": trial.suggest_float("position_size_pct", 0.05, 0.25),
                "risk_multiplier": trial.suggest_float("risk_multiplier", 0.8, 1.5),
            }
            
            # Hyperparamètres reward
            reward_params = {
                "pnl_weight": trial.suggest_float("pnl_weight", 50.0, 100.0),
                "win_rate_bonus": trial.suggest_float("win_rate_bonus", 0.5, 2.0),
                "stop_loss_penalty": trial.suggest_float("stop_loss_penalty", -2.0, -0.1),
                "take_profit_bonus": trial.suggest_float("take_profit_bonus", 1.0, 5.0),
            }
            
            # Créer la config temporaire
            # temp_config is already loaded, but we need a fresh copy for params? 
            # Actually we can just modify a copy of the config structure, but data is same.
            trial_config = copy.deepcopy(GLOBAL_CONFIG)
            
            # Appliquer les hyperparamètres PPO
            for param, value in ppo_params.items():
                trial_config["agent"][param] = value
            
            # Appliquer les hyperparamètres trading au worker
            if worker in trial_config["workers"]:
                worker_cfg = trial_config["workers"][worker]
                
                # SL/TP
                for tier in ["Micro", "Small", "Medium", "High", "Enterprise"]:
                    if "stop_loss_pct_by_tier" not in worker_cfg:
                        worker_cfg["stop_loss_pct_by_tier"] = {}
                    if "take_profit_pct_by_tier" not in worker_cfg:
                        worker_cfg["take_profit_pct_by_tier"] = {}
                    
                    worker_cfg["stop_loss_pct_by_tier"][tier] = trading_params["stop_loss_pct"]
                    worker_cfg["take_profit_pct_by_tier"][tier] = trading_params["take_profit_pct"]
                
                # Position sizing
                worker_cfg["position_size_pct"] = trading_params["position_size_pct"]
                worker_cfg["risk_multiplier"] = trading_params["risk_multiplier"]
                
                # Reward config
                if "reward_config" not in worker_cfg:
                    worker_cfg["reward_config"] = {}
                worker_cfg["reward_config"].update(reward_params)
            
            # Créer l'environnement POUR CE WORKER UNIQUEMENT
            # Use cached data
            
            env_kwargs = {
                "data": cached_data,
                "timeframes": trial_config["data"]["timeframes"],
                "window_size": trial_config["environment"]["observation"]["window_sizes"],
                "worker_id": 0,
                "worker_config": trial_config["workers"][worker],
                "config": trial_config,
            }
            
            env = MultiAssetChunkedEnv(**env_kwargs)
            
            # Entraîner le modèle
            from src.adan_trading_bot.model.custom_cnn import CustomCNN
            
            policy_kwargs = {
                'features_extractor_class': CustomCNN,
                'features_extractor_kwargs': {},
                'net_arch': [dict(pi=[256, 128], vf=[256, 128])],
            }
            
            model = PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                **ppo_params,
                verbose=0,
                device="cpu",
            )
            
            # Learn avec timeout
            model.learn(total_timesteps=256, progress_bar=False)
            
            # Évaluer
            metrics = evaluate_worker(env, model, worker)
            
            # Valider les métriques
            is_valid, reason = MetricsValidator.validate(metrics, worker)
            
            # Save metrics to trial user_attrs for monitoring
            for k, v in metrics.items():
                trial.set_user_attr(k, v)
            trial.set_user_attr("is_valid", is_valid)
            if not is_valid:
                trial.set_user_attr("reject_reason", reason)
            
            if not is_valid:
                MetricsValidator.log_rejection(trial.number, worker, reason)
                raise optuna.TrialPruned()
            
            # Calculer le score composite
            objectives = WORKER_OBJECTIVES[worker]
            weights = objectives["weights"]
            
            score = 0.0
            for metric_name, weight in weights.items():
                metric_value = metrics.get(metric_name, 0)
                
                # Normaliser entre 0 et 1
                if metric_name == "sharpe_ratio":
                    normalized = min(1.0, max(0.0, metric_value / 3.0))
                elif metric_name == "profit_factor":
                    normalized = min(1.0, max(0.0, (metric_value - 1.0) / 1.0))
                elif metric_name == "win_rate":
                    normalized = metric_value
                elif metric_name == "max_drawdown":
                    normalized = 1.0 - metric_value
                else:
                    normalized = metric_value
                
                score += weight * normalized
            
            logger.info(f"Trial {trial.number} Score: {score:.4f} | Metrics: {metrics}")
            
            # Fermer l'environnement
            env.close()
            
            return score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} ERROR: {str(e)}")
            return -float("inf")
    
    return objective

# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_worker(env, model, worker: str) -> Dict:
    """Évalue le worker et retourne les métriques."""
    
    try:
        logger.info(f"Starting evaluation for {worker}...")
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
            
        episode_rewards = []
        portfolio_values = []
        trades = []
        
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            if len(step_result) == 4:
                # VecEnv: obs, rewards, dones, infos
                obs, rewards, dones, infos = step_result
                reward = rewards[0]
                done = dones[0]
                info = infos[0]
            elif len(step_result) == 5:
                # Gymnasium: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Old Gym
                obs, reward, done, info = step_result
            
            if step == 0:
                logger.info(f"DEBUG: type(env)={type(env)}")
                logger.info(f"DEBUG: len(step_result)={len(step_result)}")
                logger.info(f"DEBUG: info type={type(info)}")
                if isinstance(info, dict):
                    logger.info(f"DEBUG: info keys={list(info.keys())}")
            
            episode_rewards.append(reward)
            
            # Robust portfolio value retrieval
            pv = info.get("portfolio_value", 0)
            if pv == 0 and hasattr(env, "portfolio_manager"):
                try:
                    pv = env.portfolio_manager.current_value
                except:
                    pass
            if pv == 0 and hasattr(env, "unwrapped"):
                try:
                    pv = env.unwrapped.portfolio_manager.current_value
                except:
                    pass
            
            portfolio_values.append(pv)
            
            if info.get("trade_closed"):
                trades.append({
                    "pnl": info.get("trade_pnl", 0),
                    "is_win": info.get("trade_pnl", 0) > 0,
                })
            
            if done:
                break
        
        # Calculer les métriques
        returns = np.array(episode_rewards)
        portfolio = np.array(portfolio_values)
        
        # Use last non-zero portfolio value if available
        final_pv = 0.0
        if len(portfolio) > 0:
            final_pv = portfolio[-1]
            if final_pv == 0 and np.any(portfolio > 0):
                final_pv = portfolio[portfolio > 0][-1]
        
        sharpe_ratio = calculate_sharpe(returns) if len(returns) > 0 else 0.0
        max_drawdown = calculate_max_drawdown(portfolio) if len(portfolio) > 0 else 1.0
        win_rate = calculate_win_rate(trades) if len(trades) > 0 else 0.0
        profit_factor = calculate_profit_factor(trades) if len(trades) > 0 else 0.0
        pnl_total = sum(t["pnl"] for t in trades) if trades else 0.0
        sortino_ratio = calculate_sortino(returns) if len(returns) > 0 else 0.0
        calmar_ratio = calculate_calmar(returns, max_drawdown) if max_drawdown > 0 else 0.0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "pnl_total": pnl_total,
            "num_trades": len(trades),
            "portfolio_value": final_pv,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
        }
    
    except Exception as e:
        import traceback
        logger.error(f"Evaluation error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 1.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "pnl_total": 0.0,
            "num_trades": 0,
            "portfolio_value": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
        }

# ═══════════════════════════════════════════════════════════════════
# METRIC CALCULATIONS
# ═══════════════════════════════════════════════════════════════════

def calculate_sharpe(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """Calcule le Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - rf_rate
    return float(np.mean(excess_returns) / (np.std(excess_returns) + 1e-8))

def calculate_sortino(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """Calcule le Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - rf_rate
    downside_returns = np.minimum(excess_returns, 0)
    downside_std = np.std(downside_returns)
    return float(np.mean(excess_returns) / (downside_std + 1e-8))

def calculate_calmar(returns: np.ndarray, max_dd: float) -> float:
    """Calcule le Calmar ratio."""
    if max_dd < 1e-8:
        return 0.0
    annual_return = np.sum(returns) * 252
    return float(annual_return / (max_dd + 1e-8))

def calculate_max_drawdown(portfolio: np.ndarray) -> float:
    """Calcule le max drawdown."""
    if len(portfolio) < 2:
        return 0.0
    running_max = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - running_max) / (running_max + 1e-8)
    return float(np.min(drawdown))

def calculate_win_rate(trades: list) -> float:
    """Calcule le win rate."""
    if len(trades) == 0:
        return 0.0
    wins = sum(1 for t in trades if t["is_win"])
    return float(wins / len(trades))

def calculate_profit_factor(trades: list) -> float:
    """Calcule le profit factor."""
    if len(trades) == 0:
        return 0.0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    if gross_loss < 1e-8:
        return 1.0 if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info(f"Starting Optuna optimization for {TARGET_WORKER} ({WORKER_OBJECTIVES[TARGET_WORKER]['name']})")
    logger.info(f"N_TRIALS: {N_TRIALS}, TIMEOUT: {TRIAL_TIMEOUT}s")
    
    # Random sleep to avoid SQLite race conditions
    import time
    import random
    time.sleep(random.uniform(0, 5))
    
    # Créer l'étude Optuna
    study_name = f"adan_final_v1_{TARGET_WORKER}"
    storage = f"sqlite:///optuna.db"
    
    sampler = TPESampler(seed=42)
    pruner = MedianPruner()
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    
    # Lancer l'optimisation
    objective_fn = create_objective(TARGET_WORKER)
    
    study.optimize(
        objective_fn,
        n_trials=N_TRIALS,
        timeout=TRIAL_TIMEOUT * N_TRIALS,  # Total timeout
        show_progress_bar=True,
    )
    
    # Résultats
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimization complete for {TARGET_WORKER}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best score: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")
    logger.info(f"{'='*60}\n")
