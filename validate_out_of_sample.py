#!/usr/bin/env python3
"""
PHASE 3: VALIDATION OUT-OF-SAMPLE
Valide les meilleurs paramètres sur des données non vues (derniers chunks).

Usage:
    python validate_out_of_sample.py --worker W1 --chunks 2
"""
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any
import copy

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Import PPO
try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("WARNING: stable_baselines3 not available!")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_best_params(worker: str) -> Dict:
    """Charge les meilleurs paramètres (Trading + PPO)"""
    # 1. Chercher params PPO (Phase 2)
    ppo_file = Path(f"optuna_results/{worker}_ppo_best_params.yaml")
    if ppo_file.exists():
        logger.info(f"Loading Phase 2 params from {ppo_file}")
        with open(ppo_file, 'r') as f:
            return yaml.safe_load(f)
            
    # 2. Sinon chercher params Trading (Phase 1)
    trading_file = Path(f"optuna_results/{worker}_best_params.yaml")
    if trading_file.exists():
        logger.info(f"Loading Phase 1 params from {trading_file}")
        with open(trading_file, 'r') as f:
            data = yaml.safe_load(f)
            # Structurer comme Phase 2 pour compatibilité
            return {
                'trading_parameters': data['parameters'],
                'ppo_parameters': None
            }
            
    raise FileNotFoundError(f"No best params found for {worker}")

def create_validation_env(worker: str, params: Dict, start_chunk: int) -> MultiAssetChunkedEnv:
    """Crée l'environnement de validation"""
    # Charger config base
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    config = copy.deepcopy(base_config)
    
    # Configurer pour validation
    config['environment']['start_chunk_index'] = start_chunk
    config['dbe'] = {'enabled': False, 'override_risk_params': False}
    
    # Appliquer trading params
    trading_params = params['trading_parameters']
    
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
    
    env = MultiAssetChunkedEnv(config=config)
    
    # Injection directe
    if hasattr(env, 'portfolio'):
        env.portfolio.sl_pct = trading_params['stop_loss_pct']
        env.portfolio.tp_pct = trading_params['take_profit_pct']
        env.portfolio.pos_size_pct = trading_params['position_size_pct']
        env.portfolio.risk_per_trade = trading_params['risk_per_trade_pct']
        logger.info(f"Injected trading params: SL={env.portfolio.sl_pct:.2%}, TP={env.portfolio.tp_pct:.2%}")
        
    return env

def train_ppo_model(env: MultiAssetChunkedEnv, ppo_params: Dict, steps: int = 5000):
    """Entraîne un nouveau modèle PPO avec les hyperparamètres optimisés"""
    logger.info("Training new PPO model with optimized hyperparameters...")
    
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=ppo_params['learning_rate'],
        n_steps=ppo_params['n_steps'],
        batch_size=ppo_params['batch_size'],
        n_epochs=ppo_params['n_epochs'],
        gamma=ppo_params['gamma'],
        gae_lambda=ppo_params['gae_lambda'],
        clip_range=ppo_params['clip_range'],
        ent_coef=ppo_params['ent_coef'],
        vf_coef=ppo_params['vf_coef'],
        max_grad_norm=ppo_params['max_grad_norm'],
        verbose=1,
        seed=42,
    )
    
    model.learn(total_timesteps=steps)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=str, required=True)
    parser.add_argument('--chunks', type=int, default=2, help='Nombre de chunks de fin à utiliser')
    args = parser.parse_args()
    
    logger.info(f"=== VALIDATION OUT-OF-SAMPLE: {args.worker} ===")
    
    # 1. Charger params
    try:
        params = load_best_params(args.worker)
    except Exception as e:
        logger.error(f"Failed to load params: {e}")
        return

    # 2. Déterminer start chunk
    # On crée un env temporaire juste pour avoir total_chunks
    temp_env = create_validation_env(args.worker, params, 0)
    total_chunks = temp_env.total_chunks
    start_chunk = max(0, total_chunks - args.chunks)
    temp_env.close()
    
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Validation on chunks {start_chunk} to {total_chunks-1} (Last {args.chunks})")
    
    # 3. Créer env de validation
    env = create_validation_env(args.worker, params, start_chunk)
    
    # 4. Préparer modèle
    model = None
    if params.get('ppo_parameters'):
        # Si on a des params PPO, on entraîne un nouveau modèle RAPIDEMENT sur le début des données
        # ou on utilise le modèle pré-entraîné si disponible ?
        # Pour la validation pure, on devrait utiliser le modèle tel qu'il serait en prod.
        # Ici, on va charger le modèle pré-entraîné par défaut pour W1/W2/W4 si pas de params PPO,
        # MAIS si on a des params PPO, on devrait idéalement avoir un modèle entraîné avec.
        # Comme on vient juste d'optimiser les hyperparams mais pas sauvegardé le modèle entraîné,
        # on va devoir faire un mini-training ou utiliser le modèle par défaut.
        
        # DECISION: Pour cette validation, on utilise le modèle par défaut (w1_final.zip) 
        # car ré-entraîner prendrait trop de temps ici.
        # L'objectif est de valider les TRADING PARAMS sur des données non vues.
        # Les hyperparams PPO ne peuvent être validés que si on entraîne avec.
        
        logger.warning("PPO params found but skipping retraining for quick validation.")
        logger.warning("Using default pre-trained model.")
    
    # Charger modèle par défaut
    model_path = Path(f"models/rl_agents/final/{args.worker.lower()}_final.zip")
    if model_path.exists() and PPO_AVAILABLE:
        logger.info(f"Loading pre-trained model: {model_path}")
        model = PPO.load(str(model_path))
    else:
        logger.warning("No pre-trained model found, using random actions!")
        
    # 5. Exécuter validation
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
        
    portfolio_values = [env.portfolio.portfolio_value]
    wins = 0
    losses = 0
    trades_count = 0
    
    steps = 5000 # Suffisant pour 2 chunks
    
    logger.info("Starting validation run...")
    for i in range(steps):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.random.normal(0, 0.1, env.action_space.shape)
            
        result = env.step(action)
        
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False
            
        portfolio_values.append(env.portfolio.portfolio_value)
        
        if isinstance(info, dict):
            for pos in info.get('closed_positions', []):
                trades_count += 1
                pnl = pos.get('pnl', 0) if isinstance(pos, dict) else getattr(pos, 'realized_pnl', 0)
                if pnl > 0: wins += 1
                elif pnl < 0: losses += 1
        
        if done or truncated:
            break
            
    # 6. Métriques
    returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
    returns = returns[~np.isnan(returns)]
    
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.sqrt(252 * 24) * np.mean(returns) / np.std(returns)
    else:
        sharpe = 0.0
        
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - np.array(portfolio_values)) / peak
    max_dd = np.max(drawdown)
    
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    win_rate = wins / trades_count if trades_count > 0 else 0.0
    
    logger.info(f"")
    logger.info(f"=" * 50)
    logger.info(f"VALIDATION RESULTS - {args.worker}")
    logger.info(f"=" * 50)
    logger.info(f"Chunks: {start_chunk} -> {total_chunks-1}")
    logger.info(f"Steps: {i+1}")
    logger.info(f"Initial Portfolio: ${portfolio_values[0]:.2f}")
    logger.info(f"Final Portfolio:   ${portfolio_values[-1]:.2f}")
    logger.info(f"Total Return:      {total_return:.2%}")
    logger.info(f"Sharpe Ratio:      {sharpe:.4f}")
    logger.info(f"Max Drawdown:      {max_dd:.2%}")
    logger.info(f"Total Trades:      {trades_count}")
    logger.info(f"Win Rate:          {win_rate:.2%}")
    logger.info(f"=" * 50)

if __name__ == "__main__":
    main()
