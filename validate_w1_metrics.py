#!/usr/bin/env python3
"""
Validation des métriques W1 - Vérifier si le return de 944% est réel
"""
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

try:
    from stable_baselines3 import PPO
    import torch
except ImportError:
    print("stable_baselines3 not available")
    sys.exit(1)

# Charger la config
config_path = Path(__file__).parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Paramètres W1 (best Phase 1)
W1_TRADING_PARAMS = {
    'stop_loss_pct': 0.0253,
    'take_profit_pct': 0.0321,
    'position_size_pct': 0.1121,
}

# Best PPO params W1
W1_PPO_PARAMS = {
    'learning_rate': 2.8348596045804754e-05,
    'n_steps': 2048,
    'batch_size': 128,
    'n_epochs': 15,
    'gamma': 0.9703841934332026,
    'gae_lambda': 0.9227107639972829,
    'clip_range': 0.34195414918908396,
    'ent_coef': 0.01954665809730191,
    'vf_coef': 0.7734809381950747,
    'max_grad_norm': 0.7621906957987917,
}

def validate_w1():
    """Validation indépendante des métriques W1"""
    
    print("="*80)
    print("VALIDATION INDÉPENDANTE W1")
    print("="*80)
    
    # Configurer l'environnement
    import copy
    test_config = copy.deepcopy(config)
    test_config['dbe'] = {'enabled': False}
    
    if 'trading_rules' not in test_config:
        test_config['trading_rules'] = {}
    test_config['trading_rules']['risk_management'] = {
        'stop_loss_pct': W1_TRADING_PARAMS['stop_loss_pct'],
        'take_profit_pct': W1_TRADING_PARAMS['take_profit_pct'],
    }
    
    env = MultiAssetChunkedEnv(config=test_config)
    
    # Créer et entraîner le modèle
    print("\n[1] Entraînement du modèle PPO...")
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        **W1_PPO_PARAMS,
        verbose=0,
        seed=42,
    )
    model.learn(total_timesteps=4000, progress_bar=True)
    
    # Évaluation
    print("\n[2] Évaluation sur 3000 steps...")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    portfolio_values = []
    trades = []
    initial_pv = env.portfolio.portfolio_value
    portfolio_values.append(initial_pv)
    
    for step in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        result = env.step(action)
        
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False
        
        current_pv = env.portfolio.portfolio_value
        portfolio_values.append(current_pv)
        
        # Collecter trades
        if isinstance(info, dict):
            closed = info.get('closed_positions', [])
            if closed:
                for pos in closed:
                    if isinstance(pos, dict):
                        trades.append(pos)
        
        if done or truncated:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
    
    # Calculer métriques
    pv = np.array(portfolio_values)
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-8)
    returns = returns[np.isfinite(returns)]
    
    total_return = (pv[-1] - pv[0]) / max(pv[0], 1e-8)
    
    # Sharpe
    if len(returns) > 1 and np.std(returns) > 1e-10:
        sharpe = np.sqrt(252 * 24 * 12) * np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
    
    # Drawdown
    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / np.maximum(peak, 1e-8)
    max_dd = np.max(drawdown)
    
    # Trades stats
    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
    losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
    total_trades = wins + losses
    win_rate = wins / max(total_trades, 1)
    
    print("\n" + "="*80)
    print("RÉSULTATS VALIDATION W1")
    print("="*80)
    print(f"Portfolio Initial: ${pv[0]:.2f}")
    print(f"Portfolio Final:   ${pv[-1]:.2f}")
    print(f"Total Return:      {total_return*100:.1f}%")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Max Drawdown:      {max_dd*100:.1f}%")
    print(f"Total Trades:      {total_trades}")
    print(f"Wins/Losses:       {wins}/{losses}")
    print(f"Win Rate:          {win_rate*100:.1f}%")
    print("="*80)
    
    # Analyse de plausibilité
    print("\n[ANALYSE DE PLAUSIBILITÉ]")
    if total_return > 5:  # >500%
        print("⚠️ WARNING: Return >500% - Potentiellement surévalué")
        print("   Causes possibles:")
        print("   - Overfitting sur les données d'entraînement")
        print("   - Période d'évaluation trop courte")
        print("   - Données de marché favorables")
    elif total_return > 1:  # >100%
        print("⚡ ATTENTION: Return >100% - À valider avec données OOS")
    else:
        print("✅ Return dans une plage réaliste")
    
    if max_dd < 0.05:
        print("⚠️ WARNING: Drawdown très faible (<5%) - Suspect")
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': total_trades,
        'win_rate': win_rate,
    }

if __name__ == "__main__":
    validate_w1()
