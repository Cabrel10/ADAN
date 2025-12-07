#!/usr/bin/env python3
"""Charge les meilleurs paramètres Optuna dans config.yaml et lance l'entraînement"""

import yaml
import sys
from pathlib import Path

def load_optuna_params():
    """Charge les meilleurs paramètres de chaque worker"""
    params = {}
    for w in ['W1', 'W2', 'W3', 'W4']:
        yaml_file = f"optuna_results/{w}_best_params.yaml"
        if Path(yaml_file).exists():
            with open(yaml_file, 'r') as f:
                params[w] = yaml.safe_load(f)
                print(f"✅ {w} chargé: Score={params[w]['score']:.4f}, Sharpe={params[w]['metrics']['sharpe']:.4f}")
        else:
            print(f"❌ {w} non trouvé: {yaml_file}")
            return None
    return params

def update_config(params):
    """Met à jour config.yaml avec les paramètres Optuna"""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Mapping W1-W4 à w1-w4
    worker_map = {'W1': 'w1', 'W2': 'w2', 'W3': 'w3', 'W4': 'w4'}
    
    print("\n📝 Mise à jour de config.yaml...")
    for w_upper, w_lower in worker_map.items():
        if w_upper in params:
            p = params[w_upper]['parameters']
            
            # Liste des paramètres à mettre à jour (agent_config et risk_management)
            param_mapping = {
                # Agent Config
                'learning_rate': ['agent_config', 'learning_rate'],
                'gamma': ['agent_config', 'gamma'],
                'ent_coef': ['agent_config', 'ent_coef'],
                'clip_range': ['agent_config', 'clip_range'],
                'n_epochs': ['agent_config', 'n_epochs'],
                'n_steps': ['agent_config', 'n_steps'],
                'batch_size': ['agent_config', 'batch_size'],
                
                # Risk Management
                'stop_loss_pct': ['risk_management', 'stop_loss_pct'],
                'take_profit_pct': ['risk_management', 'take_profit_pct'],
                'risk_per_trade_pct': ['risk_management', 'risk_per_trade_pct'],
                'position_size_pct': ['risk_management', 'position_size_pct'],
                
                # Trading Rules
                'max_concurrent_positions': ['trading_rules', 'max_positions', 'total'],
                'min_holding_period_steps': ['trading_rules', 'position_hold_min']
            }

            print(f"  -> Worker {w_lower}:")
            for param_name, path in param_mapping.items():
                if param_name in p:
                    value = p[param_name]
                    
                    # Naviguer et mettre à jour la configuration
                    sub_config = config['workers'][w_lower]
                    for key in path[:-1]:
                        sub_config = sub_config.setdefault(key, {})
                    sub_config[path[-1]] = value
                    print(f"    - {'.'.join(path)}: {value:.6f}")

    # Sauvegarder la config mise à jour
    with open('config/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("\n✅ config.yaml mis à jour avec TOUS les paramètres Optuna.")
    return True

if __name__ == '__main__':
    print("🔍 Chargement des meilleurs paramètres Optuna...")
    params = load_optuna_params()
    
    if params:
        print("\n📝 Mise à jour de config.yaml...")
        if update_config(params):
            print("\n✅ Prêt pour l'entraînement!")
            print("\nLance: python scripts/train_parallel_agents.py --config config/config.yaml --log-level INFO --steps 1000000")
    else:
        print("\n❌ Erreur: Impossible de charger les paramètres Optuna")
        sys.exit(1)
