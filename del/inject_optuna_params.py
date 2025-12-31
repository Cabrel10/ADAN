#!/usr/bin/env python3
"""
T9 : Injecter les hyperparamètres Optuna dans config.yaml

Script pour lire les résultats Optuna et mettre à jour config.yaml
avec les paramètres de trading et les hyperparamètres d'agent optimisés.
"""
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

def load_optuna_results(worker: str) -> Dict[str, Any]:
    """Charge les résultats Optuna pour un worker"""
    result_file = Path(f"optuna_results/{worker}_ppo_best_params.yaml")
    
    if not result_file.exists():
        print(f"❌ Fichier non trouvé : {result_file}")
        return None
    
    with open(result_file, 'r') as f:
        return yaml.safe_load(f)

def load_config() -> Dict[str, Any]:
    """Charge la configuration actuelle"""
    config_path = Path("config/config.yaml")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any]) -> None:
    """Sauvegarde la configuration mise à jour"""
    config_path = Path("config/config.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def inject_worker_params(config: Dict[str, Any], worker: str, optuna_result: Dict[str, Any]) -> None:
    """Injecte les paramètres Optuna pour un worker"""
    
    worker_key = worker.lower()  # w1, w2, w3, w4
    
    if worker_key not in config.get('workers', {}):
        print(f"⚠️  Worker {worker_key} non trouvé dans config.yaml")
        return
    
    worker_config = config['workers'][worker_key]
    
    # 1. Mettre à jour les paramètres de trading
    trading_params = optuna_result.get('trading_parameters', {})
    
    if 'trading_parameters' not in worker_config:
        worker_config['trading_parameters'] = {}
    
    for key, value in trading_params.items():
        worker_config['trading_parameters'][key] = value
        print(f"   ✅ {worker_key}.trading_parameters.{key} = {value}")
    
    # 2. Mettre à jour les hyperparamètres d'agent (PPO)
    ppo_params = optuna_result.get('ppo_parameters', {})
    
    if 'agent_config' not in worker_config:
        worker_config['agent_config'] = {}
    
    if 'ppo_hyperparams' not in worker_config['agent_config']:
        worker_config['agent_config']['ppo_hyperparams'] = {}
    
    for key, value in ppo_params.items():
        worker_config['agent_config']['ppo_hyperparams'][key] = value
        print(f"   ✅ {worker_key}.agent_config.ppo_hyperparams.{key} = {value}")
    
    # 3. Stocker les métriques Optuna pour référence
    metrics = optuna_result.get('metrics', {})
    
    if 'optuna_metrics' not in worker_config:
        worker_config['optuna_metrics'] = {}
    
    for key, value in metrics.items():
        worker_config['optuna_metrics'][key] = value
        print(f"   ℹ️  {worker_key}.optuna_metrics.{key} = {value}")

def main():
    print(f"\n{'='*80}")
    print(f"T9 : INJECTION DES HYPERPARAMÈTRES OPTUNA DANS CONFIG.YAML")
    print(f"{'='*80}\n")
    
    # Charger la configuration actuelle
    print("📖 Chargement de config.yaml...")
    config = load_config()
    print("✅ Config chargée\n")
    
    # Injecter les paramètres pour chaque worker
    workers = ['W1', 'W2', 'W3', 'W4']
    
    for worker in workers:
        print(f"🔄 Injection des paramètres pour {worker}...")
        
        # Charger les résultats Optuna
        optuna_result = load_optuna_results(worker)
        
        if optuna_result is None:
            print(f"❌ Impossible de charger les résultats pour {worker}\n")
            continue
        
        # Injecter les paramètres
        inject_worker_params(config, worker, optuna_result)
        print()
    
    # Sauvegarder la configuration mise à jour
    print("💾 Sauvegarde de config.yaml...")
    save_config(config)
    print("✅ Config sauvegardée\n")
    
    print(f"{'='*80}")
    print(f"✅ T9 COMPLÉTÉ - HYPERPARAMÈTRES INJECTÉS AVEC SUCCÈS")
    print(f"{'='*80}\n")
    
    # Afficher un résumé
    print("📊 RÉSUMÉ DES INJECTIONS")
    print(f"{'─'*80}")
    
    for worker in workers:
        worker_key = worker.lower()
        if worker_key in config.get('workers', {}):
            worker_config = config['workers'][worker_key]
            
            trading_params = worker_config.get('trading_parameters', {})
            ppo_params = worker_config.get('agent_config', {}).get('ppo_hyperparams', {})
            metrics = worker_config.get('optuna_metrics', {})
            
            print(f"\n{worker} ({worker_key}):")
            print(f"  Trading Parameters: {len(trading_params)} paramètres")
            print(f"  PPO Hyperparams: {len(ppo_params)} paramètres")
            print(f"  Optuna Metrics: {len(metrics)} métriques")
            
            if metrics:
                print(f"    - Sharpe: {metrics.get('sharpe', 'N/A'):.2f}")
                print(f"    - Drawdown: {metrics.get('drawdown', 'N/A'):.1%}")
                print(f"    - Win Rate: {metrics.get('win_rate', 'N/A'):.1%}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
