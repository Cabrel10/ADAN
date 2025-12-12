#!/usr/bin/env python3
"""
Vérifie la complétion de l'optimisation Optuna et affiche les résultats
"""
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime

def check_worker_completion(worker: str, timeout_minutes: int = 180) -> bool:
    """Vérifie si un worker est complété"""
    
    result_file = Path(f"optuna_results/{worker}_ppo_best_params.yaml")
    log_file = Path(f"optuna_results/{worker}_optimization.log")
    
    print(f"\n{'='*80}")
    print(f"🔍 VÉRIFICATION - {worker}")
    print(f"{'='*80}")
    print(f"Fichier résultat attendu : {result_file}")
    print(f"Fichier log : {log_file}")
    print(f"Timeout : {timeout_minutes} minutes")
    print(f"Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    while True:
        elapsed = (time.time() - start_time) / 60
        
        # Vérifier le timeout
        if elapsed > timeout_minutes:
            print(f"❌ TIMEOUT : {worker} n'a pas complété en {timeout_minutes} minutes")
            return False
        
        # Vérifier si le fichier résultat existe
        if result_file.exists():
            print(f"✅ {worker} COMPLÉTÉ !")
            print(f"   Durée : {elapsed:.1f} minutes")
            
            # Charger et afficher les résultats
            try:
                with open(result_file, 'r') as f:
                    results = yaml.safe_load(f)
                
                print(f"\n📊 RÉSULTATS {worker}")
                print(f"{'─'*80}")
                print(f"Score : {results['score']:.2f}")
                print(f"Sharpe Ratio : {results['metrics']['sharpe']:.2f}")
                print(f"Max Drawdown : {results['metrics']['drawdown']:.1%}")
                print(f"Win Rate : {results['metrics']['win_rate']:.1%}")
                print(f"Total Return : {results['metrics']['total_return']:.1%}")
                print(f"Profit Factor : {results['metrics']['profit_factor']:.2f}")
                print(f"Total Trades : {results['metrics']['trades']}")
                
                print(f"\n🎯 HYPERPARAMÈTRES PPO")
                print(f"{'─'*80}")
                for key, value in results['ppo_parameters'].items():
                    print(f"{key:20s} : {value}")
                
                print(f"\n{'─'*80}")
                return True
            
            except Exception as e:
                print(f"⚠️  Erreur lors de la lecture des résultats : {e}")
                return False
        
        # Afficher la progression
        progress = min(100, int((elapsed / timeout_minutes) * 100))
        bar = '█' * (progress // 5) + '░' * (20 - progress // 5)
        print(f"\r[{bar}] {progress}% ({elapsed:.1f}/{timeout_minutes} min)", end='', flush=True)
        
        time.sleep(10)  # Vérifier toutes les 10 secondes

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_optuna_completion.py <WORKER> [TIMEOUT_MINUTES]")
        print("Example: python check_optuna_completion.py W1 120")
        sys.exit(1)
    
    worker = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    
    success = check_worker_completion(worker, timeout)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
