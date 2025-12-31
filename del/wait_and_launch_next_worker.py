#!/usr/bin/env python3
"""
Attend la fin d'un worker et lance automatiquement le suivant
"""
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def wait_for_worker(worker: str, timeout_minutes: int = 180) -> bool:
    """Attend la fin d'un worker"""
    
    result_file = Path(f"optuna_results/{worker}_ppo_best_params.yaml")
    
    print(f"\n{'='*80}")
    print(f"⏳ ATTENTE - {worker}")
    print(f"{'='*80}")
    print(f"Fichier attendu : {result_file}")
    print(f"Timeout : {timeout_minutes} minutes")
    print(f"Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    while True:
        elapsed = (time.time() - start_time) / 60
        
        if elapsed > timeout_minutes:
            print(f"\n❌ TIMEOUT : {worker} n'a pas complété")
            return False
        
        if result_file.exists():
            print(f"\n✅ {worker} COMPLÉTÉ !")
            print(f"   Durée : {elapsed:.1f} minutes")
            return True
        
        progress = min(100, int((elapsed / timeout_minutes) * 100))
        bar = '█' * (progress // 5) + '░' * (20 - progress // 5)
        print(f"\r[{bar}] {progress}% ({elapsed:.1f}/{timeout_minutes} min)", end='', flush=True)
        
        time.sleep(10)

def launch_worker(worker: str, trials: int = 20, steps: int = 5000, eval_steps: int = 2000):
    """Lance l'optimisation pour un worker"""
    
    print(f"\n{'='*80}")
    print(f"🚀 LANCEMENT - {worker}")
    print(f"{'='*80}")
    print(f"Trials: {trials} | Steps: {steps} | Eval Steps: {eval_steps}")
    print(f"Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    cmd = [
        "python", "optuna_optimize_ppo.py",
        "--worker", worker,
        "--trials", str(trials),
        "--steps", str(steps),
        "--eval-steps", str(eval_steps),
        "--output-dir", "optuna_results"
    ]
    
    log_file = Path(f"optuna_results/{worker}_optimization.log")
    
    with open(log_file, 'w') as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    print(f"Processus lancé, log : {log_file}")

def main():
    workers = ["W1", "W2", "W3", "W4"]
    trials = 20
    steps = 5000
    eval_steps = 2000
    timeout = 180  # 3 heures par worker
    
    print(f"\n{'╔'+'═'*78+'╗'}")
    print(f"║ T8 : OPTIMISATION OPTUNA SÉQUENTIELLE - ORCHESTRATION AUTOMATIQUE {'':>10}║")
    print(f"{'╚'+'═'*78+'╝'}\n")
    
    for i, worker in enumerate(workers):
        print(f"\n{'─'*80}")
        print(f"ÉTAPE {i+1}/4 : {worker}")
        print(f"{'─'*80}")
        
        # Attendre le worker précédent (sauf pour W1)
        if i > 0:
            prev_worker = workers[i-1]
            if not wait_for_worker(prev_worker, timeout):
                print(f"❌ {prev_worker} n'a pas complété, abandon")
                return False
        
        # Lancer le worker actuel
        launch_worker(worker, trials, steps, eval_steps)
        
        # Attendre ce worker
        if not wait_for_worker(worker, timeout):
            print(f"❌ {worker} n'a pas complété, abandon")
            return False
    
    print(f"\n{'╔'+'═'*78+'╗'}")
    print(f"║ ✅ T8 COMPLÉTÉ - TOUS LES WORKERS OPTIMISÉS {'':>30}║")
    print(f"{'╚'+'═'*78+'╝'}\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
