#!/usr/bin/env python3
"""
Extrait les hyperparamètres CORRECTS des top trials
Utilise grep pour éviter les erreurs de parsing
"""

import subprocess
import json
import re
from pathlib import Path

def extract_trial_hyperparams(log_file, trial_num):
    """Extrait les hyperparamètres d'un trial spécifique"""
    
    hyperparams = {}
    
    try:
        # Chercher la section du trial
        result = subprocess.run(
            f"grep -A 500 'Trial {trial_num}' {log_file} | grep -B 500 'Trial {trial_num + 1}' | head -500",
            shell=True, capture_output=True, text=True, timeout=10
        )
        
        lines = result.stdout
        
        # Extraire les hyperparamètres
        patterns = {
            'learning_rate': r'learning_rate[:\s=]+([0-9e\-\.]+)',
            'batch_size': r'batch_size[:\s=]+(\d+)',
            'n_steps': r'n_steps[:\s=]+(\d+)',
            'gamma': r'gamma[:\s=]+([0-9\.]+)',
            'ent_coef': r'ent_coef[:\s=]+([0-9e\-\.]+)',
            'clip_range': r'clip_range[:\s=]+([0-9\.]+)',
            'n_epochs': r'n_epochs[:\s=]+(\d+)',
            'vf_coef': r'vf_coef[:\s=]+([0-9\.]+)',
            'max_grad_norm': r'max_grad_norm[:\s=]+([0-9\.]+)',
            'gae_lambda': r'gae_lambda[:\s=]+([0-9\.]+)',
            'position_size': r'position_size[:\s=]+([0-9\.]+)',
            'stop_loss_pct': r'stop_loss_pct[:\s=]+([0-9\.]+)',
            'take_profit_pct': r'take_profit_pct[:\s=]+([0-9\.]+)',
            'pnl_weight': r'pnl_weight[:\s=]+([0-9\.]+)',
            'win_rate_bonus': r'win_rate_bonus[:\s=]+([0-9\.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, lines, re.IGNORECASE)
            if match:
                hyperparams[key] = match.group(1)
        
    except Exception as e:
        print(f"  ⚠️  Erreur extraction: {e}")
    
    return hyperparams

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║      🔧 EXTRACTION HYPERPARAMÈTRES - TOP TRIALS               ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Charger les résultats corrects
    with open('true_portfolios_w2_w3.json', 'r') as f:
        results = json.load(f)
    
    all_hyperparams = {}
    
    for worker in ['W2', 'W3']:
        if worker not in results or not results[worker]['top_3']:
            print(f"❌ {worker}: Aucun top trial")
            continue
        
        log_pattern = f"logs/workers/optuna_{worker.lower()}_*.log"
        log_files = list(Path('.').glob(log_pattern))
        
        if not log_files:
            print(f"❌ {worker}: Aucun log trouvé")
            continue
        
        log_file = str(log_files[0])
        
        print(f"🔍 {worker}:")
        print(f"  📝 Log: {Path(log_file).name}")
        print()
        
        worker_hyperparams = {}
        
        for i, (trial_num, portfolio) in enumerate(results[worker]['top_3'], 1):
            print(f"  {i}. Trial {trial_num} (Portfolio: ${portfolio:.2f}):")
            
            hyperparams = extract_trial_hyperparams(log_file, trial_num)
            worker_hyperparams[trial_num] = hyperparams
            
            if hyperparams:
                print(f"     ✅ Hyperparamètres trouvés: {len(hyperparams)}")
                for key, val in hyperparams.items():
                    print(f"       • {key}: {val}")
            else:
                print(f"     ⚠️  Aucun hyperparamètre trouvé")
            
            print()
        
        all_hyperparams[worker] = worker_hyperparams
    
    # Sauvegarder
    with open('hyperparams_top_trials_correct.json', 'w') as f:
        json.dump(all_hyperparams, f, indent=2)
    
    print("✅ Résultats sauvegardés: hyperparams_top_trials_correct.json")

if __name__ == "__main__":
    main()
