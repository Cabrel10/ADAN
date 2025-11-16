#!/usr/bin/env python3
"""
Script optimisé pour extraire les hyperparamètres détaillés des top trials
Collecte tous les hyperparamètres pour W2 et W3
"""

import subprocess
import re
import json
from pathlib import Path

def extract_hyperparams_from_log(log_file, top_n=3):
    """Extrait les hyperparamètres des top trials"""
    
    results = {
        'total_trials': 0,
        'top_trials': []
    }
    
    try:
        # Extraire toutes les sections de trial
        result = subprocess.run(
            f"grep -n 'Trial\\|learning_rate\\|batch_size\\|n_steps\\|position_size\\|stop_loss\\|take_profit\\|pnl_weight\\|win_rate\\|Sharpe\\|PnL\\|portfolio\\|Portfolio' {log_file} | tail -500",
            shell=True, capture_output=True, text=True, timeout=30
        )
        
        lines = result.stdout.strip().split('\n')
        
        current_trial = None
        trial_count = 0
        
        for line in lines:
            if not line.strip():
                continue
            
            # Nouveau trial
            if 'Trial' in line and ('[' in line or 'finished' in line):
                if current_trial:
                    results['top_trials'].append(current_trial)
                    trial_count += 1
                
                current_trial = {
                    'trial_num': trial_count,
                    'sharpe': None,
                    'pnl': None,
                    'capital': None,
                    'hyperparams': {}
                }
            
            if not current_trial:
                continue
            
            # Extraire les hyperparamètres
            if 'learning_rate' in line.lower():
                match = re.search(r'learning_rate[:\s=]+([0-9e\-\.]+)', line, re.IGNORECASE)
                if match:
                    current_trial['hyperparams']['learning_rate'] = match.group(1)
            
            if 'batch_size' in line.lower():
                match = re.search(r'batch_size[:\s=]+(\d+)', line, re.IGNORECASE)
                if match:
                    current_trial['hyperparams']['batch_size'] = match.group(1)
            
            if 'n_steps' in line.lower():
                match = re.search(r'n_steps[:\s=]+(\d+)', line, re.IGNORECASE)
                if match:
                    current_trial['hyperparams']['n_steps'] = match.group(1)
            
            if 'position_size' in line.lower():
                match = re.search(r'position_size[:\s=]+([0-9\.]+)', line, re.IGNORECASE)
                if match:
                    current_trial['hyperparams']['position_size'] = match.group(1)
            
            if 'stop_loss' in line.lower():
                match = re.search(r'stop_loss[:\s=]+([0-9\.]+)', line, re.IGNORECASE)
                if match:
                    current_trial['hyperparams']['stop_loss_pct'] = match.group(1)
            
            if 'take_profit' in line.lower():
                match = re.search(r'take_profit[:\s=]+([0-9\.]+)', line, re.IGNORECASE)
                if match:
                    current_trial['hyperparams']['take_profit_pct'] = match.group(1)
            
            if 'pnl_weight' in line.lower():
                match = re.search(r'pnl_weight[:\s=]+([0-9\.]+)', line, re.IGNORECASE)
                if match:
                    current_trial['hyperparams']['pnl_weight'] = match.group(1)
            
            if 'win_rate' in line.lower():
                match = re.search(r'win_rate[:\s=]+([0-9\.]+)', line, re.IGNORECASE)
                if match:
                    current_trial['hyperparams']['win_rate_bonus'] = match.group(1)
            
            # Métriques
            if 'Sharpe' in line and current_trial['sharpe'] is None:
                numbers = re.findall(r'[-+]?\d+\.?\d*', line)
                if numbers:
                    try:
                        current_trial['sharpe'] = float(numbers[-1])
                    except:
                        pass
            
            if 'PnL' in line and current_trial['pnl'] is None:
                match = re.search(r'\$[-+]?\d+\.?\d*', line)
                if match:
                    try:
                        current_trial['pnl'] = float(match.group(0)[1:])
                    except:
                        pass
            
            if ('portfolio' in line.lower() or 'balance' in line.lower()) and current_trial['capital'] is None:
                numbers = re.findall(r'[-+]?\d+\.?\d*', line)
                if numbers:
                    try:
                        val = float(numbers[-1])
                        if val > 0:
                            current_trial['capital'] = val
                    except:
                        pass
        
        if current_trial:
            results['top_trials'].append(current_trial)
        
        results['total_trials'] = trial_count
        
        # Trier par capital
        results['top_trials'].sort(key=lambda x: x['capital'] if x['capital'] else 0, reverse=True)
        results['top_trials'] = results['top_trials'][:top_n]
    
    except Exception as e:
        print(f"  ⚠️  Erreur: {e}")
    
    return results

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║      📊 EXTRACTION HYPERPARAMÈTRES DÉTAILLÉS W2 & W3          ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    all_results = {}
    
    for worker in ['W2', 'W3']:
        log_pattern = f"logs/workers/optuna_{worker.lower()}_*.log"
        log_files = list(Path('.').glob(log_pattern))
        
        if not log_files:
            print(f"❌ {worker}: Aucun log trouvé")
            continue
        
        log_file = str(log_files[0])
        print(f"🔍 {worker}:")
        print(f"  📝 Log: {Path(log_file).name}")
        
        # Extraire
        results = extract_hyperparams_from_log(log_file, top_n=3)
        all_results[worker] = results
        
        print(f"  📈 Total trials: {results['total_trials']}")
        print(f"  🏆 TOP 3 HYPERPARAMÈTRES:")
        print()
        
        for i, trial in enumerate(results['top_trials'], 1):
            print(f"  {i}. Trial {trial['trial_num']}:")
            print(f"     Sharpe: {trial['sharpe']}")
            print(f"     PnL: ${trial['pnl']}")
            print(f"     Capital: ${trial['capital']:.2f}")
            print(f"     Hyperparamètres:")
            for key, val in trial['hyperparams'].items():
                print(f"       • {key}: {val}")
            print()
    
    # Sauvegarder
    if all_results:
        with open('hyperparams_top_trials_w2_w3.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("✅ Résultats sauvegardés: hyperparams_top_trials_w2_w3.json")

if __name__ == "__main__":
    main()
