#!/usr/bin/env python3
"""
Script optimisé pour agréger et classer les résultats des trials W2 & W3
Collecte, structure et restitue les meilleurs hyperparamètres
"""

import re
import json
from pathlib import Path
from collections import defaultdict

def parse_log_for_trials(log_file, worker):
    """Parse le log pour extraire les trials avec leurs métriques"""
    trials = []
    current_trial = None
    
    try:
        with open(log_file, 'r', errors='ignore') as f:
            for line in f:
                # Début d'un trial
                if f'Trial {worker}' in line or 'Trial [' in line:
                    if current_trial:
                        trials.append(current_trial)
                    
                    # Extraire le numéro du trial
                    match = re.search(r'Trial\s+(\d+)', line)
                    if match:
                        current_trial = {
                            'trial_num': int(match.group(1)),
                            'sharpe': None,
                            'pnl': None,
                            'capital': None,
                            'trades': 0,
                            'hyperparams': {}
                        }
                
                if not current_trial:
                    continue
                
                # Sharpe Ratio
                if 'Sharpe' in line and current_trial['sharpe'] is None:
                    match = re.search(r'Sharpe[:\s]+[-+]?\d+\.?\d*', line)
                    if match:
                        try:
                            current_trial['sharpe'] = float(re.findall(r'[-+]?\d+\.?\d*', match.group(0))[-1])
                        except:
                            pass
                
                # PnL
                if 'PnL' in line and current_trial['pnl'] is None:
                    match = re.search(r'\$[-+]?\d+\.?\d*', line)
                    if match:
                        try:
                            current_trial['pnl'] = float(match.group(0)[1:])
                        except:
                            pass
                
                # Capital/Portfolio
                if 'portfolio' in line.lower() or 'balance' in line.lower():
                    match = re.search(r':\s*([-+]?\d+\.?\d*)', line)
                    if match and current_trial['capital'] is None:
                        try:
                            val = float(match.group(1))
                            if val > 0:
                                current_trial['capital'] = val
                        except:
                            pass
                
                # Trades
                if 'trade' in line.lower():
                    current_trial['trades'] += 1
                
                # Hyperparamètres
                if 'learning_rate' in line:
                    match = re.search(r'learning_rate[:\s]+([0-9e\-\.]+)', line)
                    if match:
                        current_trial['hyperparams']['learning_rate'] = match.group(1)
                
                if 'batch_size' in line:
                    match = re.search(r'batch_size[:\s]+(\d+)', line)
                    if match:
                        current_trial['hyperparams']['batch_size'] = match.group(1)
                
                if 'n_steps' in line:
                    match = re.search(r'n_steps[:\s]+(\d+)', line)
                    if match:
                        current_trial['hyperparams']['n_steps'] = match.group(1)
                
                if 'position_size' in line:
                    match = re.search(r'position_size[:\s]+([0-9\.]+)', line)
                    if match:
                        current_trial['hyperparams']['position_size'] = match.group(1)
        
        if current_trial:
            trials.append(current_trial)
    
    except Exception as e:
        print(f"  ⚠️  Erreur parsing: {e}")
    
    return trials

def aggregate_results(trials, min_capital=28):
    """Agrège les résultats et classe par performance"""
    
    # Filtrer les trials avec capital >= min_capital
    good_trials = [t for t in trials if t['capital'] and t['capital'] >= min_capital]
    
    if not good_trials:
        return None
    
    # Classer par Sharpe Ratio
    good_trials.sort(key=lambda x: x['sharpe'] if x['sharpe'] else 0, reverse=True)
    
    return {
        'total_trials': len(trials),
        'good_trials': len(good_trials),
        'top_3': good_trials[:3],
        'all_good': good_trials
    }

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║      📊 AGRÉGATION RÉSULTATS W2 & W3 - TOP HYPERPARAMÈTRES   ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    results_by_worker = {}
    
    for worker in ['W2', 'W3']:
        log_pattern = f"logs/workers/optuna_{worker.lower()}_*.log"
        log_files = list(Path('.').glob(log_pattern))
        
        if not log_files:
            print(f"❌ {worker}: Aucun log trouvé")
            continue
        
        log_file = log_files[0]
        print(f"🔍 {worker}:")
        print(f"  📝 Log: {log_file.name}")
        
        # Parser les trials
        trials = parse_log_for_trials(str(log_file), worker)
        print(f"  📈 Trials parsés: {len(trials)}")
        
        # Agréger
        agg = aggregate_results(trials, min_capital=28)
        
        if not agg:
            print(f"  ⚠️  Aucun trial avec capital >= 28")
            print()
            continue
        
        results_by_worker[worker] = agg
        
        print(f"  ✅ Trials avec capital >= 28: {agg['good_trials']}")
        print()
        print(f"  🏆 TOP 3 HYPERPARAMÈTRES ({worker}):")
        print("  " + "━"*66)
        
        for i, trial in enumerate(agg['top_3'], 1):
            print(f"\n  {i}. Trial {trial['trial_num']}:")
            print(f"     Sharpe: {trial['sharpe']}")
            print(f"     PnL: ${trial['pnl']}")
            print(f"     Capital: ${trial['capital']}")
            print(f"     Trades: {trial['trades']}")
            print(f"     Hyperparamètres:")
            for key, val in trial['hyperparams'].items():
                print(f"       • {key}: {val}")
        
        print()
        print("  " + "━"*66)
        print()
    
    # Sauvegarder les résultats
    if results_by_worker:
        output_file = "results_w2_w3_aggregated.json"
        with open(output_file, 'w') as f:
            json.dump(results_by_worker, f, indent=2)
        print(f"\n✅ Résultats sauvegardés: {output_file}")

if __name__ == "__main__":
    main()
