#!/usr/bin/env python3
"""
Script optimisé pour extraire les top trials avec capital >= 28
Utilise grep et traitement ligne par ligne pour économiser la RAM
"""

import subprocess
import re
import json
from pathlib import Path
from collections import defaultdict

def extract_top_trials(log_file, min_capital=28):
    """Extrait les trials avec capital >= min_capital"""
    
    results = {
        'total_trials': 0,
        'good_trials': [],
        'top_3': []
    }
    
    try:
        # Compter les trials
        result = subprocess.run(
            f"grep -c 'Trial' {log_file}",
            shell=True, capture_output=True, text=True
        )
        results['total_trials'] = int(result.stdout.strip()) if result.stdout else 0
        
        # Extraire les lignes avec Portfolio et capital
        result = subprocess.run(
            f"grep -E 'Portfolio|portfolio' {log_file} | tail -100",
            shell=True, capture_output=True, text=True
        )
        
        portfolio_lines = result.stdout.strip().split('\n')
        
        for line in portfolio_lines:
            if not line.strip():
                continue
            
            # Chercher les valeurs numériques
            numbers = re.findall(r'[-+]?\d+\.?\d*', line)
            if numbers:
                try:
                    capital = float(numbers[-1])
                    if capital >= min_capital:
                        results['good_trials'].append({
                            'capital': capital,
                            'line': line[:100]
                        })
                except:
                    pass
        
        # Trier par capital décroissant
        results['good_trials'].sort(key=lambda x: x['capital'], reverse=True)
        results['top_3'] = results['good_trials'][:3]
        
    except Exception as e:
        print(f"  ⚠️  Erreur: {e}")
    
    return results

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         📊 EXTRACTION TOP TRIALS (CAPITAL >= 28)             ║")
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
        results = extract_top_trials(log_file, min_capital=28)
        all_results[worker] = results
        
        print(f"  📈 Total trials: {results['total_trials']}")
        print(f"  ✅ Trials avec capital >= 28: {len(results['good_trials'])}")
        
        if results['top_3']:
            print(f"  🏆 TOP 3:")
            for i, trial in enumerate(results['top_3'], 1):
                print(f"    {i}. Capital: ${trial['capital']:.2f}")
        else:
            print(f"  ⚠️  Aucun trial avec capital >= 28")
        
        print()
    
    # Sauvegarder
    if all_results:
        with open('top_trials_capital_28.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("✅ Résultats sauvegardés: top_trials_capital_28.json")

if __name__ == "__main__":
    main()
