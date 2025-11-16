#!/usr/bin/env python3
"""
Script CORRECT pour extraire les vrais portfolio values
Ignore les prix d'actifs, prend seulement les Portfolio value réels
"""

import re
import json
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path("logs/workers")
PAT_TRIAL_START = re.compile(r"\bTrial\b\s*[:#]?\s*(\d+)", re.IGNORECASE)
PAT_PORTF = re.compile(r"Portfolio value[:=]?\s*\$?\s*([0-9]+\.[0-9]+)", re.IGNORECASE)

def process_log(log_path):
    """Extrait les portfolio values réels par trial"""
    trials = defaultdict(lambda: {"last_portfolio": None})
    current_trial = None
    
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            # Détecter début d'un trial
            m = PAT_TRIAL_START.search(line)
            if m:
                current_trial = int(m.group(1))
            
            if current_trial is None:
                continue
            
            # Extraire Portfolio value
            mp = PAT_PORTF.search(line)
            if mp:
                try:
                    val = float(mp.group(1))
                    # Filtre: portfolio value doit être entre 10 et 100 (réaliste)
                    # Ignore les prix BTC (60000+)
                    if 10 <= val <= 100:
                        trials[current_trial]["last_portfolio"] = val
                except:
                    pass
    
    return trials

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║      🔍 EXTRACTION CORRECTE DES PORTFOLIO VALUES              ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    all_results = {}
    
    for worker in ['W2', 'W3']:
        log_pattern = f"logs/workers/optuna_{worker.lower()}_*.log"
        log_files = list(Path('.').glob(log_pattern))
        
        if not log_files:
            print(f"❌ {worker}: Aucun log trouvé")
            continue
        
        log_file = log_files[0]
        print(f"🔍 {worker}:")
        print(f"  📝 Log: {log_file.name}")
        
        # Extraire
        trials = process_log(str(log_file))
        
        # Filtrer les trials avec portfolio >= 28
        good_trials = {t: info["last_portfolio"] for t, info in trials.items() 
                      if info["last_portfolio"] and info["last_portfolio"] >= 28}
        
        print(f"  📈 Total trials: {len(trials)}")
        print(f"  ✅ Trials avec portfolio >= 28: {len(good_trials)}")
        
        if good_trials:
            sorted_trials = sorted(good_trials.items(), key=lambda x: x[1], reverse=True)
            print(f"  🏆 TOP 3:")
            for i, (trial_num, portfolio) in enumerate(sorted_trials[:3], 1):
                gain = portfolio - 20.5
                roi = (gain / 20.5) * 100
                print(f"    {i}. Trial {trial_num}: ${portfolio:.2f} (Gain: +${gain:.2f}, ROI: {roi:.1f}%)")
        else:
            print(f"  ⚠️  Aucun trial avec portfolio >= 28")
            print(f"  📊 Portfolio max trouvé: ${max([v for v in trials.values() if v['last_portfolio']], default={'last_portfolio': 0})['last_portfolio']:.2f}")
        
        all_results[worker] = {
            'total_trials': len(trials),
            'good_trials': len(good_trials),
            'top_3': sorted_trials[:3] if good_trials else []
        }
        
        print()
    
    # Sauvegarder
    with open('true_portfolios_w2_w3.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("✅ Résultats sauvegardés: true_portfolios_w2_w3.json")

if __name__ == "__main__":
    main()
