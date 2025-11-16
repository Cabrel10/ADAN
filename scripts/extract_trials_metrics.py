#!/usr/bin/env python3
"""
Script optimisé pour extraire les métriques des trials W2 & W3
Utilise des algorithmes légers pour ne pas surcharger la RAM
"""

import sqlite3
import json
import re
from pathlib import Path
from collections import defaultdict

def extract_metrics_from_logs(log_file):
    """Extrait les métriques des logs sans charger tout en mémoire"""
    metrics = {
        'trials': 0,
        'sharpe_values': [],
        'pnl_values': [],
        'capital_values': [],
        'trades': 0
    }
    
    try:
        with open(log_file, 'r', errors='ignore') as f:
            for line in f:
                # Compter les trials
                if 'Trial' in line and 'finished' in line:
                    metrics['trials'] += 1
                
                # Sharpe Ratio
                if 'Sharpe' in line or 'sharpe' in line:
                    numbers = re.findall(r'[-+]?\d+\.\d+', line)
                    if numbers:
                        metrics['sharpe_values'].append(float(numbers[0]))
                
                # PnL
                if 'PnL' in line or 'pnl' in line or 'profit' in line.lower():
                    if '$' in line:
                        pnl_match = re.search(r'\$[-+]?\d+\.\d+', line)
                        if pnl_match:
                            metrics['pnl_values'].append(float(pnl_match.group(0)[1:]))
                
                # Capital/Portfolio
                if 'portfolio' in line.lower() or 'capital' in line.lower() or 'balance' in line.lower():
                    numbers = re.findall(r'[-+]?\d+\.\d+', line)
                    if numbers:
                        metrics['capital_values'].append(float(numbers[0]))
                
                # Trades
                if 'trade' in line.lower():
                    metrics['trades'] += 1
    
    except Exception as e:
        print(f"  ⚠️  Erreur lecture log: {e}")
    
    return metrics

def extract_from_optuna_db():
    """Extrait les données de la base Optuna"""
    db_path = Path("optuna.db")
    if not db_path.exists():
        return {}
    
    results = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Récupérer les études W2 et W3
        cursor.execute("SELECT study_name FROM studies WHERE study_name LIKE '%w2%' OR study_name LIKE '%w3%'")
        studies = cursor.fetchall()
        
        for (study_name,) in studies:
            worker = 'W2' if 'w2' in study_name.lower() else 'W3'
            
            # Récupérer l'ID de l'étude
            cursor.execute("SELECT id FROM studies WHERE study_name = ?", (study_name,))
            study_id_row = cursor.fetchone()
            if not study_id_row:
                continue
            
            study_id = study_id_row[0]
            
            # Récupérer les trials
            cursor.execute("""
                SELECT number, value FROM trials 
                WHERE study_id = ? 
                ORDER BY number DESC
                LIMIT 100
            """, (study_id,))
            
            trials = cursor.fetchall()
            results[worker] = {
                'study': study_name,
                'trials': len(trials),
                'top_trials': trials[:10]
            }
        
        conn.close()
    
    except Exception as e:
        print(f"  ⚠️  Erreur DB: {e}")
    
    return results

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         📊 EXTRACTION MÉTRIQUES W2 & W3 (OPTIMISÉ)           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Extraire de la base Optuna
    print("🔍 Extraction de la base Optuna...")
    db_results = extract_from_optuna_db()
    
    for worker in ['W2', 'W3']:
        if worker not in db_results:
            print(f"\n❌ {worker}: Aucune étude trouvée")
            continue
        
        data = db_results[worker]
        print(f"\n✅ {worker}:")
        print(f"  📚 Étude: {data['study']}")
        print(f"  📈 Total trials: {data['trials']}")
        print(f"  🏆 Top 5 trials:")
        
        for i, (trial_num, value) in enumerate(data['top_trials'][:5], 1):
            print(f"    {i}. Trial {trial_num}: Sharpe={value}")
    
    # Extraire des logs
    print("\n\n🔍 Extraction des logs...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    for worker in ['W2', 'W3']:
        log_pattern = f"logs/workers/optuna_{worker.lower()}_*.log"
        log_files = list(Path('.').glob(log_pattern))
        
        if not log_files:
            print(f"\n❌ {worker}: Aucun log trouvé")
            continue
        
        log_file = log_files[0]
        print(f"\n✅ {worker}:")
        print(f"  📝 Log: {log_file.name}")
        
        metrics = extract_metrics_from_logs(str(log_file))
        
        print(f"  📈 Trials: {metrics['trials']}")
        if metrics['sharpe_values']:
            print(f"  🎯 Sharpe (derniers): {metrics['sharpe_values'][-5:]}")
        if metrics['pnl_values']:
            print(f"  💰 PnL (derniers): {metrics['pnl_values'][-5:]}")
        if metrics['capital_values']:
            print(f"  💵 Capital (derniers): {metrics['capital_values'][-5:]}")
        print(f"  📊 Trades: {metrics['trades']}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
