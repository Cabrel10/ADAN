#!/usr/bin/env python3
"""
Extrait les hyperparamètres directement de la base Optuna
"""

import sqlite3
import json
from pathlib import Path

def extract_optuna_hyperparams():
    """Extrait les hyperparamètres depuis optuna.db"""
    
    db_path = Path("optuna.db")
    if not db_path.exists():
        print("❌ optuna.db non trouvé")
        return {}
    
    results = {}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Récupérer les études
        cursor.execute("SELECT study_id, study_name FROM studies")
        studies = cursor.fetchall()
        
        print(f"📚 Études trouvées: {len(studies)}")
        
        for study_id, study_name in studies:
            if 'w2' not in study_name.lower() and 'w3' not in study_name.lower():
                continue
            
            worker = 'W2' if 'w2' in study_name.lower() else 'W3'
            
            print(f"\n🔍 {worker} - Étude: {study_name}")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # Récupérer les trials
            cursor.execute("""
                SELECT trial_id, number 
                FROM trials 
                WHERE study_id = ? 
                ORDER BY number DESC 
                LIMIT 50
            """, (study_id,))
            
            trials = cursor.fetchall()
            
            worker_data = {}
            
            for trial_id, trial_num in trials:
                # Récupérer les paramètres du trial
                cursor.execute("""
                    SELECT param_name, param_value 
                    FROM trial_params 
                    WHERE trial_id = ?
                """, (trial_id,))
                
                params = cursor.fetchall()
                
                if params:
                    param_dict = {name: val for name, val in params}
                    worker_data[trial_num] = {
                        'hyperparams': param_dict
                    }
            
            results[worker] = worker_data
            
            # Afficher top 3
            top_3 = sorted(worker_data.items(), key=lambda x: len(x[1]['hyperparams']), reverse=True)[:3]
            
            print(f"  ✅ Top 3 trials:")
            for trial_num, data in top_3:
                print(f"    Trial {trial_num}: params={len(data['hyperparams'])}")
        
        conn.close()
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║      📊 EXTRACTION HYPERPARAMÈTRES DEPUIS OPTUNA.DB           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    results = extract_optuna_hyperparams()
    
    if results:
        with open('optuna_hyperparams_extracted.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✅ Résultats sauvegardés: optuna_hyperparams_extracted.json")
    else:
        print("\n❌ Aucun résultat")

if __name__ == "__main__":
    main()
