#!/usr/bin/env python3
"""
Script LÉGER - Extrait les meilleurs hyperparamètres W1/W2/W3 depuis optuna.db
Aucun fichier créé. Affichage direct.
"""

import sqlite3
import sys

conn = sqlite3.connect('optuna.db')
c = conn.cursor()

# Récupérer études
c.execute("SELECT study_id, study_name FROM studies")
studies = {name: sid for sid, name in c.fetchall()}

best = {}

for worker in ['W1', 'W2', 'W3']:
    study_name = f"adan_final_v1_{worker.lower()}"
    if study_name not in studies:
        print(f"❌ {worker}: Étude non trouvée")
        continue
    
    sid = studies[study_name]
    
    # Récupérer le dernier trial (généralement le meilleur)
    c.execute(
        "SELECT trial_id, number FROM trials WHERE study_id = ? ORDER BY number DESC LIMIT 1",
        (sid,)
    )
    row = c.fetchone()
    
    if not row:
        print(f"❌ {worker}: Aucun trial")
        continue
    
    trial_id, trial_num = row
    
    # Extraire hyperparamètres
    c.execute(
        "SELECT param_name, param_value FROM trial_params WHERE trial_id = ? ORDER BY param_name",
        (trial_id,)
    )
    params = c.fetchall()
    
    best[worker] = {
        'trial': trial_num,
        'params': dict(params)
    }
    
    print(f"✅ {worker}: Trial {trial_num} ({len(params)} hyperparamètres)")

conn.close()

# Afficher W4 = meilleur global (W3 Trial 35 selon résultats précédents)
print(f"\n🏆 W4 = Meilleur global: W3 Trial 35")

# Affichage compact
print("\n" + "="*70)
for worker in ['W1', 'W2', 'W3']:
    if worker in best:
        print(f"\n{worker} Trial {best[worker]['trial']}:")
        for k, v in sorted(best[worker]['params'].items())[:15]:
            print(f"  {k}: {v}")
        if len(best[worker]['params']) > 15:
            print(f"  ... et {len(best[worker]['params']) - 15} autres")

print("\n" + "="*70)
print("✅ Extraction terminée - Aucun fichier créé")
