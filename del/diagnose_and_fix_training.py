#!/usr/bin/env python3
"""
🔍 DIAGNOSTIC ET CORRECTION DU PROBLÈME NaN
Analyse les logs et propose des solutions
"""

import json
import re
from pathlib import Path
from datetime import datetime

def analyze_logs():
    log_file = Path("/mnt/new_data/adan_logs/training_20251207_121358.log")
    
    print("🔍 ANALYSE DU PROBLÈME NaN")
    print("=" * 60)
    
    # Lire les dernières lignes
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Chercher les erreurs
    nan_errors = [l for l in lines if 'nan' in l.lower() or 'invalid values' in l.lower()]
    
    print(f"\n📊 Statistiques:")
    print(f"   Total lignes: {len(lines)}")
    print(f"   Erreurs NaN trouvées: {len(nan_errors)}")
    print(f"   Taille log: {log_file.stat().st_size / 1e9:.2f} GB")
    
    # Chercher les hyperparamètres Optuna
    optuna_lines = [l for l in lines if 'Trial' in l or 'learning_rate' in l.lower()]
    if optuna_lines:
        print(f"\n🎯 Hyperparamètres détectés:")
        for line in optuna_lines[-5:]:
            if 'Trial' in line:
                print(f"   {line.strip()[:100]}")
    
    # Chercher les étapes avant crash
    step_lines = [l for l in lines if '[STEP' in l]
    if step_lines:
        last_step = step_lines[-1]
        print(f"\n📈 Dernier step avant crash:")
        print(f"   {last_step.strip()[:100]}")
    
    print("\n" + "=" * 60)
    print("🔧 SOLUTIONS PROPOSÉES:")
    print("=" * 60)
    print("""
1. ✅ RÉDUIRE LE LEARNING RATE
   - Les hyperparamètres Optuna sont trop agressifs
   - Solution: Diviser par 2-5x

2. ✅ AJOUTER GRADIENT CLIPPING
   - Limiter les gradients pour éviter l'explosion
   - Solution: max_grad_norm = 0.5

3. ✅ NORMALISER LES ENTRÉES
   - Les observations peuvent avoir des valeurs extrêmes
   - Solution: Vérifier la normalisation des données

4. ✅ RÉDUIRE LA TAILLE DU BATCH
   - Batch trop grand = instabilité
   - Solution: Réduire de 64 à 32

5. ✅ AJOUTER LAYER NORMALIZATION
   - Stabiliser les activations du réseau
   - Solution: Ajouter LayerNorm dans le réseau

6. ✅ UTILISER UN LEARNING RATE SCHEDULER
   - Réduire le LR progressivement
   - Solution: Utiliser CosineAnnealingLR
    """)

def check_optuna_params():
    """Vérifier les paramètres Optuna sauvegardés"""
    optuna_dir = Path("optuna_results")
    if optuna_dir.exists():
        print("\n📁 Fichiers Optuna trouvés:")
        for f in optuna_dir.glob("*.pkl"):
            print(f"   {f.name}")

if __name__ == "__main__":
    analyze_logs()
    check_optuna_params()
    
    print("\n" + "=" * 60)
    print("🚀 PROCHAINES ÉTAPES:")
    print("=" * 60)
    print("""
1. Éditer config/config.yaml:
   - Réduire learning_rate: 0.0003 → 0.00005
   - Ajouter max_grad_norm: 0.5
   - Réduire batch_size: 64 → 32

2. Relancer l'entraînement:
   ./launch_training_detached.sh

3. Monitorer les logs:
   tail -f /mnt/new_data/adan_logs/training_*.log
    """)
