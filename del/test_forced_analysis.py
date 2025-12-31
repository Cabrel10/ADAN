#!/usr/bin/env python3
"""
Test de l'analyse forcée après fermeture de position
Simule une fermeture de position et vérifie que l'analyse se déclenche
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta

state_file = Path("/mnt/new_data/t10_training/phase2_results/paper_trading_state.json")

print("🧪 TEST: ANALYSE FORCÉE APRÈS FERMETURE DE POSITION")
print("="*70)

# Étape 1: Vérifier l'état initial
print("\n1️⃣  État initial:")
with open(state_file) as f:
    state = json.load(f)

positions = state.get('portfolio', {}).get('positions', [])
print(f"   Positions ouvertes: {len(positions)}")
if positions:
    for pos in positions:
        print(f"   - {pos['pair']}: {pos['side']} @ {pos['entry_price']:.2f}")

# Étape 2: Fermer la position
print("\n2️⃣  Fermeture de la position...")
state['portfolio']['positions'] = []
state['portfolio']['balance'] = 29.0
state['portfolio']['equity'] = 29.0

with open(state_file, 'w') as f:
    json.dump(state, f, indent=2)

print("   ✅ Position fermée dans l'état")

# Étape 3: Attendre et vérifier les logs
print("\n3️⃣  Attente de l'analyse forcée (vérifier les logs)...")
print("   Commande pour surveiller:")
print("   tail -f monitor_corrected.log | grep -E '(ANALYSE FORCÉE|CONSENSUS|worker|BUY|SELL)'")

print("\n4️⃣  Résultats attendus:")
print("   ✅ Message: '🎯 ANALYSE FORCÉE (Position fermée récemment)'")
print("   ✅ Consensus des 4 workers")
print("   ✅ Décision finale (BUY/SELL/HOLD)")
print("   ✅ Nouveau trade exécuté (si signal valide)")

print("\n" + "="*70)
print("✅ Test configuré - Vérifiez les logs du monitor")
print("="*70)
