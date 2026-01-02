#!/usr/bin/env python3
"""
Force la fermeture de la position pour tester l'analyse forcée
"""

import json
from pathlib import Path
from datetime import datetime, timedelta

state_file = Path("/mnt/new_data/t10_training/phase2_results/paper_trading_state.json")

# Charger l'état
with open(state_file) as f:
    state = json.load(f)

# Vérifier les positions
positions = state.get('portfolio', {}).get('positions', [])
print(f"Positions actuelles: {len(positions)}")

if positions:
    for pos in positions:
        print(f"  - {pos['pair']}: {pos['side']} @ {pos['entry_price']:.2f}")
    
    # Forcer la fermeture
    print(f"\n🔄 Fermeture forcée...")
    state['portfolio']['positions'] = []
    state['portfolio']['balance'] = 29.0
    state['portfolio']['equity'] = 29.0
    
    # Sauvegarder
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"✅ Position fermée - État mis à jour")
else:
    print(f"⚠️  Aucune position à fermer")

print(f"\nÉtat final:")
print(f"  Balance: ${state['portfolio'].get('balance', 0):.2f}")
print(f"  Positions: {len(state['portfolio'].get('positions', []))}")

