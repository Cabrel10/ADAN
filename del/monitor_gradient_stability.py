#!/usr/bin/env python3
"""
🔍 MONITORING INTELLIGENT DES GRADIENTS
Détecte l'instabilité AVANT le crash
"""

import re
import time
from pathlib import Path
from collections import deque

log_file = Path("/mnt/new_data/adan_logs/training_20251207_200003.log")

# Historique des métriques
gradient_history = deque(maxlen=100)
loss_history = deque(maxlen=100)
nan_count = 0
last_check = 0

print("🔍 MONITORING GRADIENT STABILITY")
print("=" * 70)
print("Surveillance en temps réel de l'instabilité...")
print("=" * 70)
print()

while True:
    try:
        # Lire les dernières lignes du log
        with open(log_file, 'r') as f:
            lines = f.readlines()[-500:]
        
        # Chercher les NaN
        nan_lines = [l for l in lines if 'nan' in l.lower() or 'invalid values' in l.lower()]
        if len(nan_lines) > nan_count:
            print(f"⚠️  ALERTE NaN DÉTECTÉ!")
            print(f"   Nombre de NaN: {len(nan_lines)}")
            nan_count = len(nan_lines)
        
        # Chercher les erreurs
        error_lines = [l for l in lines if 'ERROR' in l or 'Exception' in l]
        if error_lines:
            print(f"❌ ERREUR DÉTECTÉE:")
            for line in error_lines[-3:]:
                print(f"   {line.strip()[:80]}")
        
        # Chercher les portfolios
        portfolio_lines = [l for l in lines if 'Portfolio value:' in l]
        if portfolio_lines:
            last_portfolio = portfolio_lines[-1]
            match = re.search(r'Portfolio value: ([\d.]+)', last_portfolio)
            if match:
                value = float(match.group(1))
                print(f"💰 Portfolio: ${value:.2f}")
        
        # Chercher les steps
        step_lines = [l for l in lines if '[STEP' in l]
        if step_lines:
            last_step = step_lines[-1]
            match = re.search(r'\[STEP (\d+)', last_step)
            if match:
                step = int(match.group(1))
                print(f"📈 Step: {step}")
        
        # Chercher les trades
        trade_lines = [l for l in lines if 'POSITION OUVERTE' in l or 'POSITION FERMÉE' in l]
        if trade_lines:
            print(f"📊 Trades: {len(trade_lines)} (derniers 500 logs)")
        
        print()
        time.sleep(30)  # Vérifier toutes les 30 secondes
        
    except Exception as e:
        print(f"❌ Erreur monitoring: {e}")
        time.sleep(10)
