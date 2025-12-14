#!/usr/bin/env python3
"""Test de l'exécution des trades ADAN"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trade_execution():
    """Test l'exécution des trades"""
    print("💸 TEST D'EXÉCUTION DE TRADES")
    print("="*60)

    # 1. Vérifier les paramètres de trading
    print("📋 PARAMÈTRES DE TRADING:")
    
    # Paramètres DBE depuis la config
    dbe_params = {
        'w1': {'tp': 0.0321, 'sl': 0.0253, 'position': 0.1121},
        'w2': {'tp': 0.0500, 'sl': 0.0250, 'position': 0.2500},
        'w3': {'tp': 0.1800, 'sl': 0.1000, 'position': 0.5000},
        'w4': {'tp': 0.0200, 'sl': 0.0120, 'position': 0.2000}
    }
    
    for worker, params in dbe_params.items():
        print(f"  {worker}: TP={params['tp']:.4f}, SL={params['sl']:.4f}, Position={params['position']:.4f}")

    # 2. Simuler des signaux
    print("\n🎯 SIMULATION DE SIGNAUX:")
    np.random.seed(42)
    signals = np.random.choice([-1, 0, 1], size=10, p=[0.3, 0.4, 0.3])
    confidences = np.random.uniform(0.6, 0.9, size=10)
    
    for i, (signal, conf) in enumerate(zip(signals, confidences)):
        action = "SELL" if signal == -1 else "HOLD" if signal == 0 else "BUY"
        print(f"  Cycle {i+1}: {action} (confiance: {conf:.2f})")

    # 3. Vérifier les conditions de trading
    print("\n🔍 CONDITIONS DE TRADING:")
    capital = 29.0  # Votre capital actuel
    min_trade_value = 11.0  # Minimum configuré
    
    print(f"  Capital disponible: ${capital:.2f}")
    print(f"  Minimum par trade: ${min_trade_value:.2f}")
    
    # Calculer la taille de position pour chaque worker
    for worker, params in dbe_params.items():
        position_value = capital * params['position']
        can_trade = position_value >= min_trade_value
        status = "✅" if can_trade else "❌"
        print(f"  {worker}: Position=${position_value:.2f} {status}")

    # 4. Vérifier la logique d'ensemble
    print("\n🤖 LOGIQUE D'ENSEMBLE:")
    votes = np.random.choice([-1, 0, 1], size=4)
    weights = [0.25, 0.25, 0.25, 0.25]  # Égal pour la démocratie
    ensemble_signal = np.average(votes, weights=weights)
    
    print(f"  Votes: {votes}")
    print(f"  Poids: {weights}")
    print(f"  Signal ensemble: {ensemble_signal:.2f}")

    # 5. Recommandations
    print("\n🎯 RECOMMANDATIONS:")
    if capital < min_trade_value * 2:
        print("1. ⚠️  Capital faible - Risque de ne pas pouvoir trader")
    else:
        print("1. ✅ Capital suffisant")
    print("2. Vérifier que les signaux ne sont pas toujours HOLD")
    print("3. Vérifier la connexion API Binance Testnet")
    print("4. Vérifier les logs 'Trade Exécuté' ou 'Order placed'")

if __name__ == "__main__":
    test_trade_execution()
