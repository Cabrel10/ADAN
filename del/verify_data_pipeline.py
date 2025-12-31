#!/usr/bin/env python3
"""Vérification complète du pipeline de données ADAN"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_pipeline():
    """Vérifie chaque étape du pipeline"""
    print("🔍 VÉRIFICATION DU PIPELINE ADAN")
    print("="*60)
    checks = []

    # 1. Vérifier les imports critiques
    try:
        import torch
        checks.append(("PyTorch", "✅", f"Version: {torch.__version__}"))
    except ImportError as e:
        checks.append(("PyTorch", "❌", f"Erreur: {e}"))

    # 2. Vérifier Stable-Baselines3
    try:
        from stable_baselines3 import PPO
        checks.append(("Stable-Baselines3", "✅", "Import réussi"))
    except ImportError as e:
        checks.append(("Stable-Baselines3", "❌", f"Erreur: {e}"))

    # 3. Vérifier le module ADAN
    try:
        from adan_trading_bot.normalization import ObservationNormalizer
        normalizer = ObservationNormalizer()
        checks.append(("Normaliseur ADAN", "✅", f"Chargé"))
    except Exception as e:
        checks.append(("Normaliseur ADAN", "❌", f"Erreur: {e}"))

    # 4. Vérifier Binance API
    try:
        from binance.client import Client
        checks.append(("Binance API", "✅", "Client disponible"))
    except Exception as e:
        checks.append(("Binance API", "⚠️", f"Attention: {e}"))

    # 5. Vérifier les modèles
    model_paths = [
        "/mnt/new_data/t10_training/checkpoints/final/w1",
        "/mnt/new_data/t10_training/checkpoints/final/w2",
        "/mnt/new_data/t10_training/checkpoints/final/w3",
        "/mnt/new_data/t10_training/checkpoints/final/w4"
    ]
    for i, path in enumerate(model_paths, 1):
        if os.path.exists(path + ".zip"):
            checks.append((f"Modèle W{i}", "✅", f"Trouvé"))
        else:
            checks.append((f"Modèle W{i}", "❌", f"Non trouvé"))

    # 6. Vérifier la config
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        checks.append(("Config YAML", "✅", "Trouvée"))
    else:
        checks.append(("Config YAML", "❌", "Non trouvée"))

    # Afficher le rapport
    print("\n📊 RAPPORT DE VÉRIFICATION:")
    for name, status, detail in checks:
        print(f"  {status} {name}: {detail}")

    # Compter les succès
    success = sum(1 for _, status, _ in checks if status == "✅")
    total = len(checks)
    print(f"\n🎯 Résumé: {success}/{total} vérifications réussies")
    
    return success == total

if __name__ == "__main__":
    verify_pipeline()
