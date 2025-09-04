#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script de débogage pour isoler l'erreur d'importation silencieuse."""

import os
import sys

print("--- DEBUG: Démarrage du débogueur d'importation ---")

# Configuration du PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
print(f"--- DEBUG: PYTHONPATH configuré pour inclure : {os.path.join(PROJECT_ROOT, 'src')}")

try:
    print("\n--- DEBUG: Test d'importation séquentielle pour data_loader ---")

    print("1. Importation du package 'data_processing'...")
    import adan_trading_bot.data_processing
    print("   -> SUCCÈS")

    print("2. Importation du module 'data_loader'...")
    from adan_trading_bot.data_processing import data_loader
    print("   -> SUCCÈS")

    print("3. Importation de la classe 'ChunkedDataLoader'...")
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    print("   -> SUCCÈS")

    print("\n--- DEBUG: Toutes les importations de data_loader ont réussi ! ---")

except Exception as e:
    print(f"\n--- DEBUG: Une erreur explicite est survenue : {e} ---")
    import traceback
    traceback.print_exc()
