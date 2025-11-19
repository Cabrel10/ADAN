#!/usr/bin/env python3
"""Test rapide de tous les imports"""
import sys
sys.path.insert(0, '/content/bot/src')

tests = [
    ("NumPy", "numpy"),
    ("Pandas", "pandas"),
    ("Gymnasium", "gymnasium"),
    ("SB3", "stable_baselines3"),
    ("CCXT", "ccxt"),
    ("PyTorch", "torch"),
    ("ConfigLoader", "adan_trading_bot.common.config_loader"),
    ("FeatureEngineer", "adan_trading_bot.data_processing.feature_engineer"),
]

print("\n" + "="*60)
print("TEST D'IMPORTS")
print("="*60)

for name, module in tests:
    try:
        __import__(module)
        print(f"✅ {name:30s} OK")
    except Exception as e:
        print(f"❌ {name:30s} ERREUR: {str(e)[:40]}")

print("="*60 + "\n")
