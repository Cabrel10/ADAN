#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour vérifier le chargement et le formatage des données pour l'entraînement.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Chemins des données
data_dir = Path("data/processed")
assets = ["BTC", "ETH", "SOL", "XRP", "ADA"]
timeframes = ["1h"]

# Charger et vérifier les données
for asset in assets:
    for tf in timeframes:
        file_path = data_dir / asset / f"{asset}_{tf}_train.parquet"
        if file_path.exists():
            print(f"\n=== Données pour {asset} ({tf}) ===")
            df = pd.read_parquet(file_path)
            
            # Afficher les informations de base
            print(f"- Taille: {len(df)} lignes x {len(df.columns)} colonnes")
            print(f"- Période: {df.index[0]} à {df.index[-1]}")
            
            # Vérifier les colonnes requises
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"  ATTENTION: Colonnes manquantes: {missing_columns}")
            else:
                print("  ✓ Toutes les colonnes OHLCV sont présentes")
            
            # Vérifier les valeurs manquantes
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                print(f"  ATTENTION: {missing_values} valeurs manquantes au total")
                for col in df.columns:
                    if df[col].isnull().any():
                        print(f"    - {col}: {df[col].isnull().sum()} valeurs manquantes")
            else:
                print("  ✓ Aucune valeur manquante détectée")
            
            # Vérifier les valeurs infinies
            inf_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_values > 0:
                print(f"  ATTENTION: {inf_values} valeurs infinies détectées")
                for col in df.select_dtypes(include=[np.number]).columns:
                    if np.isinf(df[col]).any():
                        print(f"    - {col}: {np.isinf(df[col]).sum()} valeurs infinies")
            else:
                print("  ✓ Aucune valeur infinie détectée")
            
            # Afficher les statistiques de base
            print("\nStatistiques descriptives:")
            print(df[['open', 'high', 'low', 'close', 'volume']].describe().round(2))
            
        else:
            print(f"\nFichier non trouvé: {file_path}")

print("\nVérification des données terminée.")
