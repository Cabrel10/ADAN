import nbformat as nbf
import os

def create_robust_notebook():
    # Créer un nouveau notebook
    nb = nbf.v4.new_notebook()
    
    # Cellule 1 : Titre et description
    markdown_cell = nbf.v4.new_markdown_cell("""# 🚀 ADAN Trading Bot - Solution Robuste

## 📋 Description
Notebook optimisé pour l'entraînement du bot de trading ADAN sur Google Colab.

## 🛠️ Prérequis
- Python 3.8+
- Bibliothèques Python essentielles
- Accès à Google Colab

## 🚀 Comment l'utiliser
1. Téléchargez ce notebook
2. Ouvrez-le dans Google Colab
3. Exécutez toutes les cellules (Runtime -> Run all)
""")

    # Cellule 2 : Installation des dépendances
    code_cell1 = nbf.v4.new_code_cell("""# Installation des dépendances
!pip install -q torch==2.0.0 stable-baselines3==2.0.0 gymnasium==0.29.0 pandas numpy matplotlib
!pip install -q ta-lib
!pip install -q git+https://github.com/Cabrel10/ADAN0.git
""")

    # Cellule 3 : Vérification de l'installation
    code_cell2 = nbf.v4.new_code_cell("""# Vérification des installations
import torch
import stable_baselines3
import gymnasium as gym
import pandas as pd
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"Stable-Baselines3 version: {stable_baselines3.__version__}")
print(f"Gymnasium version: {gym.__version__}")
print(f"Pandas version: {pd.__version__}")
print("✅ Toutes les dépendances sont installées avec succès !")
""")

    # Cellule 4 : Configuration de l'environnement
    code_cell3 = nbf.v4.new_code_cell("""# Configuration de l'environnement
import os
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Configuration de base
config = {
    "data": {
        "timeframes": ["5m", "15m", "1h"],
        "symbols": ["BTC/USDT"],
        "initial_balance": 10000.0,
        "max_position_size": 1000.0,
        "max_trades_per_day": 20
    }
}

# Création de l'environnement
env = MultiAssetChunkedEnv(config)
print("✅ Environnement créé avec succès !")
""")

    # Ajouter les cellules au notebook
    nb.cells = [markdown_cell, code_cell1, code_cell2, code_cell3]
    
    # Sauvegarder le notebook
    output_file = "ADAN_Robust_Launcher.ipynb"
    with open(output_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"✅ Notebook créé avec succès : {output_file}")

if __name__ == "__main__":
    create_robust_notebook()
