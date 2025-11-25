# adan_rl_validation_adapted_2025.py
# Validation anti-overfit ADAPTÉE pour ADAN v1 (Agent PPO avec MultiAssetChunkedEnv)
# Auteur: Gemini Assistant, basé sur la stratégie de l'utilisateur
# Date: 2025-11-25

import pandas as pd
import numpy as np
import gymnasium as gym
import glob
import os
import shutil
import yaml
from stable_baselines3 import PPO

# --- IMPORTANT: Import de l'environnement complexe réel ---
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# --- CONFIGURATION DES CHEMINS ---
# Chemin du modèle trouvé
MODEL_PATH = "/home/morningstar/Documents/trading/bot/bot_pres/model/adan_model_checkpoint_640000_steps.zip"

# Chemins des données originales
DATA_DIR_BASE = "/home/morningstar/Documents/trading/bot/data/processed/indicators"
# On se concentre sur BTCUSDT 5m pour cette validation
ORIGINAL_TRAIN_DATA_PATH = os.path.join(DATA_DIR_BASE, "train/BTCUSDT/5m.parquet")
ORIGINAL_TEST_DATA_PATH = os.path.join(DATA_DIR_BASE, "test/BTCUSDT/5m.parquet")

# Chemin temporaire où le script va écrire les données de test pour que l'environnement les lise
# On utilise le chemin que le DataLoader est configuré pour lire (le fichier 'train')
TEMP_DATA_PATH_FOR_ENV = ORIGINAL_TRAIN_DATA_PATH
BACKUP_DATA_PATH = f"{TEMP_DATA_PATH_FOR_ENV}.bak"

# Chemin de la configuration principale
CONFIG_PATH = "/home/morningstar/Documents/trading/bot/config/config.yaml"

def load_full_dataset():
    """Charge et concatène les données de train et de test."""
    print("Chargement et fusion des datasets train/test...")
    if not os.path.exists(ORIGINAL_TRAIN_DATA_PATH) or not os.path.exists(ORIGINAL_TEST_DATA_PATH):
        print(f"ERREUR: Les fichiers de données originaux n'ont pas été trouvés.")
        print(f"Train: {ORIGINAL_TRAIN_DATA_PATH}")
        print(f"Test: {ORIGINAL_TEST_DATA_PATH}")
        return None
        
    df_train = pd.read_parquet(ORIGINAL_TRAIN_DATA_PATH)
    df_test = pd.read_parquet(ORIGINAL_TEST_DATA_PATH)
    
    full_df = pd.concat([df_train, df_test])
    
    # Assurer que le timestamp est un DatetimeIndex
    if 'timestamp' in full_df.columns:
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        full_df = full_df.set_index('timestamp')
        
    full_df = full_df.sort_index()
    print(f"Dataset complet chargé: {len(full_df)} lignes de {full_df.index.min()} à {full_df.index.max()}")
    return full_df

def backup_original_data():
    """Sauvegarde le fichier de données original."""
    if os.path.exists(TEMP_DATA_PATH_FOR_ENV):
        print(f"Sauvegarde du fichier de données original: {TEMP_DATA_PATH_FOR_ENV} -> {BACKUP_DATA_PATH}")
        shutil.move(TEMP_DATA_PATH_FOR_ENV, BACKUP_DATA_PATH)

def restore_original_data():
    """Restaure le fichier de données original."""
    if os.path.exists(BACKUP_DATA_PATH):
        print(f"Restauration du fichier de données original: {BACKUP_DATA_PATH} -> {TEMP_DATA_PATH_FOR_ENV}")
        shutil.move(BACKUP_DATA_PATH, TEMP_DATA_PATH_FOR_ENV)
    # Supprimer le fichier temporaire s'il existe encore
    if os.path.exists(TEMP_DATA_PATH_FOR_ENV) and not os.path.exists(BACKUP_DATA_PATH):
        os.remove(TEMP_DATA_PATH_FOR_ENV)


def run_validation():
    """Exécute la validation Walk-Forward."""
    
    # --- 1. CHARGEMENT ---
    print(f"Chargement du modèle: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("ERREUR: Fichier modèle introuvable.")
        return
    agent = PPO.load(MODEL_PATH)

    print(f"Chargement de la config: {CONFIG_PATH}")
    if not os.path.exists(CONFIG_PATH):
        print("ERREUR: Fichier de configuration introuvable.")
        return
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    full_df = load_full_dataset()
    if full_df is None:
        return

    # --- 2. SAUVEGARDE ET PRÉPARATION ---
    restore_original_data() # Nettoyage au cas où un précédent run aurait échoué
    backup_original_data()

    try:
        # --- 3. PURGED WALK-FORWARD (Anchored) ---
        windows = [
            ("2021-01-01", "2022-01-01"),
            ("2022-01-01", "2023-01-01"),
            ("2023-01-01", "2024-01-01"),
            ("2024-01-01", "2025-01-01"),
            ("2025-01-01", "2025-11-25"),
        ]
        results = []

        for i, (start, end) in enumerate(windows):
            test_df = full_df[start:end]
            if test_df.empty:
                print(f"\nAucune donnée pour la fenêtre {i+1}: {start} -> {end}. On saute.")
                continue

            print(f"\n--- Fenêtre Walk-Forward {i+1}: Test sur {start} -> {end} ---")
            
            # Écrire la tranche de données pour que l'environnement la lise
            print(f"Préparation des données de test temporaires...")
            test_df.to_parquet(TEMP_DATA_PATH_FOR_ENV)

            # Recréer l'env. Il lira automatiquement le fichier qu'on vient de créer.
            # L'environnement est complexe, on lui passe la config complète.
            env = MultiAssetChunkedEnv(config=config)
            obs, _ = env.reset()
            
            equity = [env.portfolio.initial_capital]
            print("Lancement du backtest sur la fenêtre...")
            while True:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                equity.append(info['portfolio_value'])
                if done or truncated:
                    break
            
            equity = np.array(equity)
            returns = np.diff(equity) / equity[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252*288) if np.std(returns) > 0 else 0
            
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_dd = np.max(drawdown)

            print(f"Résultats Fenêtre {i+1}: Sharpe: {sharpe:.2f} | Max DD: {max_dd:.1%}")
            results.append({"window": i+1, "sharpe": sharpe, "max_dd": max_dd})

    finally:
        # --- 4. RESTAURATION ---
        print("\nNettoyage et restauration des données originales...")
        restore_original_data()

    print("\n--- VALIDATION WALK-FORWARD TERMINÉE ---")
    for res in results:
        print(f"Fenêtre {res['window']}: Sharpe={res['sharpe']:.2f}, MaxDD={res['max_dd']:.1%}")
        
    print("\nNOTE: La partie Block Bootstrap n'a pas été implémentée car elle serait trop lente avec cette méthode.")


if __name__ == "__main__":
    run_validation()
