#!/usr/bin/env python3
"""
Checkpoint 1.3 (Corrigé) : Mesure de la divergence de normalisation.
Ce script charge les artefacts réels (données, stats de normalisation)
et mesure la différence entre la méthode de normalisation de l'entraînement (correcte)
et une méthode manuelle (incorrecte, similaire à celle du paper trading).
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- Configuration des Chemins (relatifs à la racine du projet) ---
CHECKPOINTS_DIR = Path("./checkpoints/final")
DATA_DIR = Path("./data/processed/indicators/train/BTCUSDT")
RESULTS_DIR = Path("./diagnostic/results")

# --- Environnement Factice (conforme à l'entraînement) ---
class DummyTradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            '5m': spaces.Box(low=-np.inf, high=np.inf, shape=(20, 14), dtype=np.float32),
            '1h': spaces.Box(low=-np.inf, high=np.inf, shape=(20, 14), dtype=np.float32),
            '4h': spaces.Box(low=-np.inf, high=np.inf, shape=(20, 14), dtype=np.float32),
            'portfolio_state': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = {k: v.sample() for k, v in self.observation_space.spaces.items()}
        return obs, {}

    def step(self, action):
        obs = {k: v.sample() for k, v in self.observation_space.spaces.items()}
        return obs, 0, False, False, {}

# --- Fonctions de Test ---

def load_vecnormalize(worker_id: str) -> VecNormalize:
    """Charge le fichier vecnormalize.pkl pour un worker."""
    vecnorm_path = CHECKPOINTS_DIR / f"{worker_id}_vecnormalize.pkl"
    if not vecnorm_path.exists():
        print(f"❌ Fichier VecNormalize non trouvé pour {worker_id} à '{vecnorm_path}'")
        return None
    
    dummy_env = DummyVecEnv([lambda: DummyTradingEnv()])
    vecnorm_env = VecNormalize.load(str(vecnorm_path), dummy_env)
    vecnorm_env.training = False
    vecnorm_env.norm_reward = False
    print(f"✅ Stats VecNormalize chargées pour {worker_id}.")
    return vecnorm_env

def get_raw_observation_sample() -> dict:
    """Charge un échantillon de données brutes et le formate en observation."""
    data_path = DATA_DIR / "5m.parquet"
    if not data_path.exists():
        print(f"❌ Fichier de données non trouvé à '{data_path}'")
        return None

    df = pd.read_parquet(data_path)
    print(f"✅ Fichier de données '{data_path}' chargé (shape: {df.shape}).")
    
    feature_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])][:14]
    if len(feature_cols) < 14:
        print(f"❌ Données insuffisantes, seulement {len(feature_cols)} features numériques trouvées.")
        return None

    sample_df = df[feature_cols].tail(20).astype(np.float32)

    raw_obs = {
        '5m': sample_df.to_numpy(),
        '1h': sample_df.to_numpy(),
        '4h': sample_df.to_numpy(),
        'portfolio_state': np.random.rand(20).astype(np.float32)
    }
    print("✅ Échantillon d'observation brute créé.")
    return raw_obs

def normalize_manually(obs: dict) -> dict:
    """Simule la normalisation manuelle incorrecte (fenêtre glissante)."""
    normalized_obs = {}
    for key, value in obs.items():
        mean = np.mean(value, axis=0)
        std = np.std(value, axis=0)
        std[std < 1e-8] = 1e-8
        normalized_obs[key] = (value - mean) / std
    print("✅ Observation normalisée avec la méthode manuelle (incorrecte).")
    return normalized_obs

def main():
    print("\n" + "="*80)
    print("--- Checkpoint 1.3 (Corrigé): Mesure de la Divergence de Normalisation ---")
    print("="*80)

    worker_id = "w1"
    
    vecnorm_env = load_vecnormalize(worker_id)
    if not vecnorm_env:
        return

    raw_obs = get_raw_observation_sample()
    if not raw_obs:
        return

    correctly_normalized_obs = vecnorm_env.normalize_obs(raw_obs)
    print("✅ Observation normalisée avec la méthode VecNormalize (correcte).")

    incorrectly_normalized_obs = normalize_manually(raw_obs)

    print("\n--- Calcul de la Divergence (w1, timeframe '5m') ---")
    
    correct_tensor = correctly_normalized_obs['5m']
    incorrect_tensor = incorrectly_normalized_obs['5m']

    divergence_abs = np.linalg.norm(correct_tensor - incorrect_tensor)
    divergence_rel = divergence_abs / (np.linalg.norm(correct_tensor) + 1e-8)

    report = {
        'worker_id': worker_id,
        'timeframe': '5m',
        'divergence_absolute': float(divergence_abs),
        'divergence_relative_percent': float(divergence_rel * 100),
        'timestamp': datetime.now().isoformat()
    }

    print(f"   - Divergence Absolue: {report['divergence_absolute']:.4f}")
    print(f"   - Divergence Relative: {report['divergence_relative_percent']:.2f}%")

    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / "divergence_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\n✅ Rapport de divergence sauvegardé dans '{report_path}'")

    print("\n--- Conclusion du Checkpoint 1.3 ---")
    if divergence_abs > 0.1:
        print("🔴 CONFIRMÉ : La divergence est CRITIQUE (> 0.1).")
        print("   Le problème de normalisation est la cause racine la plus probable des échecs de performance.")
    else:
        print("🟢 La divergence est faible. Le problème de normalisation n'est peut-être pas la cause principale.")

if __name__ == "__main__":
    main()