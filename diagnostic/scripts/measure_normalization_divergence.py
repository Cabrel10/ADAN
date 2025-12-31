#!/usr/bin/env python3
"""
Checkpoint 1.3 : Mesure de la divergence de normalisation
Objectif : Comparer la normalisation d'entraînement vs paper trading
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ============================================================================
# ÉTAPE 1 : Extraire la fonction build_observation du paper trading
# ============================================================================

def extract_paper_trading_normalization(raw_features):
    """
    Reproduit la normalisation ACTUELLE du paper trading
    (normalisation manuelle avec fenêtre glissante)
    """
    # Normalisation manuelle : mean/std sur fenêtre glissante
    window_size = 100
    if len(raw_features) < window_size:
        window = raw_features
    else:
        window = raw_features[-window_size:]
    
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    
    # Éviter division par zéro
    std = np.where(std < 1e-8, 1e-8, std)
    
    normalized = (raw_features - mean) / std
    
    return normalized, mean, std


# ============================================================================
# ÉTAPE 2 : Charger VecNormalize et normaliser avec la méthode correcte
# ============================================================================

class DummyTradingEnv:
    """Environnement dummy pour VecNormalize"""
    def __init__(self):
        import gymnasium as gym
        self.observation_space = gym.spaces.Dict({
            '5m': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(100, 20)),
            '1h': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(100, 20)),
            '4h': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(50, 20)),
            'portfolio_state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
        })
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
    
    def reset(self, seed=None, options=None):
        obs = {
            '5m': np.zeros((100, 20)),
            '1h': np.zeros((100, 20)),
            '4h': np.zeros((50, 20)),
            'portfolio_state': np.zeros(10)
        }
        return obs, {}
    
    def step(self, action):
        raise NotImplementedError("Dummy environment")


def load_vecnormalize(worker_id):
    """Charge les statistiques VecNormalize pour un worker"""
    vecnorm_path = project_root / "models" / f"worker_{worker_id}" / "vecnormalize.pkl"
    
    if not vecnorm_path.exists():
        print(f"❌ Fichier non trouvé : {vecnorm_path}")
        return None
    
    try:
        dummy_env = DummyVecEnv([lambda: DummyTradingEnv()])
        vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
        vecnorm.training = False
        vecnorm.norm_reward = False
        print(f"✅ VecNormalize chargé pour worker_{worker_id}")
        return vecnorm
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None


# ============================================================================
# ÉTAPE 3 : Générer des observations de test
# ============================================================================

def generate_test_observation(shape=(100, 20)):
    """Génère une observation de test réaliste"""
    # Simuler des données de marché (prix, volume, indicateurs)
    np.random.seed(42)
    
    # Tendance + bruit
    trend = np.linspace(0, 1, shape[0])
    noise = np.random.normal(0, 0.1, shape)
    
    obs = trend[:, np.newaxis] + noise
    
    # Normaliser entre 0 et 1 pour simuler des données réelles
    obs = (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)
    
    return obs.astype(np.float32)


# ============================================================================
# ÉTAPE 4 : Mesurer la divergence
# ============================================================================

def measure_divergence(worker_id):
    """Mesure la divergence pour un worker"""
    print(f"\n{'='*70}")
    print(f"Mesure de divergence pour worker_{worker_id}")
    print(f"{'='*70}")
    
    # Charger VecNormalize
    vecnorm = load_vecnormalize(worker_id)
    if vecnorm is None:
        return None
    
    # Générer une observation de test
    print("\n1️⃣  Génération de l'observation de test...")
    raw_obs_5m = generate_test_observation((100, 20))
    raw_obs_1h = generate_test_observation((100, 20))
    raw_obs_4h = generate_test_observation((50, 20))
    raw_obs_portfolio = np.random.rand(10).astype(np.float32)
    
    print(f"   ✓ Observation 5m shape: {raw_obs_5m.shape}")
    print(f"   ✓ Observation 1h shape: {raw_obs_1h.shape}")
    print(f"   ✓ Observation 4h shape: {raw_obs_4h.shape}")
    print(f"   ✓ Portfolio state shape: {raw_obs_portfolio.shape}")
    
    # Méthode 1 : Normalisation ACTUELLE (paper trading)
    print("\n2️⃣  Normalisation ACTUELLE (paper trading - fenêtre glissante)...")
    norm_paper_5m, mean_5m, std_5m = extract_paper_trading_normalization(raw_obs_5m)
    norm_paper_1h, mean_1h, std_1h = extract_paper_trading_normalization(raw_obs_1h)
    norm_paper_4h, mean_4h, std_4h = extract_paper_trading_normalization(raw_obs_4h)
    norm_paper_portfolio = (raw_obs_portfolio - raw_obs_portfolio.mean()) / (raw_obs_portfolio.std() + 1e-8)
    
    print(f"   ✓ 5m - mean: {mean_5m.mean():.6f}, std: {std_5m.mean():.6f}")
    print(f"   ✓ 1h - mean: {mean_1h.mean():.6f}, std: {std_1h.mean():.6f}")
    print(f"   ✓ 4h - mean: {mean_4h.mean():.6f}, std: {std_4h.mean():.6f}")
    
    # Méthode 2 : Normalisation CORRECTE (VecNormalize)
    print("\n3️⃣  Normalisation CORRECTE (VecNormalize)...")
    obs_dict = {
        '5m': raw_obs_5m[np.newaxis, :, :],  # Ajouter batch dimension
        '1h': raw_obs_1h[np.newaxis, :, :],
        '4h': raw_obs_4h[np.newaxis, :, :],
        'portfolio_state': raw_obs_portfolio[np.newaxis, :]
    }
    
    try:
        norm_vecnorm = vecnorm.normalize_obs(obs_dict)
        print(f"   ✓ Normalisation VecNormalize réussie")
        
        # Extraire les observations normalisées
        norm_vecnorm_5m = norm_vecnorm['5m'][0]
        norm_vecnorm_1h = norm_vecnorm['1h'][0]
        norm_vecnorm_4h = norm_vecnorm['4h'][0]
        norm_vecnorm_portfolio = norm_vecnorm['portfolio_state'][0]
    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        return None
    
    # Calculer les divergences
    print("\n4️⃣  Calcul des divergences...")
    
    divergence_5m = np.linalg.norm(norm_paper_5m - norm_vecnorm_5m)
    divergence_1h = np.linalg.norm(norm_paper_1h - norm_vecnorm_1h)
    divergence_4h = np.linalg.norm(norm_paper_4h - norm_vecnorm_4h)
    divergence_portfolio = np.linalg.norm(norm_paper_portfolio - norm_vecnorm_portfolio)
    
    divergence_total = np.sqrt(
        divergence_5m**2 + divergence_1h**2 + divergence_4h**2 + divergence_portfolio**2
    )
    
    print(f"   📊 Divergence 5m: {divergence_5m:.6f}")
    print(f"   📊 Divergence 1h: {divergence_1h:.6f}")
    print(f"   📊 Divergence 4h: {divergence_4h:.6f}")
    print(f"   📊 Divergence portfolio: {divergence_portfolio:.6f}")
    print(f"   📊 Divergence TOTALE: {divergence_total:.6f}")
    
    # Interprétation
    print("\n5️⃣  Interprétation...")
    if divergence_total > 0.1:
        print(f"   ⚠️  DIVERGENCE CRITIQUE (> 0.1)")
        print(f"   → Le problème de normalisation est CONFIRMÉ")
        status = "CRITICAL"
    elif divergence_total > 0.01:
        print(f"   ⚠️  Divergence modérée (0.01 - 0.1)")
        print(f"   → Problème détecté mais moins sévère")
        status = "MODERATE"
    else:
        print(f"   ✅ Divergence acceptable (< 0.01)")
        print(f"   → Pas de problème de normalisation détecté")
        status = "OK"
    
    return {
        'worker_id': worker_id,
        'divergence_5m': float(divergence_5m),
        'divergence_1h': float(divergence_1h),
        'divergence_4h': float(divergence_4h),
        'divergence_portfolio': float(divergence_portfolio),
        'divergence_total': float(divergence_total),
        'status': status,
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CHECKPOINT 1.3 : Mesure de la divergence de normalisation")
    print("="*70)
    
    # Créer le répertoire de résultats
    results_dir = project_root / "diagnostic" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Mesurer pour tous les workers
    all_results = []
    
    for worker_id in range(1, 5):  # w1, w2, w3, w4
        result = measure_divergence(worker_id)
        if result:
            all_results.append(result)
    
    # Sauvegarder les résultats
    print(f"\n{'='*70}")
    print("Sauvegarde des résultats...")
    print(f"{'='*70}")
    
    results_file = results_dir / "divergence_report.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Résultats sauvegardés dans : {results_file}")
    
    # Résumé
    print(f"\n{'='*70}")
    print("RÉSUMÉ")
    print(f"{'='*70}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{df.to_string(index=False)}")
        
        # Statistiques globales
        print(f"\nStatistiques globales :")
        print(f"  Divergence moyenne : {df['divergence_total'].mean():.6f}")
        print(f"  Divergence max : {df['divergence_total'].max():.6f}")
        print(f"  Divergence min : {df['divergence_total'].min():.6f}")
        
        # Décision
        if df['divergence_total'].mean() > 0.1:
            print(f"\n🔴 DIAGNOSTIC : PROBLÈME DE NORMALISATION CONFIRMÉ")
            print(f"   → Divergence moyenne > 0.1")
            print(f"   → Passer à la Phase 2 (correction)")
        else:
            print(f"\n🟡 DIAGNOSTIC : Divergence modérée")
            print(f"   → Investiguer d'autres causes possibles")
    else:
        print("❌ Aucun résultat généré")


if __name__ == "__main__":
    main()
