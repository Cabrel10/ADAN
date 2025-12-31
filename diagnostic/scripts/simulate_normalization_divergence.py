#!/usr/bin/env python3
"""
Checkpoint 1.3 : Simulation de la divergence de normalisation
Objectif : Mesurer la divergence THÉORIQUE entre les deux méthodes
(sans dépendre des fichiers vecnormalize.pkl)
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# SIMULATION : Reproduire le comportement de VecNormalize
# ============================================================================

class SimulatedVecNormalize:
    """Simule le comportement de VecNormalize avec des statistiques réalistes"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # Statistiques typiques d'un entraînement RL
        # (accumulées sur 350k steps)
        self.obs_mean_5m = np.random.randn(20) * 0.5
        self.obs_var_5m = np.abs(np.random.randn(20)) + 0.5
        
        self.obs_mean_1h = np.random.randn(20) * 0.5
        self.obs_var_1h = np.abs(np.random.randn(20)) + 0.5
        
        self.obs_mean_4h = np.random.randn(20) * 0.5
        self.obs_var_4h = np.abs(np.random.randn(20)) + 0.5
        
        self.obs_mean_portfolio = np.random.randn(10) * 0.1
        self.obs_var_portfolio = np.abs(np.random.randn(10)) + 0.1
    
    def normalize_obs(self, obs_dict):
        """Normalise une observation avec les statistiques accumulées"""
        normalized = {}
        
        # Normalisation : (obs - mean) / sqrt(var + eps)
        eps = 1e-8
        
        normalized['5m'] = (obs_dict['5m'] - self.obs_mean_5m) / np.sqrt(self.obs_var_5m + eps)
        normalized['1h'] = (obs_dict['1h'] - self.obs_mean_1h) / np.sqrt(self.obs_var_1h + eps)
        normalized['4h'] = (obs_dict['4h'] - self.obs_mean_4h) / np.sqrt(self.obs_var_4h + eps)
        normalized['portfolio_state'] = (obs_dict['portfolio_state'] - self.obs_mean_portfolio) / np.sqrt(self.obs_var_portfolio + eps)
        
        return normalized


# ============================================================================
# MÉTHODE 1 : Normalisation ACTUELLE (paper trading)
# ============================================================================

def normalize_paper_trading(obs_dict, window_size=100):
    """
    Reproduit la normalisation ACTUELLE du paper trading
    (normalisation manuelle avec fenêtre glissante)
    """
    normalized = {}
    eps = 1e-8
    
    for key in obs_dict:
        obs = obs_dict[key]
        
        # Fenêtre glissante
        if len(obs) < window_size:
            window = obs
        else:
            window = obs[-window_size:]
        
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        std = np.where(std < eps, eps, std)
        
        normalized[key] = (obs - mean) / std
    
    return normalized


# ============================================================================
# GÉNÉRATION DE DONNÉES DE TEST
# ============================================================================

def generate_realistic_observation():
    """Génère une observation réaliste de marché"""
    np.random.seed(42)
    
    # Simuler une série temporelle réaliste
    # (prix avec tendance + bruit)
    
    obs_5m = np.random.randn(100, 20) * 0.1 + np.linspace(0, 1, 100)[:, np.newaxis]
    obs_1h = np.random.randn(100, 20) * 0.1 + np.linspace(0, 1, 100)[:, np.newaxis]
    obs_4h = np.random.randn(50, 20) * 0.1 + np.linspace(0, 1, 50)[:, np.newaxis]
    obs_portfolio = np.random.randn(10) * 0.1 + 0.5
    
    return {
        '5m': obs_5m.astype(np.float32),
        '1h': obs_1h.astype(np.float32),
        '4h': obs_4h.astype(np.float32),
        'portfolio_state': obs_portfolio.astype(np.float32)
    }


# ============================================================================
# MESURE DE DIVERGENCE
# ============================================================================

def measure_divergence_simulated(worker_id):
    """Mesure la divergence pour un worker (simulation)"""
    print(f"\n{'='*70}")
    print(f"Mesure de divergence pour worker_{worker_id} (SIMULATION)")
    print(f"{'='*70}")
    
    # Générer une observation de test
    print("\n1️⃣  Génération de l'observation de test...")
    obs_dict = generate_realistic_observation()
    print(f"   ✓ Observation 5m shape: {obs_dict['5m'].shape}")
    print(f"   ✓ Observation 1h shape: {obs_dict['1h'].shape}")
    print(f"   ✓ Observation 4h shape: {obs_dict['4h'].shape}")
    print(f"   ✓ Portfolio state shape: {obs_dict['portfolio_state'].shape}")
    
    # Méthode 1 : Normalisation ACTUELLE (paper trading)
    print("\n2️⃣  Normalisation ACTUELLE (paper trading - fenêtre glissante)...")
    norm_paper = normalize_paper_trading(obs_dict)
    print(f"   ✓ Normalisation paper trading réussie")
    
    # Méthode 2 : Normalisation CORRECTE (VecNormalize simulé)
    print("\n3️⃣  Normalisation CORRECTE (VecNormalize simulé)...")
    vecnorm = SimulatedVecNormalize(seed=worker_id)
    norm_vecnorm = vecnorm.normalize_obs(obs_dict)
    print(f"   ✓ Normalisation VecNormalize réussie")
    
    # Calculer les divergences
    print("\n4️⃣  Calcul des divergences...")
    
    divergence_5m = np.linalg.norm(norm_paper['5m'] - norm_vecnorm['5m'])
    divergence_1h = np.linalg.norm(norm_paper['1h'] - norm_vecnorm['1h'])
    divergence_4h = np.linalg.norm(norm_paper['4h'] - norm_vecnorm['4h'])
    divergence_portfolio = np.linalg.norm(norm_paper['portfolio_state'] - norm_vecnorm['portfolio_state'])
    
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
        'timestamp': datetime.now().isoformat(),
        'note': 'SIMULATION (données synthétiques)'
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CHECKPOINT 1.3 : Mesure de la divergence de normalisation (SIMULATION)")
    print("="*70)
    print("\n⚠️  NOTE : Cette simulation utilise des données synthétiques")
    print("   Les résultats réels dépendront des données d'entraînement réelles")
    
    # Créer le répertoire de résultats
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "diagnostic" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Mesurer pour tous les workers
    all_results = []
    
    for worker_id in range(1, 5):  # w1, w2, w3, w4
        result = measure_divergence_simulated(worker_id)
        if result:
            all_results.append(result)
    
    # Sauvegarder les résultats
    print(f"\n{'='*70}")
    print("Sauvegarde des résultats...")
    print(f"{'='*70}")
    
    results_file = results_dir / "divergence_report_simulated.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Résultats sauvegardés dans : {results_file}")
    
    # Résumé
    print(f"\n{'='*70}")
    print("RÉSUMÉ")
    print(f"{'='*70}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{df[['worker_id', 'divergence_5m', 'divergence_1h', 'divergence_4h', 'divergence_total', 'status']].to_string(index=False)}")
        
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
        elif df['divergence_total'].mean() > 0.01:
            print(f"\n🟡 DIAGNOSTIC : Divergence modérée")
            print(f"   → Problème détecté mais moins sévère")
        else:
            print(f"\n✅ DIAGNOSTIC : Pas de problème détecté")
            print(f"   → Investiguer d'autres causes possibles")
    else:
        print("❌ Aucun résultat généré")


if __name__ == "__main__":
    main()
