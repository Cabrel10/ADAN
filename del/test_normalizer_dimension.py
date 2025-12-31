"""
Test de validation pour Action 2.1 : Correction Mismatch Normalisation

Objectif : Vérifier que le normalizer gère correctement la dimension 20 du portfolio_state
"""

import numpy as np
import sys
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import direct du module sans passer par __init__ (évite erreur gym)
normalizer_path = project_root / "src" / "adan_trading_bot" / "normalization"
sys.path.insert(0, str(normalizer_path))

from observation_normalizer import ObservationNormalizer

def test_normalizer_dimension_20():
    """Test que le normalizer accepte et normalise correctement un vecteur de 20 features"""
    print("\n" + "="*80)
    print("TEST 1: Normaliser portfolio_state (dimension 20)")
    print("="*80)
    
    try:
        # Initialiser le normalizer
        normalizer = ObservationNormalizer()
        
        # Simuler un portfolio_state de dimension 20 (comme dans le monitor)
        portfolio_obs = np.random.rand(20).astype(np.float32)
        
        print(f"✅ Portfolio state créé: shape={portfolio_obs.shape}, dtype={portfolio_obs.dtype}")
        print(f"   Valeurs (premiers 5): {portfolio_obs[:5]}")
        
        # Normaliser
        normalized = normalizer.normalize(portfolio_obs)
        
        # Vérifications
        assert normalized.shape == (20,), f"Shape incorrect: attendu (20,), obtenu {normalized.shape}"
        assert normalized.dtype == np.float32, f"Dtype incorrect: {normalized.dtype}"
        assert not np.isnan(normalized).any(), "NaN détecté dans normalisation"
        assert not np.isinf(normalized).any(), "Inf détecté dans normalisation"
        
        print(f"✅ Normalisation réussie: shape={normalized.shape}")
        print(f"   Valeurs normalisées (premiers 5): {normalized[:5]}")
        print(f"✅ TEST 1: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 1: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_recalibrage():
    """Test que le normalizer se recalibre automatiquement si dimension incorrecte"""
    print("\n" + "="*80)
    print("TEST 2: Auto-recalibrage sur dimension incorrecte")
    print("="*80)
    
    try:
        # Initialiser le normalizer (stats par défaut = 20 features)
        normalizer = ObservationNormalizer()
        
        print(f"✅ Normalizer initialisé avec mean shape: {len(normalizer.mean)}")
        
        # Essayer avec une dimension différente (14 features, comme dans les logs)
        obs_14 = np.random.rand(14).astype(np.float32)
        
        print(f"✅ Observation 14 features créée: {obs_14.shape}")
        print(f"   → Le normalizer devrait détecter le mismatch et se recalibrer automatiquement")
        
        # Normaliser (devrait auto-recalibrer)
        normalized_14 = normalizer.normalize(obs_14)
        
        # Vérifications
        assert normalized_14.shape == (14,), f"Shape incorrect: {normalized_14.shape}"
        assert len(normalizer.mean) == 14, f"Mean pas recalibré: {len(normalizer.mean)}"
        assert len(normalizer.var) == 14, f"Var pas recalibré: {len(normalizer.var)}"
        
        print(f"✅ Auto-recalibrage réussi: mean shape {20} → {len(normalizer.mean)}")
        print(f"✅ TEST 2: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 2: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_monitor():
    """Test d'intégration simulant le comportement du monitor"""
    print("\n" + "="*80)
    print("TEST 3: Intégration avec format monitor (portfolio_state 20)")
    print("="*80)
    
    try:
        normalizer = ObservationNormalizer()
        
        # Simuler la construction du portfolio_state comme dans le monitor
        balance = 29.0
        current_price = 88073.27
        has_position = 1.0
        num_positions = 1
        max_positions = 1
        
        portfolio_obs = np.zeros(20, dtype=np.float32)
        portfolio_obs[0] = balance / 100  # Normalized balance
        portfolio_obs[1] = balance / 100  # Normalized equity
        portfolio_obs[2] = current_price / 100000  # Current price normalized
        portfolio_obs[3] = has_position  # has_position
        portfolio_obs[8] = float(num_positions)  # CRITIQUE: index 8
        portfolio_obs[9] = float(max_positions)  # CRITIQUE: index 9
        
        print(f"✅ Portfolio state simulé: shape={portfolio_obs.shape}")
        print(f"   - balance: {portfolio_obs[0]:.4f}")
        print(f"   - current_price: {portfolio_obs[2]:.4f}")
        print(f"   - has_position: {portfolio_obs[3]:.0f}")
        print(f"   - num_positions (index 8): {portfolio_obs[8]:.0f}")
        print(f"   - max_positions (index 9): {portfolio_obs[9]:.0f}")
        
        # Normaliser
        normalized = normalizer.normalize(portfolio_obs)
        
        # Vérifications
        assert len(portfolio_obs) == 20, f"Portfolio doit avoir 20 features, a {len(portfolio_obs)}"
        assert normalized.shape == (20,), f"Normalized shape incorrect: {normalized.shape}"
        
        print(f"✅ Normalisation intégrée réussie")
        print(f"✅ TEST 3: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 3: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔍 TESTS AUTOMATISÉS - ACTION 2.1: CORRECTION NORMALISATION")
    print("="*80)
    
    results = []
    
    # Exécuter les tests
    results.append(("Normalisation dimension 20", test_normalizer_dimension_20()))
    results.append(("Auto-recalibrage", test_auto_recalibrage()))
    results.append(("Intégration monitor", test_integration_monitor()))
    
    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print("\n" + "="*80)
    print(f"Résultat final: {passed}/{total} tests passés")
    print("="*80)
    
    if passed == total:
        print("\n✅ TOUS LES TESTS SONT PASSÉS - Action 2.1 validée!")
        print("   → Vous pouvez continuer avec Action 2.2 (mode warmup)")
        exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) échoué(s) - NE PAS continuer")
        print("   → Corriger les problèmes avant de passer à Action 2.2")
        exit(1)
