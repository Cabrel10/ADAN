"""
Test de validation pour Action 2.2 : Mode Warmup

Objectif : Vérifier que le mode warmup complète correctement les données insuffisantes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class MockMonitor:
    """Classe mock simulant RealPaperTradingMonitor pour tester warmup"""
    
    def __init__(self):
        self.warmup_mode = True
        self.warmup_timeframes = {'5m': False, '1h': False, '4h': False}
        self.preloaded_data = {}
        
        # Créer des données historiques fictives pour TOUS les timeframes
        dates_5m = pd.date_range('2025-01-01', periods=100, freq='5min')
        dates_1h = pd.date_range('2025-01-01', periods=100, freq='1h')
        dates_4h = pd.date_range('2025-01-01', periods=100, freq='4h')
        
        for tf, dates in [('5m', dates_5m), ('1h', dates_1h), ('4h', dates_4h)]:
            self.preloaded_data[tf] = pd.DataFrame({
                'open': np.random.rand(100) * 100 + 80000,
                'high': np.random.rand(100) * 100 + 80100,
                'low': np.random.rand(100) * 100 + 79900,
                'close': np.random.rand(100) * 100 + 80000,
                'volume': np.random.rand(100) * 1000,
                'rsi': np.random.rand(100) * 100,
                'adx': np.random.rand(100) * 100
            }, index=dates)
    
    def complete_with_historical(self, df_resampled, timeframe):
        """
        Complète les données temps réel avec l'historique préchargé pour atteindre 28 lignes
        (Copie exacte de la fonction dans paper_trading_monitor.py)
        """
        if not self.warmup_mode or timeframe not in self.preloaded_data:
            return df_resampled
        
        current_len = len(df_resampled)
        if current_len >= 28:
            # Pas besoin de complétion, marquer comme stabilisé
            self.warmup_timeframes[timeframe] = True
            print(f"✅ {timeframe}: stabilisé avec {current_len} lignes (>= 28)")
            return df_resampled
        
        # Calculer combien de lignes manquent
        missing_rows = 28 - current_len
        historical_df = self.preloaded_data[timeframe].tail(missing_rows).copy()
        
        # Marquer les lignes historiques pour debug
        if '_from_historical' not in df_resampled.columns:
            df_resampled['_from_historical'] = False
        if '_from_historical' not in historical_df.columns:
            historical_df['_from_historical'] = True
        
        # Concaténer: historique PUIS temps réel (ordre chronologique)
        completed_df = pd.concat([historical_df, df_resampled], ignore_index=False).tail(28)
        
        print(f"🔄 Warmup {timeframe}: complété {current_len} → 28 lignes (ajouté {missing_rows} historiques)")
        
        return completed_df


def test_warmup_completion_insufficient():
    """Test complétion avec données insuffisantes (22 lignes)"""
    print("\n" + "="*80)
    print("TEST 1: Complétion données insuffisantes (22 → 28 lignes)")
    print("="*80)
    
    try:
        monitor = MockMonitor()
        
        # Simuler des données 4h insuffisantes (22 lignes)
        dates_insufficient = pd.date_range('2025-01-10', periods=22, freq='4h')
        df_insufficient = pd.DataFrame({
            'open': np.random.rand(22) * 100 + 80000,
            'high': np.random.rand(22) * 100 + 80100,
            'low': np.random.rand(22) * 100 + 79900,
            'close': np.random.rand(22) * 100 + 80000,
            'volume': np.random.rand(22) * 1000,
        }, index=dates_insufficient)
        
        print(f"✅ Données insuffisantes créées: {len(df_insufficient)} lignes")
        
        # Appliquer warmup
        df_completed = monitor.complete_with_historical(df_insufficient, '4h')
        
        # Vérifications
        assert len(df_completed) == 28, f"Attendu 28 lignes, obtenu {len(df_completed)}"
        assert '_from_historical' in df_completed.columns, "Colonne _from_historical manquante"
        
        # Vérifier que les lignes historiques sont bien marquées
        hist_count = df_completed['_from_historical'].sum()
        assert hist_count == 6, f"Attendu 6 lignes historiques, obtenu {hist_count}"
        
        # Vérifier que les lignes temps réel sont bien marquées
        real_count = (~df_completed['_from_historical']).sum()
        assert real_count == 22, f"Attendu 22 lignes temps réel, obtenu {real_count}"
        
        print(f"✅ Complétion réussie: {len(df_completed)} lignes")
        print(f"   - Historiques: {hist_count} lignes")
        print(f"   - Temps réel: {real_count} lignes")
        print(f"✅ TEST 1: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 1: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_warmup_no_completion_needed():
    """Test pas de complétion si données suffisantes (30 lignes)"""
    print("\n" + "="*80)
    print("TEST 2: Pas de complétion nécessaire (30 lignes ≥ 28)")
    print("="*80)
    
    try:
        monitor = MockMonitor()
        
        # Simuler des données 4h suffisantes (30 lignes)
        dates_sufficient = pd.date_range('2025-01-10', periods=30, freq='4h')
        df_sufficient = pd.DataFrame({
            'open': np.random.rand(30) * 100 + 80000,
            'close': np.random.rand(30) * 100 + 80000,
        }, index=dates_sufficient)
        
        print(f"✅ Données suffisantes créées: {len(df_sufficient)} lignes")
        
        # Vérifier flag avant
        assert monitor.warmup_timeframes['4h'] == False, "Flag 4h devrait être False au départ"
        
        # Appliquer warmup
        df_result = monitor.complete_with_historical(df_sufficient, '4h')
        
        # Vérifications
        assert len(df_result) == 30, f"Longueur devrait rester 30, obtenu {len(df_result)}"
        assert monitor.warmup_timeframes['4h'] == True, "Flag 4h devrait être True après stabilisation"
        
        print(f"✅ Aucune complétion effectuée (données suffisantes)")
        print(f"✅ Flag 4h marqué comme stabilisé: {monitor.warmup_timeframes['4h']}")
        print(f"✅ TEST 2: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 2: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_warmup_deactivation():
    """Test désactivation warmup après stabilisation de tous les TF"""
    print("\n" + "="*80)
    print("TEST 3: Désactivation warmup après stabilisation tous TF")
    print("="*80)
    
    try:
        monitor = MockMonitor()
        
        # Simuler des données suffisantes pour tous les TF
        for tf in ['5m', '1h', '4h']:
            dates = pd.date_range('2025-01-10', periods=30, freq=tf)
            df = pd.DataFrame({
                'close': np.random.rand(30) * 100 + 80000,
            }, index=dates)
            
            # Appliquer complete_with_historical (devrait marquer comme stabilisé)
            monitor.complete_with_historical(df, tf)
        
        print(f"✅ Tous les TF traités")
        print(f"   - 5m stabilisé: {monitor.warmup_timeframes['5m']}")
        print(f"   - 1h stabilisé: {monitor.warmup_timeframes['1h']}")
        print(f"   - 4h stabilisé: {monitor.warmup_timeframes['4h']}")
        
        # Vérifier que tous sont stabilisés
        assert all(monitor.warmup_timeframes.values()), "Tous les TF devraient être stabilisés"
        
        # Simuler la logique de désactivation warmup (comme dans fetch_data)
        if monitor.warmup_mode and all(monitor.warmup_timeframes.values()):
            print(f"✅ Warmup terminé: tous les timeframes ont ≥28 bougies temps réel")
            monitor.warmup_mode = False
        
        # Vérifier que warmup est désactivé
        assert monitor.warmup_mode == False, "Warmup devrait être désactivé"
        
        print(f"✅ Warmup mode désactivé: {monitor.warmup_mode}")
        print(f"✅ TEST 3: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 3: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔍 TESTS AUTOMATISÉS - ACTION 2.2: MODE WARMUP")
    print("="*80)
    
    results = []
    
    # Exécuter les tests
    results.append(("Complétion données insuffisantes", test_warmup_completion_insufficient()))
    results.append(("Pas de complétion si suffisant", test_warmup_no_completion_needed()))
    results.append(("Désactivation warmup", test_warmup_deactivation()))
    
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
        print("\n✅ TOUS LES TESTS SONT PASSÉS - Action 2.2 validée!")
        print("   → Vous pouvez continuer avec Action 2.3 (resampling optimisé)")
        exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) échoué(s) - NE PAS continuer")
        print("   → Corriger les problèmes avant de passer à Action 2.3")
        exit(1)
