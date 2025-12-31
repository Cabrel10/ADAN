"""
Test de resampling 4h isolé pour diagnostiquer pourquoi on obtient 22 bougies au lieu de 28+

Ce script reproduit exactement la logique de fetch_data() pour comprendre:
1. Combien de bougies 5m sont récupérées de l'API
2. Quelle plage temporelle est couverte
3. Combien de bougies 4h sont générées après resampling
4. Pourquoi le nombre est insuffisant (<28)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt

def test_synthetic_resampling():
    """Test avec des données synthétiques pour valider le calcul théorique"""
    print("=" * 80)
    print("TEST 1: RESAMPLING SYNTHÉTIQUE (1000 bougies 5m)")
    print("=" * 80)
    
    # Générer 1000 bougies 5m synthétiques
    start = datetime.now() - timedelta(minutes=1000*5)
    dates = pd.date_range(start, periods=1000, freq='5min')
    df_5m = pd.DataFrame({
        'open': np.random.rand(1000) * 100 + 80000,
        'high': np.random.rand(1000) * 100 + 80100,
        'low': np.random.rand(1000) * 100 + 79900,
        'close': np.random.rand(1000) * 100 + 80000,
        'volume': np.random.rand(1000) * 1000
    }, index=dates)
    
    print(f"\n📊 Données 5m:")
    print(f"   - Nombre de bougies: {len(df_5m)}")
    print(f"   - Plage: {df_5m.index.min()} → {df_5m.index.max()}")
    
    # Calculer la durée couverte
    duration_hours = (df_5m.index.max() - df_5m.index.min()).total_seconds() / 3600
    print(f"   - Durée: {duration_hours:.1f} heures")
    
    # Théoriquement, combien de bougies 4h ?
    theoretical_4h = duration_hours / 4
    print(f"   - Bougies 4h théoriques: {theoretical_4h:.1f}")
    
    # Resample en 4h (comme dans fetch_data) @line 622
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    df_4h_before_dropna = df_5m.resample('4h').agg(agg_rules)
    print(f"\n🔄 Resampling:")
    print(f"   - Bougies 4h AVANT dropna(): {len(df_4h_before_dropna)}")
    
    df_4h_after_dropna = df_4h_before_dropna.dropna()
    print(f"   - Bougies 4h APRÈS dropna(): {len(df_4h_after_dropna)}")
    print(f"   - Lignes supprimées par dropna(): {len(df_4h_before_dropna) - len(df_4h_after_dropna)}")
    
    # Vérifier s'il y a des NaN
    if len(df_4h_before_dropna) > len(df_4h_after_dropna):
        print(f"\n⚠️  Des NaN ont été détectés!")
        print(f"   NaN par colonne:")
        for col in df_4h_before_dropna.columns:
            nan_count = df_4h_before_dropna[col].isna().sum()
            if nan_count > 0:
                print(f"      - {col}: {nan_count} NaN")
    
    print(f"\n✅ Résultat: {len(df_4h_after_dropna)} bougies 4h (attendu ≥28)")
    
    return df_5m, df_4h_after_dropna


def test_real_api_fetch():
    """Test avec des données réelles de Binance testnet"""
    print("\n" + "=" * 80)
    print("TEST 2: FETCH API BINANCE TESTNET (réel)")
    print("=" * 80)
    
    try:
        # Configuration Binance testnet (comme dans le monitor)
        exchange = ccxt.binance({
            'apiKey': 'OBpX76eDVonGa51ycDN6NKUtk1tE3FXRsc3wTrFKq5SfFoWTL2U9ZS005nTvQ3oW',
            'secret': 'wEqgNGKE2sf6PrchcNYFAMoNkof7p7Jk33YzdOzLjvstM4eO3PD3tzWbAXoe2LoZ',
            'enableRateLimit': True,
            'sandbox': True,
            'options': {'defaultType': 'spot'}
        })
        
        print(f"\n📡 Fetch API:")
        print(f"   - Exchange: Binance Testnet")
        print(f"   - Pair: BTC/USDT")
        print(f"   - Timeframe: 5m")
        print(f"   - Limit demandé: 1500 (augmenté depuis Action 2.3)")
        
        # Fetch 5m data (NOUVEAU limit 1500 après Action 2.3)
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=1500)
        df_5m = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'], unit='ms')
        df_5m.set_index('timestamp', inplace=True)
        
        print(f"\n📊 Données 5m reçues:")
        print(f"   - Nombre de bougies: {len(df_5m)} (sur 1000 demandées)")
        print(f"   - Plage: {df_5m.index.min()} → {df_5m.index.max()}")
        
        # Calculer la durée couverte
        duration_hours = (df_5m.index.max() - df_5m.index.min()).total_seconds() / 3600
        print(f"   - Durée: {duration_hours:.1f} heures ({duration_hours/24:.1f} jours)")
        
        # Théoriquement, combien de bougies 4h ?
        theoretical_4h = duration_hours / 4
        print(f"   - Bougies 4h théoriques: {theoretical_4h:.1f}")
        
        # Vérifier s'il y a des gaps temporels
        time_diffs = df_5m.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=5)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        
        if len(gaps) > 0:
            print(f"\n⚠️  Gaps temporels détectés: {len(gaps)}")
            print(f"   Premiers gaps:")
            for idx, gap in gaps.head(5).items():
                print(f"      - {idx}: {gap}")
        else:
            print(f"\n✅ Aucun gap temporel détecté")
        
        # Resample en 4h (exactement comme ligne 622)
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        df_4h_before_dropna = df_5m.resample('4h').agg(agg_rules)
        print(f"\n🔄 Resampling:")
        print(f"   - Bougies 4h AVANT dropna(): {len(df_4h_before_dropna)}")
        
        df_4h_after_dropna = df_4h_before_dropna.dropna()
        print(f"   - Bougies 4h APRÈS dropna(): {len(df_4h_after_dropna)}")
        print(f"   - Lignes supprimées par dropna(): {len(df_4h_before_dropna) - len(df_4h_after_dropna)}")
        
        # Vérifier s'il y a des NaN
        if len(df_4h_before_dropna) > len(df_4h_after_dropna):
            print(f"\n⚠️  Des NaN ont été détectés!")
            print(f"   NaN par colonne:")
            for col in df_4h_before_dropna.columns:
                nan_count = df_4h_before_dropna[col].isna().sum()
                if nan_count > 0:
                    print(f"      - {col}: {nan_count} NaN")
                    # Afficher les lignes avec NaN
                    nan_rows = df_4h_before_dropna[df_4h_before_dropna[col].isna()]
                    print(f"        Lignes avec NaN:")
                    for row_idx in nan_rows.index[:3]:
                        print(f"          {row_idx}")
        
        print(f"\n{'✅' if len(df_4h_after_dropna) >= 28 else '❌'} Résultat: {len(df_4h_after_dropna)} bougies 4h (attendu ≥28)")
        
        # Si insuffisant, analyser pourquoi
        if len(df_4h_after_dropna) < 28:
            print(f"\n🔍 ANALYSE DU PROBLÈME:")
            
            # Hypothèse A: L'API ne renvoie pas assez de données
            if len(df_5m) < 1000:
                print(f"   ❌ HYPOTHÈSE A CONFIRMÉE: L'API ne renvoie que {len(df_5m)} bougies au lieu de 1000")
                print(f"      → Recommandation: Augmenter limit ou utiliser un fetch différent")
            else:
                print(f"   ✅ Hypothèse A rejetée: L'API renvoie bien {len(df_5m)} bougies")
            
            # Hypothèse B: La durée couverte est trop courte
            min_hours_needed = 28 * 4  # 112 heures minimum
            if duration_hours < min_hours_needed:
                print(f"   ❌ HYPOTHÈSE B CONFIRMÉE: Durée couverte ({duration_hours:.1f}h) < minimum requis ({min_hours_needed}h)")
                print(f"      → Recommandation: Fetch plus de données historiques ou augmenter limit")
            else:
                print(f"   ✅ Hypothèse B rejetée: Durée couverte suffisante ({duration_hours:.1f}h)")
            
            # Hypothèse C: dropna() supprime trop de lignes
            removed = len(df_4h_before_dropna) - len(df_4h_after_dropna)
            if removed > 5:
                print(f"   ❌ HYPOTHÈSE C CONFIRMÉE: dropna() supprime {removed} lignes")
                print(f"      → Recommandation: Utiliser fillna() au lieu de dropna(), ou dropna(subset=['close'])")
            else:
                print(f"   ✅ Hypothèse C rejetée: dropna() ne supprime que {removed} lignes")
        
        return df_5m, df_4h_after_dropna
        
    except Exception as e:
        print(f"\n❌ Erreur lors du fetch API: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def produce_report(df_5m_synth, df_4h_synth, df_5m_real, df_4h_real):
    """Produit le rapport final Action 1.2"""
    print("\n" + "=" * 80)
    print("📊 RAPPORT ACTION 1.2 – AUDIT RESAMPLING 4H")
    print("=" * 80)
    
    print(f"\n1. Données 5m récupérées")
    if df_5m_real is not None:
        duration_hours = (df_5m_real.index.max() - df_5m_real.index.min()).total_seconds() / 3600
        theoretical_4h = duration_hours / 4
        
        print(f"   - Nombre de bougies fetch : {len(df_5m_real)} (sur 1000 demandées)")
        print(f"   - Plage temporelle : {df_5m_real.index.min()} → {df_5m_real.index.max()}")
        print(f"   - Durée couverte : {duration_hours:.1f} heures ({duration_hours/24:.1f} jours)")
        print(f"   - Nombre théorique de bougies 4h : {theoretical_4h:.1f} ({duration_hours:.1f}h / 4)")
    else:
        print(f"   ❌ Impossible de récupérer les données réelles de l'API")
    
    print(f"\n2. Resampling 5m → 4h")
    if df_4h_real is not None:
        # Il faudrait stocker ces valeurs dans la fonction test_real_api_fetch
        print(f"   - Résultat: {len(df_4h_real)} bougies 4h {'✅' if len(df_4h_real) >= 28 else '❌'}")
    
    print(f"\n3. Comparaison synthétique vs réel")
    if df_4h_synth is not None and df_4h_real is not None:
        print(f"   - Synthétique (1000 bougies 5m) : {len(df_4h_synth)} bougies 4h")
        print(f"   - Réel (API Binance)            : {len(df_4h_real)} bougies 4h")
        print(f"   - Écart                         : {len(df_4h_synth) - len(df_4h_real)} bougies")
    
    print(f"\n4. Cause racine identifiée")
    if df_5m_real is not None and df_4h_real is not None:
        if len(df_4h_real) < 28:
            # Analyser quelle hypothèse est confirmée
            duration_hours = (df_5m_real.index.max() - df_5m_real.index.min()).total_seconds() / 3600
            min_hours_needed = 28 * 4
            
            if len(df_5m_real) < 1000:
                print(f"   ✅ L'API Binance testnet ne renvoie que {len(df_5m_real)} bougies au lieu de 1000")
                print(f"      → C'est une limitation de l'API testnet (historique limité)")
            elif duration_hours < min_hours_needed:
                print(f"   ✅ La durée couverte ({duration_hours:.1f}h) est insuffisante pour générer 28 bougies 4h")
                print(f"      → Il faudrait {min_hours_needed}h de données")
            else:
                print(f"   ✅ Le dropna() supprime trop de lignes (NaN dans les données)")
        else:
            print(f"   ✅ Le système fonctionne correctement: {len(df_4h_real)} bougies 4h disponibles")
    
    print(f"\n5. Solution proposée")
    if df_5m_real is not None and df_4h_real is not None and len(df_4h_real) < 28:
        print(f"   📝 RECOMMANDATIONS:")
        print(f"   1. Implémenter le mode 'warmup' (Action 2.2)")
        print(f"      - Compléter les données temps réel avec les données préchargées")
        print(f"      - Permettre au système de démarrer même avec <28 bougies")
        print(f"   2. Optimiser le resampling (Action 2.3)")
        print(f"      - Utiliser fillna() au lieu de dropna() sur certaines colonnes")
        print(f"      - Ou augmenter le limit de fetch (tester 1500, 2000)")
        print(f"   3. Pour production: switcher vers mainnet qui a plus d'historique")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("🔍 Script de test de resampling 4h")
    print("Objectif: Comprendre pourquoi on obtient 22 bougies au lieu de 28+\n")
    
    # Test 1: Synthétique
    df_5m_synth, df_4h_synth = test_synthetic_resampling()
    
    # Test 2: Réel API
    df_5m_real, df_4h_real = test_real_api_fetch()
    
    # Rapport final
    produce_report(df_5m_synth, df_4h_synth, df_5m_real, df_4h_real)
