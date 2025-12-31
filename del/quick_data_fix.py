#!/usr/bin/env python3
"""
Solution RAPIDE pour résoudre immédiatement le problème de données insuffisantes
Télécharge le minimum nécessaire et relance le système
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from pathlib import Path
import json

async def quick_data_download():
    """Téléchargement rapide des données essentielles"""
    print("🚀 TÉLÉCHARGEMENT RAPIDE DES DONNÉES ESSENTIELLES")
    print("="*50)
    
    # Configuration Binance Testnet
    api_key = 'OBpX76eDVonGa51ycDN6NKUtk1tE3FXRsc3wTrFKq5SfFoWTL2U9ZS005nTvQ3oW'
    api_secret = 'wEqgNGKE2sf6PrchcNYFAMoNkof7p7Jk33YzdOzLjvstM4eO3PD3tzWbAXoe2LoZ'
    
    # Initialiser l'exchange (mode sandbox)
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'sandbox': True,
        'options': {'defaultType': 'spot'}
    })
    
    try:
        await exchange.load_markets()
        print("✅ Connecté à Binance Testnet")
        
        # Créer le répertoire de données
        data_dir = Path("historical_data")
        data_dir.mkdir(exist_ok=True)
        
        # Timeframes critiques avec nombre minimum de périodes
        timeframes = {
            '5m': 100,   # 8.3 heures
            '1h': 50,    # 2 jours
            '4h': 30     # 5 jours
        }
        
        all_data = {}
        
        for tf, limit in timeframes.items():
            print(f"\n📊 Téléchargement {tf} ({limit} périodes)...")
            
            try:
                # Télécharger les données
                candles = await exchange.fetch_ohlcv('BTC/USDT', tf, limit=limit)
                
                if not candles:
                    print(f"❌ Aucune donnée pour {tf}")
                    continue
                
                # Convertir en DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculer les indicateurs essentiels
                print(f"   📈 Calcul des indicateurs...")
                
                # RSI (14)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # ATR (14)
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = tr.rolling(window=14).mean()
                df['atr_percent'] = (df['atr'] / df['close']) * 100
                
                # ADX simplifié (14)
                plus_dm = df['high'].diff()
                minus_dm = df['low'].diff().abs()
                plus_dm[plus_dm <= 0] = 0
                minus_dm[minus_dm <= 0] = 0
                
                plus_di = 100 * (plus_dm.rolling(window=14).mean() / df['atr'])
                minus_di = 100 * (minus_dm.rolling(window=14).mean() / df['atr'])
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                df['adx'] = dx.rolling(window=14).mean()
                
                # Volatilité (facteur d'annualisation selon timeframe)
                df['returns'] = df['close'].pct_change()
                
                # Facteur d'annualisation correct selon le timeframe
                if tf == '5m':
                    periods_per_year = 365 * 24 * 12  # 5min periods per year
                elif tf == '1h':
                    periods_per_year = 365 * 24  # 1h periods per year
                else:  # 4h
                    periods_per_year = 365 * 6  # 4h periods per year
                
                df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(periods_per_year)
                
                # Sauvegarder
                filename = data_dir / f"BTC_USDT_{tf}_data.csv"
                df.to_csv(filename)
                all_data[tf] = df
                
                print(f"   ✅ {len(df)} périodes sauvegardées")
                print(f"   📊 RSI: {df['rsi'].iloc[-1]:.1f}, ADX: {df['adx'].iloc[-1]:.1f}")
                print(f"   📈 Volatilité: {df['volatility'].iloc[-1]*100:.2f}%")
                
            except Exception as e:
                print(f"   ❌ Erreur {tf}: {e}")
                continue
        
        # Créer un fichier de statut
        status = {
            'created_at': datetime.utcnow().isoformat(),
            'timeframes_loaded': list(all_data.keys()),
            'total_periods': sum(len(df) for df in all_data.values()),
            'status': 'READY' if len(all_data) >= 3 else 'PARTIAL'
        }
        
        with open(data_dir / "quick_load_status.json", 'w') as f:
            json.dump(status, f, indent=2)
        
        print(f"\n✅ Téléchargement terminé: {len(all_data)}/3 timeframes")
        return len(all_data) >= 3
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return False
    finally:
        await exchange.close()

def create_minimal_patch():
    """Crée un patch minimal pour le monitor"""
    patch_code = '''
# PATCH RAPIDE - Ajout au début de fetch_market_data()
async def fetch_market_data(self):
    """Version patchée avec vérification des données préchargées"""
    
    # Vérifier si on a des données préchargées
    data_dir = Path("historical_data")
    if data_dir.exists():
        preloaded_data = {}
        timeframes = ['5m', '1h', '4h']
        
        for tf in timeframes:
            file_path = data_dir / f"BTC_USDT_{tf}_data.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                    if len(df) >= 28:  # Assez de données
                        # Prendre seulement la fenêtre nécessaire
                        window = {'5m': 20, '1h': 10, '4h': 5}[tf]
                        preloaded_data[tf] = {
                            'df': df.tail(window),
                            'indicators': self.indicator_calculator.calculate_indicators(df.tail(50), tf)
                        }
                        self.logger.info(f"✅ Utilisation données préchargées {tf}: {len(df)} périodes")
                except Exception as e:
                    self.logger.error(f"❌ Erreur lecture {tf}: {e}")
        
        if len(preloaded_data) >= 3:
            return preloaded_data
    
    # Sinon, utiliser la méthode originale (mais elle va échouer)
    self.logger.warning("⚠️  Pas de données préchargées, utilisation méthode originale")
    '''
    
    print("\n📝 Patch minimal créé")
    print("💡 Pour l'appliquer manuellement, remplacez fetch_market_data() dans paper_trading_monitor.py")
    
    return patch_code

async def main():
    """Fonction principale de réparation rapide"""
    print("🔧 RÉPARATION RAPIDE DU SYSTÈME ADAN")
    print("="*40)
    
    # 1. Télécharger les données essentielles
    print("\n1. Téléchargement des données...")
    success = await quick_data_download()
    
    if not success:
        print("❌ Échec du téléchargement")
        return False
    
    # 2. Créer le patch minimal
    print("\n2. Création du patch...")
    patch_code = create_minimal_patch()
    
    # 3. Instructions pour l'utilisateur
    print("\n" + "="*40)
    print("✅ DONNÉES TÉLÉCHARGÉES AVEC SUCCÈS")
    print("="*40)
    
    print("\n📋 PROCHAINES ÉTAPES:")
    print("1. Arrêter le monitor actuel:")
    print("   pkill -f paper_trading_monitor.py")
    
    print("\n2. Appliquer le patch automatique:")
    print("   python scripts/fix_monitor_data_loading.py")
    
    print("\n3. Redémarrer le système:")
    print("   python scripts/paper_trading_monitor.py &")
    print("   python scripts/adan_btc_dashboard.py &")
    
    print("\n4. Vérifier les logs:")
    print("   tail -f paper_trading.log | grep -E '(✅|❌|RSI|ADX)'")
    
    print("\n🎯 RÉSULTAT ATTENDU:")
    print("   - RSI, ADX calculés correctement")
    print("   - Volatilité > 0%")
    print("   - Workers dynamiques")
    print("   - Trades réguliers")
    
    return True

if __name__ == "__main__":
    asyncio.run(main())