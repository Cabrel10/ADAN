#!/usr/bin/env python3
"""
Script de préchargement COMPLET des données historiques pour ADAN
Résout le problème "Need at least 28 rows for all indicators"
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import ccxt.async_support as ccxt

class HistoricalDataPreloader:
    def __init__(self, exchange='binance', testnet=True):
        self.exchange = exchange
        self.testnet = testnet
        self.api_key = 'OBpX76eDVonGa51ycDN6NKUtk1tE3FXRsc3wTrFKq5SfFoWTL2U9ZS005nTvQ3oW'
        self.api_secret = 'wEqgNGKE2sf6PrchcNYFAMoNkof7p7Jk33YzdOzLjvstM4eO3PD3tzWbAXoe2LoZ'
        self.client = None
        self.data_dir = Path("historical_data")
        self.data_dir.mkdir(exist_ok=True)

    async def initialize(self):
        """Initialise la connexion à l'exchange"""
        exchange_class = getattr(ccxt, self.exchange)
        config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        
        if self.testnet:
            config['sandbox'] = True
        
        self.client = exchange_class(config)
        await self.client.load_markets()
        print(f"✅ Connecté à {self.exchange} ({'Testnet' if self.testnet else 'Mainnet'})")

    async def fetch_candles(self, symbol='BTC/USDT', timeframe='1h', limit=500):
        """Récupère les candles historiques avec assez de données"""
        try:
            # Calculer le temps nécessaire pour avoir assez de données
            timeframe_to_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
            
            # Pour avoir au moins 100 périodes + 50 pour les indicateurs
            total_periods_needed = limit + 50
            minutes_needed = total_periods_needed * timeframe_to_minutes.get(timeframe, 60)
            
            # Calculer le timestamp de départ
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=minutes_needed)
            since = self.client.parse8601(start_time.isoformat() + 'Z')
            
            print(f"📊 Récupération {timeframe} depuis {start_time}...")
            
            # Récupération par lots de 1000 (limite API)
            all_candles = []
            while len(all_candles) < limit + 50:
                candles = await self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not candles:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + 1  # Décalage d'1ms pour éviter les doublons
                print(f"  → {len(candles)} candles récupérées (total: {len(all_candles)})")
                
                if len(candles) < 1000:
                    break
            
            # Convertir en DataFrame
            df = pd.DataFrame(all_candles[-limit:],  # Garder seulement le nombre demandé
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ {timeframe}: {len(df)} candles récupérées")
            return df
            
        except Exception as e:
            print(f"❌ Erreur {timeframe}: {e}")
            return None

    async def calculate_indicators(self, df, timeframe):
        """Calcule tous les indicateurs nécessaires"""
        try:
            print(f"📈 Calcul des indicateurs pour {timeframe}...")
            
            # Assurer qu'on a assez de données
            if len(df) < 50:
                print(f"⚠️  Données insuffisantes pour {timeframe}: {len(df)} < 50")
                return df
            
            # RSI (14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # ADX (14)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs()
            plus_dm[plus_dm <= 0] = 0
            plus_dm[plus_dm <= minus_dm] = 0
            minus_dm[minus_dm <= 0] = 0
            minus_dm[minus_dm <= plus_dm] = 0
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()
            
            # ATR
            df['atr'] = atr
            df['atr_percent'] = (atr / df['close']) * 100
            
            # Volatilité (rolling std sur 20 périodes)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(365*24*60)  # Annualisée
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume MA
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            print(f"✅ Indicateurs calculés pour {timeframe}")
            return df
            
        except Exception as e:
            print(f"❌ Erreur calcul indicateurs {timeframe}: {e}")
            return df

    async def preload_all_timeframes(self, symbol='BTC/USDT'):
        """Précharge toutes les données pour tous les timeframes"""
        print("="*60)
        print("🚀 PRÉCHARGEMENT DES DONNÉES HISTORIQUES")
        print("="*60)
        
        # Définir les timeframes et le nombre de périodes nécessaires
        timeframes_config = {
            '5m': {
                'limit': 500,   # ~42 heures
                'description': 'Court terme (trading)'
            },
            '1h': {
                'limit': 300,    # ~12.5 jours
                'description': 'Moyen terme (analyse)'
            },
            '4h': {
                'limit': 200,    # ~33 jours
                'description': 'Long terme (tendance)'
            }
        }
        
        all_data = {}
        
        for tf, config in timeframes_config.items():
            print(f"\n📊 Timeframe: {tf} - {config['description']}")
            print(f"   Périodes demandées: {config['limit']}")
            
            # Récupérer les données
            df = await self.fetch_candles(symbol, tf, config['limit'])
            
            if df is not None and len(df) >= 50:
                # Calculer les indicateurs
                df = await self.calculate_indicators(df, tf)
                
                # Sauvegarder
                filename = self.data_dir / f"{symbol.replace('/', '_')}_{tf}_data.csv"
                df.to_csv(filename)
                all_data[tf] = df
                
                # Afficher un résumé
                print(f"   ✅ Données sauvegardées: {filename}")
                print(f"   📊 RSI: {df['rsi'].iloc[-1]:.2f}, ADX: {df['adx'].iloc[-1]:.2f}")
                print(f"   📈 Vol: {df['volatility'].iloc[-1]*100:.2f}%, ATR: {df['atr_percent'].iloc[-1]:.2f}%")
            else:
                print(f"   ❌ Échec du préchargement pour {tf}")
        
        return all_data

    async def create_stats_file(self, all_data):
        """Crée un fichier de statistiques pour le normaliseur"""
        stats = {}
        
        for tf, df in all_data.items():
            if df is not None and not df.empty:
                stats[tf] = {
                    'mean': df.mean().to_dict(),
                    'std': df.std().to_dict(),
                    'min': df.min().to_dict(),
                    'max': df.max().to_dict(),
                    'percentile_25': df.quantile(0.25).to_dict(),
                    'percentile_75': df.quantile(0.75).to_dict()
                }
        
        stats_file = self.data_dir / "normalization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\n✅ Fichier de statistiques créé: {stats_file}")
        return stats

    async def generate_report(self, all_data):
        """Génère un rapport de qualité des données"""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'timeframes': {}
        }
        
        for tf, df in all_data.items():
            if df is not None:
                report['timeframes'][tf] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'date_range': {
                        'start': df.index[0].isoformat(),
                        'end': df.index[-1].isoformat()
                    },
                    'data_quality': {
                        'null_count': df.isnull().sum().to_dict(),
                        'zero_count': (df == 0).sum().to_dict()
                    },
                    'latest_indicators': {
                        'price': float(df['close'].iloc[-1]),
                        'rsi': float(df['rsi'].iloc[-1]),
                        'adx': float(df['adx'].iloc[-1]),
                        'volatility': float(df['volatility'].iloc[-1]),
                        'atr_percent': float(df['atr_percent'].iloc[-1])
                    }
                }
        
        report_file = self.data_dir / "data_quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Rapport généré: {report_file}")

    async def run(self):
        """Exécute le préchargement complet"""
        try:
            await self.initialize()
            
            # Précharger toutes les données
            all_data = await self.preload_all_timeframes()
            
            # Créer les statistiques de normalisation
            await self.create_stats_file(all_data)
            
            # Générer un rapport
            await self.generate_report(all_data)
            
            print("\n" + "="*60)
            print("🎉 PRÉCHARGEMENT TERMINÉ AVEC SUCCÈS")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Erreur lors du préchargement: {e}")
        finally:
            if self.client:
                await self.client.close()

async def main():
    """Fonction principale"""
    print("🔄 Démarrage du préchargement...")
    preloader = HistoricalDataPreloader(testnet=True)
    await preloader.run()
    print("\n✅ Prêt pour le démarrage du trading avec données complètes!")

if __name__ == "__main__":
    asyncio.run(main())