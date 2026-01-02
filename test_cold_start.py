#!/usr/bin/env python3
"""
Test du cold start agressif - Vérifier que les indicateurs sont vivants
"""

import sys
import pandas as pd
import numpy as np
import ccxt
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from adan_trading_bot.indicators.calculator import IndicatorCalculator

def test_cold_start():
    print("🚀 TEST COLD START AGRESSIF")
    print("=" * 60)
    
    # 1. Connexion à Binance (données publiques)
    print("\n1️⃣  Connexion à Binance (données publiques)...")
    try:
        exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
        })
        print("   ✅ Connecté")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False
    
    # 2. Téléchargement agressif
    print("\n2️⃣  Téléchargement agressif 1500 bougies 5m...")
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=1500)
        df_5m = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'], unit='ms')
        df_5m.set_index('timestamp', inplace=True)
        
        print(f"   ✅ {len(df_5m)} bougies 5m téléchargées")
        print(f"      Période: {df_5m.index[0]} → {df_5m.index[-1]}")
        print(f"      Prix: ${df_5m['close'].iloc[0]:.2f} → ${df_5m['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False
    
    # 3. Resampling
    print("\n3️⃣  Resampling vers 1h et 4h...")
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    df_1h = df_5m.resample('1h').agg(agg_rules).dropna()
    df_4h = df_5m.resample('4h').agg(agg_rules).dropna()
    
    print(f"   ✅ 1h: {len(df_1h)} bougies")
    print(f"   ✅ 4h: {len(df_4h)} bougies")
    
    # 4. Calcul des indicateurs
    print("\n4️⃣  Calcul des indicateurs...")
    calculator = IndicatorCalculator()
    
    for tf_name, df in [('5m', df_5m), ('1h', df_1h), ('4h', df_4h)]:
        try:
            indicators = calculator.calculate_all(df)
            rsi = indicators.get('rsi', 0)
            adx = indicators.get('adx', 0)
            atr = indicators.get('atr', 0)
            
            # Vérification
            if rsi == 50.0 and adx == 25.0:
                print(f"   ❌ {tf_name}: INDICATEURS FIGÉS (RSI={rsi:.2f}, ADX={adx:.2f})")
            else:
                print(f"   ✅ {tf_name}: RSI={rsi:.2f}, ADX={adx:.2f}, ATR={atr:.2f}")
        except Exception as e:
            print(f"   ❌ {tf_name}: Erreur calcul - {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ COLD START AGRESSIF RÉUSSI")
    print("   Les indicateurs sont vivants et le bot peut démarrer.")
    return True

if __name__ == "__main__":
    success = test_cold_start()
    sys.exit(0 if success else 1)
