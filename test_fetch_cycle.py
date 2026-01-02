#!/usr/bin/env python3
"""
Test du cycle fetch_data dans la boucle principale
"""

import sys
import time
import pandas as pd
import ccxt
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from adan_trading_bot.indicators.calculator import IndicatorCalculator

def fetch_data_cycle():
    """Simule le cycle fetch_data du script principal"""
    
    print("🚀 TEST CYCLE FETCH_DATA")
    print("=" * 60)
    
    # Initialisation
    exchange = ccxt.binance({
        'sandbox': False,
        'enableRateLimit': True,
    })
    calculator = IndicatorCalculator()
    
    pairs = ['BTC/USDT']
    timeframes = ['5m', '1h', '4h']
    
    # Cycle 1
    print("\n📊 CYCLE 1 - Fetch initial...")
    start = time.time()
    
    data = {}
    for pair in pairs:
        data[pair] = {}
        
        # Fetch 5m
        print(f"   Téléchargement {pair} 5m...")
        ohlcv = exchange.fetch_ohlcv(pair, timeframe='5m', limit=1500)
        df_5m = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'], unit='ms')
        df_5m.set_index('timestamp', inplace=True)
        
        print(f"   ✅ {len(df_5m)} bougies 5m")
        data[pair]['5m'] = df_5m.reset_index()
        
        # Resampling
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        for tf in ['1h', '4h']:
            df_res = df_5m.resample(tf).agg(agg_rules).dropna()
            print(f"   ✅ {len(df_res)} bougies {tf}")
            data[pair][tf] = df_res.reset_index()
        
        # Calcul indicateurs
        for tf in timeframes:
            df_tf = data[pair][tf]
            try:
                indicators = calculator.calculate_all(df_tf)
                rsi = indicators.get('rsi', 0)
                adx = indicators.get('adx', 0)
                print(f"   📊 {tf}: RSI={rsi:.2f}, ADX={adx:.2f}")
            except Exception as e:
                print(f"   ❌ {tf}: {e}")
    
    elapsed = time.time() - start
    print(f"\n✅ Cycle 1 terminé en {elapsed:.1f}s")
    
    # Cycle 2 (après 5 secondes)
    print(f"\n⏳ Attente 5s avant cycle 2...")
    time.sleep(5)
    
    print("\n📊 CYCLE 2 - Fetch update...")
    start = time.time()
    
    data = {}
    for pair in pairs:
        data[pair] = {}
        
        # Fetch 5m
        print(f"   Téléchargement {pair} 5m...")
        ohlcv = exchange.fetch_ohlcv(pair, timeframe='5m', limit=1500)
        df_5m = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'], unit='ms')
        df_5m.set_index('timestamp', inplace=True)
        
        print(f"   ✅ {len(df_5m)} bougies 5m")
        data[pair]['5m'] = df_5m.reset_index()
        
        # Resampling
        for tf in ['1h', '4h']:
            df_res = df_5m.resample(tf).agg(agg_rules).dropna()
            print(f"   ✅ {len(df_res)} bougies {tf}")
            data[pair][tf] = df_res.reset_index()
        
        # Calcul indicateurs
        for tf in timeframes:
            df_tf = data[pair][tf]
            try:
                indicators = calculator.calculate_all(df_tf)
                rsi = indicators.get('rsi', 0)
                adx = indicators.get('adx', 0)
                print(f"   📊 {tf}: RSI={rsi:.2f}, ADX={adx:.2f}")
            except Exception as e:
                print(f"   ❌ {tf}: {e}")
    
    elapsed = time.time() - start
    print(f"\n✅ Cycle 2 terminé en {elapsed:.1f}s")
    
    print("\n" + "=" * 60)
    print("✅ CYCLE FETCH_DATA FONCTIONNE")
    print("   Le bot peut boucler sans bloquer.")

if __name__ == "__main__":
    fetch_data_cycle()
