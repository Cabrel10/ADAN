#!/usr/bin/env python3
"""
DIAGNOSTIC RÉEL des indicateurs sur données Binance LIVE
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime

print("🔍 DIAGNOSTIC RÉEL DES INDICATEURS")
print("=" * 60)

# 1. Téléchargement direct avec CCXT (sans wrapper)
exchange = ccxt.binance({'enableRateLimit': True})
symbol = 'BTC/USDT'

print("1️⃣  Téléchargement 200 bougies 5m...")
ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=200)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

print(f"   {len(df)} bougies: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"   Prix: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
print(f"   Variation: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")

# 2. Inspection des données
print("\n2️⃣  Inspection des données:")
print(f"   Shape: {df.shape}")
print(f"   Colonnes: {list(df.columns)}")
print(f"   NaN: {df.isna().sum().sum()}")
print(f"   Inf: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
print(f"   Close stats: mean=${df['close'].mean():.2f}, std=${df['close'].std():.2f}")
print(f"   Close min/max: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# 3. Test pandas_ta DIRECT
print("\n3️⃣  Test pandas_ta DIRECT:")
try:
    import pandas_ta as ta
    
    # RSI
    print("   Calcul RSI(14)...")
    rsi = ta.rsi(df['close'], length=14)
    print(f"   ✅ RSI(14): type={type(rsi)}, shape={rsi.shape}")
    print(f"      min={rsi.min():.2f}, max={rsi.max():.2f}, last={rsi.iloc[-1]:.2f}")
    print(f"      NaN count: {rsi.isna().sum()}")
    
    # ADX
    print("   Calcul ADX(14)...")
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
    print(f"   ✅ ADX result type: {type(adx_result)}")
    print(f"      Colonnes: {list(adx_result.columns) if hasattr(adx_result, 'columns') else 'N/A'}")
    
    if isinstance(adx_result, pd.DataFrame) and 'ADX_14' in adx_result.columns:
        adx_values = adx_result['ADX_14']
        print(f"      min={adx_values.min():.2f}, max={adx_values.max():.2f}, last={adx_values.iloc[-1]:.2f}")
        print(f"      NaN count: {adx_values.isna().sum()}")
    else:
        print(f"   ⚠️  ADX result structure inattendue: {adx_result}")
    
    # MACD
    print("   Calcul MACD...")
    macd = ta.macd(df['close'])
    print(f"   ✅ MACD: last={macd['MACD_12_26_9'].iloc[-1]:.4f}")
    
    # ATR
    print("   Calcul ATR(14)...")
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    print(f"   ✅ ATR(14): last=${atr.iloc[-1]:.2f}")
    
except Exception as e:
    print(f"   ❌ ERREUR pandas_ta: {e}")
    import traceback
    traceback.print_exc()

# 4. Test MANUEL (fallback)
print("\n4️⃣  Test MANUEL des indicateurs:")

# RSI manuel
print("   Calcul RSI manuel...")
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi_manual = 100 - (100 / (1 + rs))
print(f"   ✅ RSI manuel: {rsi_manual.iloc[-1]:.2f}")

# ATR manuel
print("   Calcul ATR manuel...")
high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift())
low_close = abs(df['low'] - df['close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr_manual = true_range.rolling(14).mean()
print(f"   ✅ ATR manuel: ${atr_manual.iloc[-1]:.2f}")

# 5. Test avec IndicatorCalculator du projet
print("\n5️⃣  Test avec IndicatorCalculator du projet:")
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from adan_trading_bot.indicators.calculator import IndicatorCalculator
    
    calculator = IndicatorCalculator()
    print(f"   ✅ IndicatorCalculator chargé")
    
    # Appeler calculate_all
    indicators = calculator.calculate_all(df)
    print(f"   ✅ calculate_all() retourné: {type(indicators)}")
    print(f"      Clés: {list(indicators.keys())}")
    
    for key, value in indicators.items():
        print(f"      {key}: {value:.2f}" if isinstance(value, (int, float)) else f"      {key}: {value}")
    
except Exception as e:
    print(f"   ❌ ERREUR IndicatorCalculator: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("📊 RÉSULTAT ATTENDU: RSI ~40-60, ADX ~20-50")
print("❌ SI TOUT EST 0.00 → BUG DANS VOTRE IndicatorCalculator")
