#!/usr/bin/env python3
"""
Correction d'urgence des indicateurs cassés - Diagnostic et solution
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

print("🔧 DIAGNOSTIC DES INDICATEURS")
print("=" * 60)

# Créer des données de test réalistes (100 bougies)
dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
np.random.seed(42)

# Données réalistes BTC
base_price = 89000
prices = base_price + np.cumsum(np.random.normal(0, 50, 100))

df = pd.DataFrame({
    'timestamp': dates,
    'open': prices + np.random.uniform(-100, 100, 100),
    'high': prices + np.random.uniform(100, 300, 100),
    'low': prices + np.random.uniform(-300, -100, 100),
    'close': prices,
    'volume': np.random.uniform(100, 1000, 100)
})

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

print(f"\n📊 Données de test: {len(df)} bougies")
print(f"   Prix: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
print(f"   Plage: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

# Test 1: RSI
print("\n1️⃣  TEST RSI")
try:
    rsi = ta.rsi(df['close'], length=14)
    rsi_value = rsi.iloc[-1]
    
    if pd.isna(rsi_value) or rsi_value == 0.0:
        print(f"   ⚠️  RSI retourne {rsi_value} (NaN ou zéro)")
    else:
        print(f"   ✅ RSI fonctionne: {rsi_value:.2f}")
except Exception as e:
    print(f"   ❌ RSI échoue: {e}")
    rsi_value = None

# Test 2: ADX
print("\n2️⃣  TEST ADX")
try:
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_result is not None and 'ADX_14' in adx_result.columns:
        adx_value = adx_result['ADX_14'].iloc[-1]
        if pd.isna(adx_value) or adx_value == 0.0:
            print(f"   ⚠️  ADX retourne {adx_value} (NaN ou zéro)")
        else:
            print(f"   ✅ ADX fonctionne: {adx_value:.2f}")
    else:
        print(f"   ⚠️  ADX retourne None ou structure invalide")
        adx_value = None
except Exception as e:
    print(f"   ❌ ADX échoue: {e}")
    adx_value = None

# Test 3: ATR
print("\n3️⃣  TEST ATR")
try:
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    atr_value = atr.iloc[-1]
    
    if pd.isna(atr_value) or atr_value == 0.0:
        print(f"   ⚠️  ATR retourne {atr_value} (NaN ou zéro)")
    else:
        print(f"   ✅ ATR fonctionne: {atr_value:.2f}")
except Exception as e:
    print(f"   ❌ ATR échoue: {e}")
    atr_value = None

# Solutions manuelles si pandas_ta échoue
print("\n" + "=" * 60)
print("🔧 SOLUTIONS MANUELLES (FALLBACK)")

# RSI manuel
print("\n📊 RSI MANUEL:")
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi_manual = 100 - (100 / (1 + rs))
rsi_manual_value = rsi_manual.iloc[-1]

if not pd.isna(rsi_manual_value):
    print(f"   ✅ RSI manuel: {rsi_manual_value:.2f}")
else:
    print(f"   ⚠️  RSI manuel retourne NaN")

# ATR manuel
print("\n📊 ATR MANUEL:")
tr1 = df['high'] - df['low']
tr2 = abs(df['high'] - df['close'].shift())
tr3 = abs(df['low'] - df['close'].shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr_manual = tr.rolling(window=14).mean()
atr_manual_value = atr_manual.iloc[-1]

if not pd.isna(atr_manual_value):
    print(f"   ✅ ATR manuel: {atr_manual_value:.2f}")
else:
    print(f"   ⚠️  ATR manuel retourne NaN")

print("\n" + "=" * 60)
print("✅ DIAGNOSTIC TERMINÉ")
print("\nRECOMMANDATIONS:")
if rsi_value is None or pd.isna(rsi_value) or rsi_value == 0.0:
    print("   → Utiliser RSI MANUEL dans IndicatorCalculator")
if atr_value is None or pd.isna(atr_value) or atr_value == 0.0:
    print("   → Utiliser ATR MANUEL dans IndicatorCalculator")
print("   → Vérifier que les données ont au moins 14 périodes")
print("   → Vérifier que les colonnes 'high', 'low', 'close' existent")
