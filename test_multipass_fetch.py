#!/usr/bin/env python3
"""
Test du multi-pass fetch pour vérifier qu'on obtient bien ~2000 bougies
"""

import ccxt
import pandas as pd
import time

print("🚀 TEST MULTI-PASS FETCH")
print("=" * 60)

exchange = ccxt.binance({
    'sandbox': False,
    'enableRateLimit': True,
})

print("\n1️⃣  1ère requête: 1000 bougies récentes...")
start = time.time()
ohlcv1 = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=1000)
elapsed1 = time.time() - start

print(f"   ✅ {len(ohlcv1)} bougies en {elapsed1:.1f}s")
print(f"      Première: {pd.Timestamp(ohlcv1[0][0], unit='ms')}")
print(f"      Dernière: {pd.Timestamp(ohlcv1[-1][0], unit='ms')}")

print("\n2️⃣  2ème requête: 1000 bougies précédentes...")
start = time.time()

# Calculer le timestamp de la première bougie
since = ohlcv1[0][0] - (1000 * 5 * 60 * 1000)  # 1000 bougies en arrière
print(f"   Since: {pd.Timestamp(since, unit='ms')}")

ohlcv2 = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', since=since, limit=1000)
elapsed2 = time.time() - start

print(f"   ✅ {len(ohlcv2)} bougies en {elapsed2:.1f}s")
if len(ohlcv2) > 0:
    print(f"      Première: {pd.Timestamp(ohlcv2[0][0], unit='ms')}")
    print(f"      Dernière: {pd.Timestamp(ohlcv2[-1][0], unit='ms')}")

print("\n3️⃣  Fusion et déduplication...")
ohlcv_all = ohlcv2 + ohlcv1

df = pd.DataFrame(ohlcv_all, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.drop_duplicates(subset='timestamp')
df = df.sort_values('timestamp')

print(f"   ✅ {len(df)} bougies après déduplication")
print(f"      Période: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

print("\n4️⃣  Resampling 4h...")
df.set_index('timestamp', inplace=True)
df_4h = df.resample('4h').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"   ✅ {len(df_4h)} bougies 4h")

print("\n" + "=" * 60)
print(f"✅ MULTI-PASS RÉUSSI")
print(f"   Total: {len(df)} bougies 5m → {len(df_4h)} bougies 4h")
print(f"   Temps total: {elapsed1 + elapsed2:.1f}s")
