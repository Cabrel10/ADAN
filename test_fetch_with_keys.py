#!/usr/bin/env python3
"""
Test du fetch avec les vraies clés API
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from adan_trading_bot.exchange_api.connector import get_exchange_client

print("🚀 TEST FETCH AVEC VRAIES CLÉS API")
print("=" * 60)

# Vérifier les clés
api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
api_secret = os.environ.get('BINANCE_TESTNET_SECRET_KEY')

if not api_key or not api_secret:
    print("❌ Clés API manquantes")
    sys.exit(1)

print(f"✅ Clés API trouvées")
print(f"   API Key: {api_key[:20]}...")
print(f"   Secret: {api_secret[:20]}...")

# Connexion
print("\n1️⃣  Connexion à Binance Testnet...")
config = {
    'paper_trading': {
        'exchange_id': 'binance',
        'use_testnet': True,
        'api_key': api_key,
        'api_secret': api_secret
    }
}

try:
    exchange = get_exchange_client(config)
    print("   ✅ Connecté")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test fetch
print("\n2️⃣  Fetch 1000 bougies 5m...")
start = time.time()

try:
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=1000)
    elapsed = time.time() - start
    
    print(f"   ✅ {len(ohlcv)} bougies en {elapsed:.1f}s")
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"      Période: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"      Prix: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
    
except Exception as e:
    print(f"   ❌ Erreur fetch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ FETCH AVEC VRAIES CLÉS RÉUSSI")
