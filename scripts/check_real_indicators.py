#!/usr/bin/env python3
"""
Check real market indicators from Binance and compare with dashboard
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs) if rs != 0 else 50
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down if down != 0 else 0
        rsi[i] = 100. - 100./(1.+rs) if rs != 0 else 50
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                              np.abs(low - np.roll(close, 1))))
    tr = tr[1:]
    atr = pd.Series(tr).rolling(period).mean().values
    return atr[-1] if len(atr) > 0 else 0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator"""
    tr = np.maximum(high - low,
                   np.maximum(np.abs(high - np.roll(close, 1)),
                             np.abs(low - np.roll(close, 1))))
    tr = tr[1:]
    
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    def smooth(series, period):
        smoothed = np.zeros_like(series)
        smoothed[:period] = series[:period].mean()
        for i in range(period, len(series)):
            smoothed[i] = (smoothed[i-1] * (period-1) + series[i]) / period
        return smoothed
    
    atr = smooth(tr, period)
    plus_di = 100 * smooth(plus_dm, period) / (atr + 1e-10)
    minus_di = 100 * smooth(minus_dm, period) / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = smooth(dx, period)
    
    return adx[-1] if len(adx) > 0 else 0

def main():
    print("🔍 CHECKING REAL MARKET INDICATORS")
    print("="*60)
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    symbol = 'BTC/USDT'
    timeframe = '5m'
    limit = 100
    
    try:
        print(f"\n📊 Fetching data from Binance: {symbol} {timeframe}")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Calculate indicators
        rsi = calculate_rsi(closes)
        current_rsi = rsi[-1]
        
        atr = calculate_atr(highs, lows, closes)
        atr_pct = (atr / closes[-1]) * 100
        
        adx = calculate_adx(highs, lows, closes)
        
        last_candle = df.iloc[-1]
        
        print(f"\n📈 REAL MARKET DATA:")
        print(f"   Price: ${last_candle['close']:.2f}")
        print(f"   RSI (14): {current_rsi:.2f}")
        print(f"   ADX (14): {adx:.2f}")
        print(f"   ATR (14): ${atr:.2f} ({atr_pct:.2f}%)")
        print(f"   Timestamp: {last_candle['timestamp']}")
        
        print(f"\n📊 DASHBOARD DATA (reported):")
        print(f"   Price: $90,102.58")
        print(f"   RSI: 44")
        print(f"   ADX: 100")
        print(f"   ATR: 0.88%")
        
        print(f"\n🔍 COMPARISON:")
        print(f"   RSI: Dashboard=44 vs Real={current_rsi:.2f} (diff: {44-current_rsi:.2f})")
        print(f"   ADX: Dashboard=100 vs Real={adx:.2f} (diff: {100-adx:.2f})")
        print(f"   ATR%: Dashboard=0.88% vs Real={atr_pct:.2f}% (diff: {0.88-atr_pct:.2f}%)")
        
        print(f"\n⚠️  ANALYSIS:")
        if abs(44 - current_rsi) > 5:
            print(f"   ❌ RSI MISMATCH: Dashboard is off by {abs(44-current_rsi):.2f} points")
        else:
            print(f"   ✅ RSI OK")
        
        if abs(100 - adx) > 20:
            print(f"   ❌ ADX MISMATCH: Dashboard is off by {abs(100-adx):.2f} points")
        else:
            print(f"   ✅ ADX OK")
        
        if abs(0.88 - atr_pct) > 0.5:
            print(f"   ❌ ATR MISMATCH: Dashboard is off by {abs(0.88-atr_pct):.2f}%")
        else:
            print(f"   ✅ ATR OK")
        
        print(f"\n🎯 INTERPRETATION:")
        if current_rsi > 70:
            print(f"   RSI: OVERBOUGHT")
        elif current_rsi < 30:
            print(f"   RSI: OVERSOLD")
        else:
            print(f"   RSI: NEUTRAL")
        
        if adx > 50:
            print(f"   ADX: VERY STRONG TREND")
        elif adx > 25:
            print(f"   ADX: STRONG TREND")
        else:
            print(f"   ADX: WEAK TREND")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
