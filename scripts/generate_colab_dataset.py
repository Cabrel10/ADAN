#!/usr/bin/env python3
"""
ADAN Master Clock Dataset Generator
====================================
Generates temporally-aligned multi-timeframe datasets for training.

The 5m timeframe is the MASTER CLOCK. Higher timeframes (1h, 4h) are
reindexed onto the 5m DatetimeIndex using forward-fill (ffill), so at
any row i, ALL timeframes refer to the same wall-clock minute.

This eliminates the "time-travel bug" where the agent could see future
candles from misaligned timeframe indices.

Two modes of operation:
  1. --live : fetch real OHLCV from Binance via ccxt (requires API keys)
  2. default: generate realistic synthetic data locally (no API needed)

Supports any symbol. Use --symbols to specify one or more pairs.

Examples:
    # Generate 5000 synthetic candles for BTC and ETH (no API needed):
    python scripts/generate_colab_dataset.py --candles 5000 --symbols BTCUSDT ETHUSDT

    # Fetch real data from Binance mainnet:
    python scripts/generate_colab_dataset.py --live --no-testnet --symbols BTC/USDT XRP/USDT

    # Quick sandbox test:
    python scripts/generate_colab_dataset.py --candles 1000

Environment variables:
    ADAN_CANDLES         Number of 5m candles (default: 5000)
    BINANCE_API_KEY      API key (only for --live mode)
    BINANCE_SECRET_KEY   API secret (only for --live mode)
    BINANCE_TESTNET      'true' for testnet (default: true)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("generate_dataset")

# ---------------------------------------------------------------------------
# Realistic price profiles for common crypto assets
# ---------------------------------------------------------------------------
ASSET_PROFILES = {
    "BTCUSDT":  {"base": 65000, "daily_vol": 0.025, "trend": 0.0001},
    "ETHUSDT":  {"base": 3400,  "daily_vol": 0.030, "trend": 0.00008},
    "XRPUSDT":  {"base": 0.55,  "daily_vol": 0.035, "trend": 0.00005},
    "SOLUSDT":  {"base": 140,   "daily_vol": 0.040, "trend": 0.00012},
    "BNBUSDT":  {"base": 580,   "daily_vol": 0.022, "trend": 0.00006},
    "DOGEUSDT": {"base": 0.15,  "daily_vol": 0.045, "trend": 0.00003},
    "ADAUSDT":  {"base": 0.45,  "daily_vol": 0.035, "trend": 0.00004},
    "AVAXUSDT": {"base": 35,    "daily_vol": 0.038, "trend": 0.00009},
    "LINKUSDT": {"base": 14,    "daily_vol": 0.033, "trend": 0.00007},
    "DOTUSDT":  {"base": 7.5,   "daily_vol": 0.036, "trend": 0.00005},
}

DEFAULT_PROFILE = {"base": 100, "daily_vol": 0.030, "trend": 0.00005}


# ---------------------------------------------------------------------------
# Indicator calculation (self-contained, no ta-lib dependency)
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on an OHLCV DataFrame.

    Adds: RSI-14, MACD + signal, Bollinger Bands (20,2), ATR-14,
    EMA-20, SMA-50, OBV. All computed in-place.
    """
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]

    # RSI-14
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20.replace(0, np.nan)

    # ATR-14
    tr = pd.concat(
        [h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # Moving averages
    df["ema_20"] = c.ewm(span=20, adjust=False).mean()
    df["sma_50"] = c.rolling(50).mean()

    # OBV
    df["obv"] = (np.sign(c.diff()) * v).fillna(0).cumsum()

    return df


# ---------------------------------------------------------------------------
# Master Clock alignment
# ---------------------------------------------------------------------------
def align_to_master_clock(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> dict:
    """Reindex 1h and 4h onto the 5m Master Clock using ffill.

    At every 5m timestamp t:
      - 1h row = last completed 1h candle as of t
      - 4h row = last completed 4h candle as of t
    """
    master_idx = df_5m.index.copy()
    df_1h_aligned = df_1h.reindex(master_idx, method="ffill")
    df_4h_aligned = df_4h.reindex(master_idx, method="ffill")

    valid = df_1h_aligned["close"].notna() & df_4h_aligned["close"].notna()
    master_idx = master_idx[valid]

    return {
        "5m": df_5m.loc[master_idx].copy(),
        "1h": df_1h_aligned.loc[master_idx].copy(),
        "4h": df_4h_aligned.loc[master_idx].copy(),
    }


# ---------------------------------------------------------------------------
# Synthetic data generation (no API needed)
# ---------------------------------------------------------------------------
def generate_synthetic_ohlcv(
    symbol: str,
    n_candles: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic 5m OHLCV data using geometric Brownian motion.

    The price process includes:
      - Trending component (slight upward drift)
      - Volatility clustering (GARCH-like effect via regime switching)
      - Realistic intrabar high/low spread
      - Volume correlated with volatility

    Args:
        symbol: Asset name (e.g. 'BTCUSDT'). Used to pick a realistic base price.
        n_candles: Number of 5-minute candles.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with DatetimeIndex (UTC) and columns: open, high, low, close, volume
    """
    rng = np.random.default_rng(seed + hash(symbol) % 2**31)
    profile = ASSET_PROFILES.get(symbol.upper(), DEFAULT_PROFILE)
    base_price = profile["base"]
    daily_vol = profile["daily_vol"]
    trend = profile["trend"]

    # Scale daily volatility to 5-minute bars (288 bars per day)
    bar_vol = daily_vol / np.sqrt(288)

    # Generate regime-switching volatility
    n = n_candles
    regimes = np.ones(n)
    regime = 0  # 0=normal, 1=high-vol
    for i in range(n):
        if regime == 0 and rng.random() < 0.005:
            regime = 1
        elif regime == 1 and rng.random() < 0.02:
            regime = 0
        regimes[i] = 1.0 if regime == 0 else 2.5

    # Log returns with GBM
    returns = trend + bar_vol * regimes * rng.standard_normal(n)

    # Build close prices
    log_prices = np.log(base_price) + np.cumsum(returns)
    close = np.exp(log_prices)

    # Intrabar OHLC
    spread = bar_vol * regimes * close
    high = close + rng.uniform(0.1, 1.0, n) * spread
    low = close - rng.uniform(0.1, 1.0, n) * spread
    low = np.maximum(low, close * 0.995)  # floor
    opn = np.roll(close, 1)
    opn[0] = base_price

    # Volume: base volume scaled by volatility regime and random noise
    base_vol = base_price * 10  # proportional to price
    volume = base_vol * regimes * rng.uniform(0.5, 2.0, n)

    # Timestamps: 5-minute intervals ending now
    end = pd.Timestamp.now(tz="UTC").floor("5min")
    idx = pd.date_range(end=end, periods=n, freq="5min", tz="UTC")

    df = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"

    logger.info(
        f"  Synthetic {symbol}: {n} candles, "
        f"price {close[0]:.2f} -> {close[-1]:.2f}, "
        f"vol regime changes included"
    )
    return df


# ---------------------------------------------------------------------------
# Live data fetching (ccxt)
# ---------------------------------------------------------------------------
def fetch_binance_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    limit: int = 1000,
    testnet: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data from Binance (or testnet) via ccxt."""
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        sys.exit(1)

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET_KEY", "")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    if testnet:
        exchange.set_sandbox_mode(True)

    tag = "(testnet)" if testnet else "(mainnet)"
    logger.info(f"Fetching {limit} candles of {symbol} {timeframe} from Binance {tag}...")

    all_candles = []
    since = None
    remaining = limit

    while remaining > 0:
        batch_limit = min(remaining, 1000)
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit)
        if not candles:
            break
        all_candles.extend(candles)
        remaining -= len(candles)
        if len(candles) < batch_limit:
            break
        since = candles[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    logger.info(f"  -> {len(df)} candles from {df.index.min()} to {df.index.max()}")
    return df


def resample_ohlcv(df_base: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample a base-timeframe OHLCV DataFrame to a higher timeframe."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df_base.resample(target_tf).agg(agg).dropna()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def generate_dataset(
    output_dir: str,
    symbol: str = "BTCUSDT",
    n_candles: int = 5000,
    live: bool = False,
    testnet: bool = True,
    split: str = "train",
    seed: int = 42,
):
    """Generate Master-Clock-aligned dataset and save as parquet.

    Args:
        output_dir: Base output directory.
        symbol: Trading pair (e.g. 'BTCUSDT' or 'BTC/USDT').
        n_candles: Number of 5m candles.
        live: If True, fetch from Binance API. If False, generate synthetic data.
        testnet: Use Binance testnet (only relevant if live=True).
        split: Data split name (train/test/val).
        seed: Random seed for synthetic generation.
    """
    # Normalize symbol name
    asset_name = symbol.replace("/", "").upper()
    ccxt_symbol = symbol if "/" in symbol else f"{symbol[:3]}/{symbol[3:]}" if len(symbol) <= 7 else symbol

    logger.info(f"Generating dataset for {asset_name} ({n_candles} candles, mode={'live' if live else 'synthetic'})")

    # 1. Get 5m data
    if live:
        df_5m_raw = fetch_binance_ohlcv(ccxt_symbol, "5m", limit=n_candles, testnet=testnet)
    else:
        df_5m_raw = generate_synthetic_ohlcv(asset_name, n_candles, seed=seed)

    if len(df_5m_raw) < 100:
        logger.error(f"Insufficient 5m data: {len(df_5m_raw)} < 100. Cannot proceed.")
        return None

    # 2. Resample to 1h and 4h
    df_1h_raw = resample_ohlcv(df_5m_raw, "1h")
    df_4h_raw = resample_ohlcv(df_5m_raw, "4h")
    logger.info(f"Resampled: 1h={len(df_1h_raw)}, 4h={len(df_4h_raw)}")

    # 3. Compute indicators BEFORE alignment
    df_5m = compute_indicators(df_5m_raw.copy())
    df_1h = compute_indicators(df_1h_raw.copy())
    df_4h = compute_indicators(df_4h_raw.copy())

    # Drop NaN rows from indicator warm-up
    df_5m = df_5m.dropna(subset=["rsi_14", "atr_14"])
    df_1h = df_1h.dropna(subset=["rsi_14", "atr_14"])
    df_4h = df_4h.dropna(subset=["rsi_14", "atr_14"])

    logger.info(f"After indicators: 5m={len(df_5m)}, 1h={len(df_1h)}, 4h={len(df_4h)}")

    # 4. Master Clock alignment
    aligned = align_to_master_clock(df_5m, df_1h, df_4h)
    for tf, df in aligned.items():
        logger.info(f"  Aligned {tf}: {len(df)} rows")

    # 5. Save to parquet
    out_base = Path(output_dir) / split / asset_name
    out_base.mkdir(parents=True, exist_ok=True)

    for tf, df in aligned.items():
        out_path = out_base / f"{tf}.parquet"
        df.to_parquet(out_path, engine="pyarrow")
        logger.info(f"  Saved {out_path} ({len(df)} rows, {len(df.columns)} cols)")

    # 6. Verify index consistency
    idx_5m = set(aligned["5m"].index)
    idx_1h = set(aligned["1h"].index)
    idx_4h = set(aligned["4h"].index)
    assert idx_5m == idx_1h == idx_4h, (
        f"Index mismatch! 5m={len(idx_5m)}, 1h={len(idx_1h)}, 4h={len(idx_4h)}"
    )
    logger.info(f"MASTER CLOCK VERIFIED: {len(idx_5m)} aligned timestamps for {asset_name}.")
    return aligned


def main():
    parser = argparse.ArgumentParser(
        description="ADAN Dataset Generator -- synthetic or live Binance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic BTC + ETH, 5000 candles (default, no API needed):
  python scripts/generate_colab_dataset.py --symbols BTCUSDT ETHUSDT

  # Live from Binance mainnet, 10000 candles:
  python scripts/generate_colab_dataset.py --live --no-testnet --candles 10000

  # Generate for training AND testing splits:
  python scripts/generate_colab_dataset.py --split train --candles 8000
  python scripts/generate_colab_dataset.py --split test  --candles 2000 --seed 99
""",
    )
    parser.add_argument("--output", default="data/processed/indicators",
                        help="Base output directory")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"],
                        help="One or more symbols (e.g. BTCUSDT ETHUSDT XRPUSDT)")
    parser.add_argument("--split", default="train",
                        help="Data split: train, test, or val")
    parser.add_argument("--candles", type=int,
                        default=int(os.getenv("ADAN_CANDLES", "5000")),
                        help="Number of 5m candles per symbol")
    parser.add_argument("--live", action="store_true",
                        help="Fetch real data from Binance API (requires keys in .env)")
    parser.add_argument("--no-testnet", action="store_true",
                        help="Use Binance mainnet instead of testnet")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic generation")
    args = parser.parse_args()

    testnet = not args.no_testnet

    for symbol in args.symbols:
        generate_dataset(
            output_dir=args.output,
            symbol=symbol,
            n_candles=args.candles,
            live=args.live,
            testnet=testnet,
            split=args.split,
            seed=args.seed,
        )

    logger.info(f"All datasets generated: {args.symbols}")


if __name__ == "__main__":
    main()
