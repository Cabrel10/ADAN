import pandas as pd
import numpy as np
# import yfinance as yf # No longer needed, replaced by ccxt
import pandas_ta as ta
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import ccxt
import datetime
import time # Added import for time.sleep

warnings.filterwarnings('ignore')

# Configuration
PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT'] # Changed to CCXT format
TIMEFRAMES = ['5m', '1h', '4h']
# PERIOD = '2y'  # Période de données à télécharger - will be replaced by start_date
START_DATE = datetime.datetime(2020, 1, 1) # Fetch data from Jan 1, 2020 for 4-5 years
DATA_DIR = Path('../data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
INDICATORS_DIR = PROCESSED_DIR / 'indicators'

# Mapping for CCXT intervals
CCXT_INTERVALS = {
    '5m': '5m',
    '1h': '1h',
    '4h': '4h'
}

# Création des répertoires
for dir_path in [RAW_DIR, PROCESSED_DIR, INDICATORS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv_ccxt(exchange, symbol, timeframe, since, limit=None):
    """Fetches OHLCV data using CCXT."""
    ohlcv = []
    print(f"  Fetching {symbol} {timeframe} from {datetime.datetime.fromtimestamp(since / 1000)}...")
    while True:
        try:
            # Fetch data
            chunk = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not chunk:
                print(f"  No more data chunks for {symbol} {timeframe}.")
                break
            ohlcv.extend(chunk)
            print(f"  Fetched {len(chunk)} new candles. Total: {len(ohlcv)}")
            # Update 'since' to fetch the next chunk
            since = chunk[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            if limit and len(ohlcv) >= limit:
                print(f"  Reached limit of {limit} candles for {symbol} {timeframe}.")
                break
        except ccxt.NetworkError as e:
            print(f"  Network error while fetching {symbol} {timeframe}: {e}. Retrying in 5 seconds...")
            time.sleep(5)  # Wait before retrying
        except Exception as e:
            print(f"  Error fetching {symbol} {timeframe}: {e}")
            break
    return ohlcv


def download_data(pair: str, interval: str) -> pd.DataFrame: # Removed 'period' argument
    """Télécharge les données OHLCV depuis une source fiable (CCXT) ou charge depuis CSV si disponible"""
    csv_file_path = RAW_DIR / f"{pair.replace('/', '')}_{interval}.csv"  # Adjusted for CCXT pair format

    if csv_file_path.exists():
        print(f"Chargement des données pour {pair} {interval} depuis {csv_file_path}...")
        try:
            df = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)
            if df.empty:
                print(f"Fichier CSV vide pour {pair} {interval}")
                # Fallback to download if CSV is empty
                pass
            else:
                return df
        except Exception as e:
            print(f"Erreur lors du chargement du CSV pour {pair} {interval}: {e}")
            # Fallback to download if CSV loading fails
            pass

    print(f"Téléchargement des données pour {pair} {interval} via CCXT...")
    try:
        exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Changed from 'future' to 'spot'
            },
        })
        
        # Load markets to ensure symbols are available
        exchange.load_markets()

        # Convert start date to milliseconds timestamp
        since_timestamp = int(START_DATE.timestamp() * 1000)
        
        ohlcv_data = fetch_ohlcv_ccxt(exchange, pair, CCXT_INTERVALS[interval], since_timestamp)

        if not ohlcv_data:
            print(f"Aucune donnée pour {pair} {interval} via CCXT.")
            return None

        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        # Rename columns to match yfinance output for compatibility with pandas_ta
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        if df.empty:
            print(f"DataFrame is empty after processing OHLCV data for {pair} {interval}.")
            return None

        # Save downloaded data to CSV for future use
        df.to_csv(csv_file_path)
        print(f"Données téléchargées et sauvegardées en CSV pour {pair} {interval}")
        return df
    except Exception as e:
        print(f"Erreur lors du téléchargement de {pair} {interval} via CCXT: {e}")
        return None


def calculate_indicators(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Calcule les indicateurs techniques en fonction du timeframe"""
    if df is None or df.empty:
        print(f"  calculate_indicators: Input DataFrame is empty for interval {interval}.")
        return None

    initial_cols = df.columns.tolist()
    print(f"  calculate_indicators: Initial DataFrame shape: {df.shape}, columns: {initial_cols}")
    print(f"  calculate_indicators: Initial NaN count: {df.isnull().sum().sum()}")

    # Copie pour éviter les modifications sur place
    df = df.copy()

    # Store columns before adding indicators to check NaNs only for new columns
    original_columns = df.columns.tolist()

    # Calcul des indicateurs communs
    df['returns'] = df['Close'].pct_change()

    # Indicateurs spécifiques au timeframe
    if interval == '5m':
        # Indicateurs pour le scalping
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df.ta.cci(length=20, append=True)
        df.ta.roc(length=9, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.ema(length=5, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.supertrend(length=14, multiplier=2.0, append=True)
        df.ta.psar(af0=0.02, af=0.2, max_af=0.2, append=True)

    elif interval == '1h':
        # Indicateurs pour le swing trading
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.cci(length=20, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=100, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.ichimoku(tenkan=9, kijun=26, senkou=52, append=True)
        df.ta.psar(af0=0.02, af=0.2, max_af=0.2, append=True)

    elif interval == '4h':
        # Indicateurs pour le position trading
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.cci(length=20, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ichimoku(tenkan=9, kijun=26, senkou=52, append=True)
        df.ta.supertrend(length=14, multiplier=3.0, append=True)
        df.ta.psar(af0=0.02, af=0.2, max_af=0.2, append=True)

    # Indicateurs communs à all timeframes
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2.0, append=True)
    df.ta.obv(append=True)

    if interval in ['1h', '4h']:
        df.ta.vwap(anchor='D', append=True)  # VWAP journalier
    if interval == '4h':
        df.ta.vwap(anchor='W', append=True)  # VWAP hebdomadaire

    print(f"  calculate_indicators: DataFrame shape after indicator calculation: {df.shape}")
    
    # Check NaN count for newly added indicator columns
    new_columns = [col for col in df.columns if col not in original_columns]
    print(f"  calculate_indicators: NaN count per new indicator column:")
    for col in new_columns:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"    {col}: {nan_count} NaNs")

    print(f"  calculate_indicators: Total NaN count after indicator calculation: {df.isnull().sum().sum()}")

    # Fill NaN values
    df.fillna(method='ffill', inplace=True) # Forward-fill NaNs
    df.fillna(0, inplace=True) # Fill any remaining NaNs (e.g., at the very beginning) with 0
    print(f"  calculate_indicators: Total NaN count after filling: {df.isnull().sum().sum()}")

    return df


def process_pair(pair: str):
    """Traite une paire de trading pour tous les timeframes"""
    pair_dir = INDICATORS_DIR / pair.replace('/', '')  # Adjusted for CCXT pair format
    pair_dir.mkdir(exist_ok=True)

    for tf in TIMEFRAMES:
        print(f"Traitement de {pair} - {tf}...")

        # Téléchargement des données
        df = download_data(pair, tf)  # Removed 'PERIOD' argument
        if df is None or df.empty:
            print(f"  Skipping {pair} - {tf} due to empty DataFrame after download.")
            continue

        # Calcul des indicateurs
        df = calculate_indicators(df, tf)
        if df is None or df.empty:
            print(f"  Skipping {pair} - {tf} due to empty DataFrame after indicator calculation.")
            continue

        # No longer dropping NaNs here, as they are filled in calculate_indicators
        # initial_rows_before_dropna = len(df)
        # df = df.dropna()
        # if len(df) == 0:
        #     print(f"  Warning: All rows dropped after NaN removal for {pair} - {tf}. Initial rows before dropna: {initial_rows_before_dropna}")
        #     continue
        # elif len(df) < initial_rows_before_dropna:
        #     print(f"  Dropped {initial_rows_before_dropna - len(df)} rows due to NaN values for {pair} - {tf}. Remaining rows: {len(df)}")


        # Division en ensembles train/val/test (70%/15%/15%)
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]

        print(f"  train_df shape: {train_df.shape}, val_df shape: {val_df.shape}, test_df shape: {test_df.shape}")

        # Sauvegarde
        for split, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if split_df.empty:
                print(f"  Warning: {split} DataFrame is empty for {pair} - {tf}. Not saving.")
                continue
            save_dir = INDICATORS_DIR / split / pair.replace('/', '') / f'{tf}.parquet'  # Adjusted for CCXT pair format
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            split_df.to_parquet(save_dir)

        print(f"  ✓ Données sauvegardées pour {pair} - {tf}")


# Traitement de toutes les paires
if __name__ == "__main__":
    for pair in tqdm(PAIRS, desc="Traitement des paires"):
        process_pair(pair)

    print("Traitement terminé avec succès!")
