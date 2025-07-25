import pandas as pd
import pandas_ta as ta
import yaml
import os
from tqdm import tqdm
import logging
from pathlib import Path # Added this import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_indicators(df, indicators_config, common_params):
    """
    Calculates technical indicators for a given DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        indicators_config (dict): Configuration for indicators for a specific timeframe.
        common_params (dict): Common parameters for indicators.
    Returns:
        pd.DataFrame: DataFrame with calculated indicators.
    """
    df_copy = df.copy()

    # Store original 'minutes_since_update' column if it exists
    minutes_since_update_data = None
    if 'minutes_since_update' in df_copy.columns:
        minutes_since_update_data = df_copy['minutes_since_update']
        df_copy = df_copy.drop(columns=['minutes_since_update'])

    # Rename OHLCV columns to lowercase for pandas_ta compatibility
    ohlcv_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    current_ohlcv_cols = [col for col in ohlcv_cols if col in df_copy.columns]
    df_copy.rename(columns={col: col.lower() for col in current_ohlcv_cols}, inplace=True)

    # Ensure OHLCV columns are present in lowercase for pandas_ta
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df_copy.columns:
            # Attempt to find it in original case if not already lowercased
            if col.upper() in df_copy.columns:
                df_copy.rename(columns={col.upper(): col}, inplace=True)
            else:
                logging.warning(f"OHLCV column '{col}' not found in DataFrame. Indicator calculation might fail.")

    # Momentum Indicators
    for indicator in indicators_config.get('momentum', []):
        if 'RSI' in indicator:
            window = int(indicator.split('_')[1])
            df_copy.ta.rsi(close=df_copy['close'], length=window, append=True)
        elif 'STOCH' in indicator:
            k_window = int(indicator.split('_')[1])
            d_window = int(indicator.split('_')[2])
            df_copy.ta.stoch(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], k=k_window, d=d_window, append=True)
        elif 'CCI' in indicator:
            window = int(indicator.split('_')[1])
            df_copy.ta.cci(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], length=window, append=True)
        elif 'ROC' in indicator:
            window = int(indicator.split('_')[1])
            df_copy.ta.roc(close=df_copy['close'], length=window, append=True)
        elif 'MFI' in indicator:
            window = int(indicator.split('_')[1])
            df_copy.ta.mfi(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], volume=df_copy['volume'], length=window, append=True)
        elif 'MACD' in indicator:
            fast = common_params['macd']['fast']
            slow = common_params['macd']['slow']
            signal = common_params['macd']['signal']
            df_copy.ta.macd(close=df_copy['close'], fast=fast, slow=slow, signal=signal, append=True)

    # Trend Indicators
    for indicator in indicators_config.get('trend', []):
        if 'EMA' in indicator:
            window = int(indicator.split('_')[1])
            df_copy.ta.ema(close=df_copy['close'], length=window, append=True)
        elif 'SMA' in indicator:
            window = int(indicator.split('_')[1])
            df_copy.ta.sma(close=df_copy['close'], length=window, append=True)
        elif 'SUPERTREND' in indicator:
            window = int(indicator.split('_')[1])
            multiplier = float(indicator.split('_')[2])
            df_copy.ta.supertrend(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], length=window, multiplier=multiplier, append=True)
        elif 'PSAR' in indicator:
            step = float(indicator.split('_')[1])
            max_step = float(indicator.split('_')[2])
            df_copy.ta.psar(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], af0=step, afmax=max_step, append=True)
        elif 'ICHIMOKU' in indicator:
            tenkan = int(indicator.split('_')[1])
            kijun = int(indicator.split('_')[2])
            senkou = int(indicator.split('_')[3])
            df_copy.ta.ichimoku(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], tenkan=tenkan, kijun=kijun, senkou=senkou, append=True)

    # Volatility Indicators
    for indicator in indicators_config.get('volatility', []):
        if 'ATR' in indicator:
            window = int(indicator.split('_')[1])
            df_copy.ta.atr(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], length=window, append=True)
        elif 'BB' in indicator:
            window = int(indicator.split('_')[1])
            std_dev = float(indicator.split('_')[2])
            df_copy.ta.bbands(close=df_copy['close'], length=window, std=std_dev, append=True)

    # Volume Indicators
    for indicator in indicators_config.get('volume', []):
        if 'VWAP' in indicator: # Simplified condition to always calculate VWAP_D if VWAP is requested
            df_copy.ta.vwap(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], volume=df_copy['volume'], append=True)
        elif 'OBV' in indicator:
            df_copy.ta.obv(close=df_copy['close'], volume=df_copy['volume'], append=True)

    # Rename columns to match config (e.g., 'RSI_14' instead of 'RSI_14')
    # pandas_ta appends indicator names, so we need to ensure they match the config's expected names.
    # This part might need refinement based on exact pandas_ta output names.
    # For now, we assume pandas_ta output names are close enough or we'll adjust later.
    
    # Convert OHLCV columns back to uppercase
    ohlcv_lower_cols = ['open', 'high', 'low', 'close', 'volume']
    current_ohlcv_lower_cols = [col for col in ohlcv_lower_cols if col in df_copy.columns]
    df_copy.rename(columns={col: col.upper() for col in current_ohlcv_lower_cols}, inplace=True)

    # Re-add 'minutes_since_update' if it was present
    if minutes_since_update_data is not None:
        df_copy['minutes_since_update'] = minutes_since_update_data

    return df_copy

def main():
    script_dir = os.path.dirname(__file__)
    # The project root is one level up from ADAN/scripts
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) 
    config_path = os.path.join(project_root, 'ADAN', 'config', 'config.yaml')
    
    config = load_config(config_path)

    # Manually resolve paths from config, as os.path.join doesn't handle ${} variables
    base_dir = Path(project_root) # This is /home/morningstar/Documents/trading
    adan_base_dir = base_dir / 'ADAN' # This is /home/morningstar/Documents/trading/ADAN

    # Resolve raw_data_dir
    # config['paths']['raw_data_dir'] is "${paths.data_dir}/raw"
    # config['paths']['data_dir'] is "${paths.base_dir}/data"
    # config['paths']['base_dir'] is "." (relative to ADAN/)
    
    # So, raw_data_dir should be ADAN/data/raw
    raw_data_dir = adan_base_dir / config['paths']['data_dir'].replace('${paths.base_dir}', '.').strip('/') / 'raw'
    indicators_data_dir = adan_base_dir / config['paths']['processed_data_dir'].replace('${paths.data_dir}', 'data').strip('/') / 'indicators'

    # Ensure paths are absolute and correct
    raw_data_dir = raw_data_dir.resolve()
    indicators_data_dir = indicators_data_dir.resolve()
    
    os.makedirs(indicators_data_dir, exist_ok=True)

    assets = config['data']['assets']
    timeframes = config['data']['timeframes']
    
    feature_engineering_config = config['feature_engineering']
    
    logging.info(f"Starting indicator preprocessing for {len(assets)} assets and {len(timeframes)} timeframes.")
    logging.info(f"Raw data will be read from: {raw_data_dir}")
    logging.info(f"Processed data will be saved to: {indicators_data_dir}")

    for tf in tqdm(timeframes, desc="Processing Timeframes"):
        tf_raw_dir = os.path.join(raw_data_dir, tf)
        
        if not os.path.exists(tf_raw_dir):
            logging.warning(f"Raw data directory not found for timeframe {tf}: {tf_raw_dir}. Skipping.")
            continue

        for asset in assets:
            # Assuming asset names in raw files are like BTCUSDT.parquet
            raw_file_name = f"{asset}USDT.csv" if 'USDT' not in asset else f"{asset}.csv"
            raw_file_path = os.path.join(tf_raw_dir, raw_file_name)
            
            asset_indicators_dir = os.path.join(indicators_data_dir, asset) # Output structure: ASSET/TIMEframe.parquet
            os.makedirs(asset_indicators_dir, exist_ok=True)
            output_file_path = os.path.join(asset_indicators_dir, f"{tf}.parquet")

            logging.info(f"Processing {asset} - {tf}...")

            if not os.path.exists(raw_file_path):
                logging.warning(f"Raw data file not found for {asset} - {tf}: {raw_file_path}. Skipping.")
                continue

            try:
                df_raw = pd.read_csv(raw_file_path)
                
                # Ensure 'timestamp' is datetime and set as index
                if 'timestamp' in df_raw.columns:
                    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                    df_raw = df_raw.set_index('timestamp')
                else:
                    logging.error(f"Timestamp column not found in {raw_file_path}")
                    continue

                # Get indicator config for the current timeframe
                tf_indicators_config = feature_engineering_config['indicators']['timeframes'].get(tf, {})
                common_params = feature_engineering_config['indicators']['common']

                df_processed = calculate_indicators(df_raw, tf_indicators_config, common_params)
                
                # Handle NaN values after indicator calculation
                if feature_engineering_config['preprocessing']['fillna']:
                    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
                    logging.info(f"Filled NaN values for {asset} - {tf}.")

                # Drop rows with any remaining NaNs (e.g., at the very beginning due to window size)
                initial_rows = len(df_processed)
                df_processed = df_processed.dropna()
                if len(df_processed) < initial_rows:
                    logging.warning(f"Dropped {initial_rows - len(df_processed)} rows with NaNs for {asset} - {tf}.")

                df_processed.to_parquet(output_file_path)
                logging.info(f"Successfully processed and saved {asset} - {tf} to {output_file_path}")

            except Exception as e:
                logging.error(f"Error processing {asset} - {tf}: {e}")

    logging.info("Indicator preprocessing completed.")

if __name__ == "__main__":
    main()
