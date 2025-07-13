#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to convert raw OHLCV data into processed data with technical indicators
for multiple timeframes based on data_config.yaml configuration.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
import yaml
from datetime import datetime, timedelta

# Add src to PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('convert_real_data.log')
    ]
)
logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent / 'config' / 'data_config.yaml'

# Standard OHLCV column names
OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate configuration from data_config.yaml.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Set default paths if not specified
        data_pipeline = config.get('data_pipeline', {})
        data_pipeline.setdefault('local_data', {})
        data_pipeline['local_data'].setdefault('directory', 'data/raw')
        
        # Ensure required keys exist
        required_sections = ['data_pipeline', 'feature_engineering']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in config: {section}")
                
        return config
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """
    Parse timeframe string into numeric value and unit.
    
    Args:
        timeframe: Timeframe string (e.g., '5m', '1h', '3h')
        
    Returns:
        Tuple of (value, unit)
    """
    value = int(''.join(filter(str.isdigit, timeframe)))
    unit = ''.join(filter(str.isalpha, timeframe)).lower()
    return value, unit

def calculate_minutes_since_update(df: pd.DataFrame) -> pd.Series:
    """
    Calculate minutes since last update for each row.
    
    Args:
        df: DataFrame with timestamp index
        
    Returns:
        Series with minutes since last update
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
        
    # Calculate time differences in minutes
    time_diffs = df.index.to_series().diff().dt.total_seconds() / 60
    time_diffs.iloc[0] = 0  # First row has no previous timestamp
    
    return time_diffs.rename('minutes_since_update')

def calculate_technical_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """
    Calculate technical indicators for the given DataFrame.
    
    Args:
        df: Input DataFrame with OHLCV data
        indicators: List of indicator names to calculate
        
    Returns:
        DataFrame with added indicator columns
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure timestamp is the index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # Ensure OHLCV columns exist
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate each indicator
    for indicator in indicators:
        try:
            if indicator.startswith('EMA_'):
                period = int(indicator.split('_')[1])
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                
            elif indicator.startswith('RSI_'):
                period = int(indicator.split('_')[1])
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                
            elif indicator == 'MACD_diff':
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_diff'] = df['macd'] - df['macd_signal']
                
            elif indicator == 'ADX_14':
                # Calculate +DM, -DM, and True Range
                df['plus_dm'] = df['high'].diff()
                df['minus_dm'] = df['low'].diff().abs()
                
                # Calculate True Range
                df['tr1'] = df['high'] - df['low']
                df['tr2'] = (df['high'] - df['close'].shift()).abs()
                df['tr3'] = (df['low'] - df['close'].shift()).abs()
                df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
                
                # Calculate +DM and -DM
                df['plus_dm'] = np.where(
                    (df['plus_dm'] > df['minus_dm']) & (df['plus_dm'] > 0),
                    df['plus_dm'], 0.0
                )
                df['minus_dm'] = np.where(
                    (df['minus_dm'] > df['plus_dm']) & (df['minus_dm'] > 0),
                    df['minus_dm'], 0.0
                )
                
                # Smooth the DMs and TR
                period = 14
                df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / 
                                     df['tr'].ewm(alpha=1/period, adjust=False).mean())
                df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / 
                                      df['tr'].ewm(alpha=1/period, adjust=False).mean())
                
                # Calculate ADX
                df['dx'] = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
                df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()
                
                # Clean up intermediate columns
                df.drop(['plus_dm', 'minus_dm', 'tr1', 'tr2', 'tr3', 'dx'], axis=1, inplace=True)
                
            elif indicator == 'Supertrend_14_3':
                # Supertrend with ATR 14 and multiplier 3
                atr_period = 14
                multiplier = 3.0
                
                high = df['high']
                low = df['low']
                close = df['close']
                
                # Calculate ATR
                tr1 = high - low
                tr2 = (high - close.shift()).abs()
                tr3 = (low - close.shift()).abs()
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
                
                # Calculate basic upper and lower bands
                hl2 = (high + low) / 2
                df['supertrend_upper'] = hl2 + (multiplier * atr)
                df['supertrend_lower'] = hl2 - (multiplier * atr)
                
                # Initialize the Supertrend column
                df['supertrend'] = 0.0
                
                # Calculate Supertrend
                for i in range(1, len(df)):
                    if close.iloc[i-1] <= df['supertrend_upper'].iloc[i-1]:
                        df['supertrend_upper'].iloc[i] = min(df['supertrend_upper'].iloc[i], 
                                                           df['supertrend_upper'].iloc[i-1])
                    else:
                        df['supertrend_upper'].iloc[i] = df['supertrend_upper'].iloc[i]
                        
                    if close.iloc[i-1] >= df['supertrend_lower'].iloc[i-1]:
                        df['supertrend_lower'].iloc[i] = max(df['supertrend_lower'].iloc[i], 
                                                           df['supertrend_lower'].iloc[i-1])
                    else:
                        df['supertrend_lower'].iloc[i] = df['supertrend_lower'].iloc[i]
                
                # Determine the trend direction
                df['supertrend'] = np.where(close > df['supertrend_upper'], 1, 
                                          np.where(close < df['supertrend_lower'], -1, np.nan))
                df['supertrend'] = df['supertrend'].ffill()
                
            elif indicator == 'Ichimoku_Cloud':
                # Tenkan-sen (Conversion Line)
                high_9 = df['high'].rolling(window=9).max()
                low_9 = df['low'].rolling(window=9).min()
                df['tenkan_sen'] = (high_9 + low_9) / 2
                
                # Kijun-sen (Base Line)
                high_26 = df['high'].rolling(window=26).max()
                low_26 = df['low'].rolling(window=26).min()
                df['kijun_sen'] = (high_26 + low_26) / 2
                
                # Senkou Span A (Leading Span A)
                df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
                
                # Senkou Span B (Leading Span B)
                high_52 = df['high'].rolling(window=52).max()
                low_52 = df['low'].rolling(window=52).min()
                df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
                
                # Chikou Span (Lagging Span)
                df['chikou_span'] = df['close'].shift(-26)
                
                # Cloud status (1 = price above cloud, -1 = price below cloud, 0 = price in cloud)
                df['cloud_status'] = np.where(
                    df['close'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1), 1,
                    np.where(df['close'] < df[['senkou_span_a', 'senkou_span_b']].min(axis=1), -1, 0)
                )
                
            # Add more indicators as needed...
            
        except Exception as e:
            logger.warning(f"Error calculating {indicator}: {e}")
            continue
    
    return df.reset_index()

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resamples OHLCV data to a target timeframe.
    
    Args:
        df: Input DataFrame with OHLCV data
        timeframe: Target timeframe (e.g., '1h', '4h')
        
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
        
    # Convert timeframe to pandas frequency string
    timeframe_map = {
        '1m': '1T',   # 1 minute
        '5m': '5T',   # 5 minutes
        '15m': '15T', # 15 minutes
        '1h': '1H',   # 1 hour
        '4h': '4H',   # 4 hours
        '1d': '1D'    # 1 day
    }
    
    freq = timeframe_map.get(timeframe, timeframe)
    
    # Define resampling rules
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_asset_volume': 'sum',
        'number_of_trades': 'sum',
        'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum'
    }
    
    # Resample the data
    resampled = df.resample(freq).agg(ohlc_dict).dropna()
    
    # Recalculate typical price and other derived columns if they exist
    if all(col in resampled.columns for col in ['high', 'low', 'close']):
        resampled['typical_price'] = (resampled['high'] + resampled['low'] + resampled['close']) / 3
    
    logger.info(f"Resampled data to {timeframe}: {len(df)} -> {len(resampled)} rows")
    return resampled

def process_asset(asset: str, config: Dict[str, Any]) -> None:
    """
    Process a single asset across all configured timeframes.
    
    Args:
        asset: Asset symbol (e.g., 'BTC/USDT')
        config: Configuration dictionary
    """
    try:
        # Get configuration
        data_pipeline = config['data_pipeline']
        local_data = data_pipeline.get('local_data', {})
        raw_data_dir = Path(local_data.get('directory', 'data/raw'))
        processed_dir = Path('data/processed')
        
        # Create output directory if it doesn't exist
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Get timeframes from config
        timeframes = config['feature_engineering'].get('timeframes', ['5m', '1h'])
        
        # Format asset name for file paths
        asset_file = asset.replace('/', '')
        
        logger.info(f"Processing {asset}...")
        
        # Process each timeframe
        for timeframe in timeframes:
            try:
                # Check if raw data file exists for this timeframe
                raw_file = raw_data_dir / timeframe / f"{asset_file}.csv"
                if not raw_file.exists():
                    logger.warning(f"Raw data file not found: {raw_file}")
                    continue
                
                # Read raw data
                logger.info(f"Reading data for {asset} {timeframe} from {raw_file}")
                df = pd.read_csv(raw_file)
                
                # Convert timestamp to datetime and set as index
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                
                # Add minutes since update
                df['minutes_since_update'] = calculate_minutes_since_update(df)
                
                # Get indicators for this timeframe
                indicators = config['feature_engineering'].get('indicators_by_timeframe', {}).get(timeframe, [])
                
                # Calculate technical indicators
                if indicators:
                    df = calculate_technical_indicators(df, indicators)
                
                # Save processed data
                tf_processed_dir = processed_dir / timeframe
                tf_processed_dir.mkdir(parents=True, exist_ok=True)
                output_file = tf_processed_dir / f"{asset_file}.parquet"
                df.to_parquet(output_file)
                logger.info(f"Saved processed data for {asset} {timeframe} to {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing {asset} {timeframe}: {e}", exc_info=True)
                continue
                
    except Exception as e:
        logger.error(f"Error processing asset {asset}: {e}", exc_info=True)
        raise

def main() -> None:
    """Main function to run the data conversion pipeline."""
    parser = argparse.ArgumentParser(description='Convert raw OHLCV data into processed data with technical indicators.')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH),
                      help='Path to configuration file (default: config/data_config.yaml)')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(Path(args.config))
        
        # Get assets from config
        data_pipeline = config['data_pipeline']
        if data_pipeline['source'] == 'ccxt':
            assets = data_pipeline['ccxt_download']['symbols']
        else:
            # For local data, we'd need to scan the directory
            raise NotImplementedError("Local data source not yet implemented")
        
        # Process each asset
        for asset in assets:
            process_asset(asset, config)
            
        logger.info("Data conversion completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in data conversion pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
