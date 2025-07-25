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
import talib
import os
import multiprocessing

# Add src to PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
project_root = SCRIPT_DIR.parent
sys.path.append(str(project_root))

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

def calculate_minutes_since_update(df: pd.DataFrame, timeframe: str = '5m') -> pd.Series:
    """
    Calculate minutes since last update for each row with improved granularity.
    
    Args:
        df: DataFrame with timestamp index
        timeframe: Expected timeframe (e.g., '5m', '1h', '4h')
        
    Returns:
        Series with minutes since last update (0 = fresh data, higher = stale data)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Parse expected interval from timeframe
    expected_interval_minutes = parse_timeframe_to_minutes(timeframe)
    
    # Calculate time differences in minutes
    time_diffs = df.index.to_series().diff().dt.total_seconds() / 60
    time_diffs.iloc[0] = 0  # First row has no previous timestamp
    
    # Calculate freshness: 0 = perfectly fresh, higher = more stale
    # For normal intervals, minutes_since_update should be 0
    # For gaps, it shows how many minutes the data is delayed
    freshness = np.maximum(0, time_diffs - expected_interval_minutes)
    
    # Handle DST transitions (detect unusual gaps/overlaps)
    dst_adjusted = handle_dst_transitions(freshness, time_diffs, expected_interval_minutes)
    
    return dst_adjusted.rename('minutes_since_update')

def parse_timeframe_to_minutes(timeframe: str) -> float:
    """
    Parse timeframe string to minutes.
    
    Args:
        timeframe: Timeframe string (e.g., '5m', '1h', '4h')
        
    Returns:
        Number of minutes in the timeframe
    """
    timeframe = timeframe.lower()
    if timeframe.endswith('m'):
        return float(timeframe[:-1])
    elif timeframe.endswith('h'):
        return float(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return float(timeframe[:-1]) * 24 * 60
    else:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

def handle_dst_transitions(freshness: pd.Series, time_diffs: pd.Series, expected_interval: float) -> pd.Series:
    """
    Handle Daylight Saving Time transitions automatically.
    
    Args:
        freshness: Current freshness values
        time_diffs: Raw time differences
        expected_interval: Expected interval in minutes
        
    Returns:
        DST-adjusted freshness values
    """
    # Detect potential DST transitions (gaps of ~60 minutes or overlaps)
    dst_gaps = (time_diffs > expected_interval + 50) & (time_diffs < expected_interval + 70)
    dst_overlaps = (time_diffs < expected_interval - 50) & (time_diffs > expected_interval - 70)
    
    adjusted_freshness = freshness.copy()
    
    # For DST gaps (spring forward), reduce the freshness penalty
    adjusted_freshness[dst_gaps] = np.maximum(0, freshness[dst_gaps] - 60)
    
    # For DST overlaps (fall back), the data is actually fresher
    adjusted_freshness[dst_overlaps] = 0
    
    return adjusted_freshness

def validate_input_data(df: pd.DataFrame, asset: str, timeframe: str) -> Dict[str, Any]:
    """
    Validate input data quality according to design specifications.
    
    Args:
        df: Input DataFrame to validate
        asset: Asset symbol for logging
        timeframe: Timeframe for logging
        
    Returns:
        Dictionary containing validation results
    """
    validation_result = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    try:
        # Check basic structure
        if df.empty:
            validation_result['valid'] = False
            validation_result['issues'].append(f"{asset} {timeframe}: Empty DataFrame")
            return validation_result
        
        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_result['valid'] = False
            validation_result['issues'].append(f"{asset} {timeframe}: Missing columns {missing_cols}")
        
        # Check data completeness
        total_rows = len(df)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / total_rows) * 100
                validation_result['metrics'][f'{col}_missing_pct'] = missing_pct
                
                if missing_pct > 10:
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"{asset} {timeframe}: High missing data in {col} ({missing_pct:.1f}%)")
                elif missing_pct > 5:
                    validation_result['warnings'].append(f"{asset} {timeframe}: Moderate missing data in {col} ({missing_pct:.1f}%)")
        
        # Check timestamp consistency
        if 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'])
                
                # Check for duplicates
                duplicates = timestamps.duplicated().sum()
                if duplicates > 0:
                    validation_result['warnings'].append(f"{asset} {timeframe}: {duplicates} duplicate timestamps")
                
                # Check for gaps
                time_diffs = timestamps.diff().dropna()
                if len(time_diffs) > 0:
                    median_diff = time_diffs.median()
                    large_gaps = (time_diffs > median_diff * 2).sum()
                    if large_gaps > 0:
                        validation_result['warnings'].append(f"{asset} {timeframe}: {large_gaps} large time gaps detected")
                        
            except Exception as e:
                validation_result['valid'] = False
                validation_result['issues'].append(f"{asset} {timeframe}: Invalid timestamp format - {str(e)}")
        
        # Check price data consistency
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Open, Low, Close
            high_issues = ((df['high'] < df['open']) | 
                          (df['high'] < df['low']) | 
                          (df['high'] < df['close'])).sum()
            
            # Low should be <= Open, High, Close
            low_issues = ((df['low'] > df['open']) | 
                         (df['low'] > df['high']) | 
                         (df['low'] > df['close'])).sum()
            
            if high_issues > 0:
                validation_result['warnings'].append(f"{asset} {timeframe}: {high_issues} inconsistent high prices")
            if low_issues > 0:
                validation_result['warnings'].append(f"{asset} {timeframe}: {low_issues} inconsistent low prices")
        
        # Check for outliers
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                prices = df[col].dropna()
                if len(prices) > 0:
                    Q1 = prices.quantile(0.25)
                    Q3 = prices.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((prices < Q1 - 1.5 * IQR) | (prices > Q3 + 1.5 * IQR)).sum()
                    outlier_pct = (outliers / len(prices)) * 100
                    
                    if outlier_pct > 5:
                        validation_result['warnings'].append(f"{asset} {timeframe}: High outliers in {col} ({outlier_pct:.1f}%)")
        
        validation_result['metrics']['total_rows'] = total_rows
        validation_result['metrics']['total_issues'] = len(validation_result['issues'])
        validation_result['metrics']['total_warnings'] = len(validation_result['warnings'])
        
    except Exception as e:
        validation_result['valid'] = False
        validation_result['issues'].append(f"{asset} {timeframe}: Validation error - {str(e)}")
        logger.error(f"Error validating {asset} {timeframe}: {e}")
    
    return validation_result

def log_validation_results(validation_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Log validation results according to design specifications.
    
    Args:
        validation_results: Dictionary of validation results by asset/timeframe
    """
    total_datasets = len(validation_results)
    valid_datasets = sum(1 for result in validation_results.values() if result['valid'])
    
    logger.info(f"Data Validation Summary: {valid_datasets}/{total_datasets} datasets valid")
    
    for dataset_key, result in validation_results.items():
        if not result['valid']:
            logger.error(f"{dataset_key}: INVALID")
            for issue in result['issues']:
                logger.error(f"  - {issue}")
        elif result['warnings']:
            logger.warning(f"{dataset_key}: Valid with warnings")
            for warning in result['warnings']:
                logger.warning(f"  - {warning}")
        else:
            logger.info(f"{dataset_key}: Valid")

def calculate_technical_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """
    Calculate technical indicators for the given DataFrame with validation.
    
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
    
    # Calculate each indicator with error handling
    for indicator in indicators:
        try:
            if indicator.startswith('EMA_'):
                period = int(indicator.split('_')[1])
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
                
            elif indicator.startswith('RSI_'):
                period = int(indicator.split('_')[1])
                df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
                
            elif indicator.startswith('MACD'):
                fastperiod = 12
                slowperiod = 26
                signalperiod = 9
                if '_' in indicator: # Allow custom periods like MACD_10_20_5
                    parts = indicator.split('_')
                    if len(parts) == 4:
                        fastperiod, slowperiod, signalperiod = int(parts[1]), int(parts[2]), int(parts[3])
                
                macd, macdsignal, macdhist = talib.MACD(df['close'], 
                                                        fastperiod=fastperiod, 
                                                        slowperiod=slowperiod, 
                                                        signalperiod=signalperiod)
                df['macd'] = macd
                df['macd_signal'] = macdsignal
                df['macd_hist'] = macdhist # Renamed from macd_diff for clarity
                
            elif indicator.startswith('ATR_'):
                period = int(indicator.split('_')[1])
                df[f'atr_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
                
            elif indicator.startswith('BBANDS_'):
                period = int(indicator.split('_')[1])
                upper, middle, lower = talib.BBANDS(df['close'], 
                                                    timeperiod=period, 
                                                    nbdevup=2, 
                                                    nbdevdn=2, 
                                                    matype=0) # 0 for SMA
                df[f'bb_upper_{period}'] = upper
                df[f'bb_middle_{period}'] = middle
                df[f'bb_lower_{period}'] = lower
                
            elif indicator.startswith('STOCH_'):
                fastk_period = 5
                slowk_period = 3
                slowd_period = 3
                if '_' in indicator: # Allow custom periods
                    parts = indicator.split('_')
                    if len(parts) == 4:
                        fastk_period, slowk_period, slowd_period = int(parts[1]), int(parts[2]), int(parts[3])
                
                slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                           fastk_period=fastk_period,
                                           slowk_period=slowk_period,
                                           slowk_matype=0, # SMA
                                           slowd_period=slowd_period,
                                           slowd_matype=0) # SMA
                df[f'stoch_k_{fastk_period}'] = slowk
                df[f'stoch_d_{slowk_period}'] = slowd
                
            elif indicator.startswith('WILLR_'):
                period = int(indicator.split('_')[1])
                df[f'willr_{period}'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=period)
                
            elif indicator.startswith('CCI_'):
                period = int(indicator.split('_')[1])
                df[f'cci_{period}'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
                
            elif indicator.startswith('ADX_'):
                period = int(indicator.split('_')[1])
                df[f'adx_{period}'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
                
            elif indicator.startswith('SAR'):
                acceleration = 0.02
                maximum = 0.2
                if '_' in indicator: # Allow custom parameters
                    parts = indicator.split('_')
                    if len(parts) == 3:
                        acceleration, maximum = float(parts[1]), float(parts[2])
                df['sar'] = talib.SAR(df['high'], df['low'], 
                                      acceleration=acceleration, 
                                      maximum=maximum)
                
            elif indicator == 'Ichimoku_Cloud':
                # Tenkan-sen (Conversion Line)
                df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
                
                # Kijun-sen (Base Line)
                df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
                
                # Senkou Span A (Leading Span A)
                df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
                
                # Senkou Span B (Leading Span B)
                df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
                
                # Chikou Span (Lagging Span)
                df['chikou_span'] = df['close'].shift(-26)
                
                # Cloud status (1 = price above cloud, -1 = price below cloud, 0 = price in cloud)
                df['cloud_status'] = np.where(
                    df['close'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1), 1,
                    np.where(df['close'] < df[['senkou_span_a', 'senkou_span_b']].min(axis=1), -1, 0)
                )
                
            elif indicator == 'OBV':
                df['obv'] = talib.OBV(df['close'], df['volume'])
                
            elif indicator == 'VWAP':
                # VWAP calculation requires cumulative sum of (price * volume) and cumulative sum of volume
                # This is a simplified version, typically VWAP is calculated per day or session
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                
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
        raw_data_dir = Path(project_root / local_data.get('directory', 'data/raw'))
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
                # Define paths
                raw_file = raw_data_dir / timeframe / f"{asset_file}.csv"
                tf_processed_dir = processed_dir / timeframe
                output_file = tf_processed_dir / f"{asset_file}.parquet"

                # Check if raw data file exists for this timeframe
                if not raw_file.exists():
                    logger.warning(f"Raw data file not found: {raw_file}")
                    continue
                
                # Check for cached processed file
                if output_file.exists() and os.path.getmtime(output_file) > os.path.getmtime(raw_file):
                    logger.info(f"Skipping {asset} {timeframe}: Processed data is up-to-date in cache.")
                    continue

                # Read raw data
                logger.info(f"Reading data for {asset} {timeframe} from {raw_file}")
                df = pd.read_csv(raw_file)
                
                # Convert timestamp to datetime and set as index, ensuring UTC timezone
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df = df.set_index('timestamp')
                
                # Add minutes since update
                df['minutes_since_update'] = calculate_minutes_since_update(df, timeframe)
                
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
    parser.add_argument('--exec_profile', type=str, default='cpu',
                        help='Execution profile (cpu or gpu)')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(Path(args.config))
        
        # Get assets from config
        data_sources = config['data_sources']
        assets = []
        
        # Extract assets from all data sources
        for source in data_sources:
            if 'assets' in source:
                assets.extend(source['assets'])
        
        # Remove duplicates and sort for consistency
        assets = sorted(list(set(assets)))
        
        if not assets:
            raise ValueError("No assets found in configuration")
        
        # Process each asset in parallel
        num_processes = os.cpu_count() or 1  # Use all available CPU cores
        logger.info(f"Starting parallel processing with {num_processes} processes.")
        
        # Create a list of arguments for process_asset
        args_list = [(asset_name, config) for asset_name in assets]

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(process_asset, args_list)
            
        logger.info("Data conversion completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in data conversion pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
