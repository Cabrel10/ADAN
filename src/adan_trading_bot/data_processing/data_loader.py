#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading, processing, and chunking utilities for the ADAN trading bot.

This module provides a robust data loader capable of handling multiple timeframes,
merging them, and serving them in chunks for efficient memory usage during training
and backtesting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging
import warnings

# Configure logging
logger = logging.getLogger(__name__)

class ComprehensiveDataLoader:
    """
    Handles loading, merging, and chunking of multi-timeframe market data based
    on a configuration dictionary.

    - Loads data from CSV files for specified timeframes.
    - Merges timeframes into a single, forward-filled DataFrame.
    - Provides data in manageable chunks to the environment.
    - Calculates optimal trade performance for reward shaping.
    """
    def __init__(self, data_config: Dict[str, Any], base_data_dir: str = 'data/raw'):
        """
        Initializes the data loader with a configuration dictionary.

        Args:
            data_config: A dictionary, typically loaded from data_config.yaml.
            base_data_dir: The base directory where raw data CSVs are stored.
        """
        self.config = data_config
        self.base_data_dir = Path(base_data_dir)
        self.timeframes = self.config['feature_engineering']['timeframes']
        self.symbol = self.config['ccxt_download']['symbol'].replace('/', '')
        self.chunk_size = self.config.get('chunk_size', 2000)  # Default chunk size
        self.price_column = self.config.get('price_column', 'close')
        
        self.merged_data: Optional[pd.DataFrame] = None
        self.current_chunk_start_index: int = 0
        self.total_rows: int = 0
        self.current_chunk_id: int = 0
        self.chunk_pnl: Dict[int, Dict[str, float]] = {}  # Track PnL per chunk
        self.current_chunk: Optional[pd.DataFrame] = None
        self.current_chunk_index: int = 0  # Track current position within chunk

        logger.info(f"ComprehensiveDataLoader initialized for symbol '{self.symbol}' "
                    f"with timeframes: {self.timeframes}")
        
    def calculate_optimal_pnl_for_chunk(self, chunk_data: pd.DataFrame) -> float:
        """
        Calculate the optimal possible PnL for a given chunk of data.
        
        This simulates a perfect trader that buys at the absolute low and sells at the absolute high.
        
        Args:
            chunk_data: DataFrame containing the chunk of market data
            
        Returns:
            float: The optimal PnL as a percentage of the initial capital
        """
        if self.price_column not in chunk_data.columns:
            logger.warning(f"Price column '{self.price_column}' not found in chunk data. Using 'close' as fallback.")
            price_series = chunk_data.get('close', chunk_data.iloc[:, 0])
        else:
            price_series = chunk_data[self.price_column]
        
        if len(price_series) < 2:
            return 0.0
            
        # Find the minimum and maximum prices in the chunk
        min_price = price_series.min()
        max_price = price_series.max()
        
        # Calculate optimal PnL (buy at absolute low, sell at absolute high)
        optimal_pnl_pct = ((max_price - min_price) / min_price) * 100
        
        # Store the chunk's optimal PnL
        self.chunk_pnl[self.current_chunk_id] = {
            'optimal_pnl': optimal_pnl_pct,
            'min_price': min_price,
            'max_price': max_price,
            'start_price': price_series.iloc[0],
            'end_price': price_series.iloc[-1]
        }
        
        logger.debug(f"Chunk {self.current_chunk_id} - Optimal PnL: {optimal_pnl_pct:.2f}% "
                    f"(Price range: {min_price:.2f} - {max_price:.2f})")
        
        return optimal_pnl_pct

    def load_and_merge_data(self) -> None:
        """
        Loads data for all specified timeframes from CSV files and merges them
        into a single master DataFrame.
        """
        logger.info("Starting data loading and merging process...")
        all_dfs: Dict[str, pd.DataFrame] = {}

        for tf in self.timeframes:
            file_path = self.base_data_dir / f"{self.symbol}_{tf}.csv"
            if not file_path.exists():
                logger.error(f"Data file not found: {file_path}")
                raise FileNotFoundError(f"Required data file not found: {file_path}")
            
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Add timeframe suffix to columns to avoid name clashes
                df = df.add_suffix(f'_{tf}')
                all_dfs[tf] = df
                logger.info(f"Successfully loaded {tf} data from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load or process {file_path}: {e}")
                raise

        if not all_dfs:
            logger.error("No data could be loaded. Aborting.")
            return

        # Use the highest frequency timeframe (first in the list) as the base
        base_tf = self.timeframes[0]
        merged = all_dfs[base_tf].copy()

        # Merge other timeframes
        for tf in self.timeframes[1:]:
            merged = pd.merge_asof(merged, all_dfs[tf], left_index=True, right_index=True, 
                                  direction='forward', tolerance=pd.Timedelta('10 days'))

        # Forward-fill any remaining NaNs after the merge
        merged.fillna(method='ffill', inplace=True)
        merged.dropna(inplace=True) # Drop any rows that couldn't be filled (at the start)

        self.merged_data = merged
        self.total_rows = len(self.merged_data)
        self.reset()
        logger.info(f"Data merging complete. Total rows: {self.total_rows}")

    def get_next_chunk(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the next chunk of the merged data.

        Returns:
            A pandas DataFrame containing the next data chunk, or None if the
            dataset has been fully consumed.
        """
        if self.merged_data is None:
            logger.warning("Data not loaded. Call load_and_merge_data() first.")
            return None

        if self.current_chunk_start_index >= self.total_rows:
            logger.info("End of dataset reached.")
            return None

        end_index = self.current_chunk_start_index + self.chunk_size
        chunk = self.merged_data.iloc[self.current_chunk_start_index:end_index]
        self.current_chunk_start_index = end_index
        
        logger.debug(f"Providing chunk: {len(chunk)} rows, progress: {self.get_progress():.2f}%")
        return chunk

    def calculate_optimal_chunk_performance(self, chunk: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates the maximum possible return from a single buy-low, sell-high
        trade within a given data chunk (perfect foresight benchmark).

        Args:
            chunk: The DataFrame chunk to analyze. Must have a 'close' column.

        Returns:
            A dictionary containing the optimal PnL and other metrics.
        """
        close_prices = chunk[f'close_{self.timeframes[0]}'] # Use base timeframe close
        if len(close_prices) < 2:
            return {'optimal_pnl_pct': 0.0, 'reason': 'Not enough data'}

        min_price = close_prices.min()
        min_price_time = close_prices.idxmin()
        
        prices_after_min = close_prices[min_price_time:]
        max_price_after_min = prices_after_min.max()

        if max_price_after_min > min_price:
            optimal_pnl = (max_price_after_min - min_price) / min_price
        else:
            optimal_pnl = 0.0

        return {
            'optimal_pnl_pct': optimal_pnl,
            'buy_price': min_price,
            'sell_price': max_price_after_min
        }
        
    def get_next_observation(self) -> pd.Series:
        """
        Returns the next observation (row) from the current chunk.
        If the current chunk is exhausted, loads the next chunk.
        
        Returns:
            A pandas Series containing the current observation, or None if no more data.
        """
        if self.merged_data is None:
            self.load_and_merge_data()
            
        if self.current_chunk is None or self.current_chunk_index >= len(self.current_chunk):
            self.current_chunk = self.get_next_chunk()
            self.current_chunk_index = 0
            
            if self.current_chunk is None:
                return None
                
        observation = self.current_chunk.iloc[self.current_chunk_index]
        self.current_chunk_index += 1
        
        return observation

    def get_current_chunk(self) -> Optional[pd.DataFrame]:
        """
        Returns the current chunk of data being processed.
        
        Returns:
            A pandas DataFrame containing the current data chunk, or None if no chunk is loaded.
        """
        return self.current_chunk
        
    def reset(self) -> None:
        """Resets the data loader to the beginning of the dataset."""
        self.current_chunk_start_index = 0
        self.current_chunk_id = 0
        self.current_chunk_index = 0
        self.current_chunk = None
        logger.info("DataLoader reset to the beginning of the dataset.")

    def get_progress(self) -> float:
        """Returns the current data consumption progress as a percentage."""
        if self.total_rows == 0:
            return 100.0
        return (self.current_chunk_start_index / self.total_rows) * 100.0

# ==============================================================================
# DEPRECATED CLASSES
# ==============================================================================

class ChunkedDataLoader:
    """
    [DEPRECATED] Use ComprehensiveDataLoader instead.
    Efficiently loads and processes large datasets in chunks to manage memory usage.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ChunkedDataLoader is deprecated and will be removed in a future version. "
            "Please switch to ComprehensiveDataLoader.",
            DeprecationWarning
        )
        # You might want to delegate to the new class or just leave it as a stub
        pass

def load_multi_timeframe_data(*args, **kwargs):
    warnings.warn(
        "load_multi_timeframe_data is deprecated and will be removed in a future version. "
        "Functionality is now part of ComprehensiveDataLoader.",
        DeprecationWarning
    )
    return {}