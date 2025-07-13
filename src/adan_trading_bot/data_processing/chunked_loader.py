#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chunked data loading and processing for the ADAN trading bot.

This module provides efficient loading of large datasets in chunks to manage memory usage
during training and backtesting. It's designed to work with the multi-timeframe data
structure produced by the merge_processed_data.py script.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import warnings
import pyarrow.parquet as pq

# Configure logging
logger = logging.getLogger(__name__)

class ChunkedDataLoader:
    """
    Efficiently loads and processes large datasets in chunks to manage memory usage.
    
    This loader is optimized for multi-timeframe data and includes utilities for
    calculating optimal PnL on a per-chunk basis.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path], 
                 chunk_size: int = 10000,
                 assets_list: Optional[List[str]] = None,
                 features_by_timeframe: Optional[Dict[str, List[str]]] = None,
                 split: str = 'train',
                 timeframes: Optional[List[str]] = None):
        """
        Initialize the ChunkedDataLoader.
        
        Args:
            data_dir: Base directory containing the final data (should contain subdirectories for each asset)
            chunk_size: Number of rows to load per chunk
            assets_list: List of asset symbols to load (e.g., ['BTCUSDT', 'ETHUSDT'])
            features_by_timeframe: Dictionary mapping timeframes to their expected features
            split: Which split to load ('train', 'val', or 'test')
            timeframes: List of timeframes to include (e.g., ['5m', '1h', '4h'])
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.assets_list = assets_list or []
        self.features_by_timeframe = features_by_timeframe or {}
        self.split = split
        self.timeframes = timeframes or ['5m', '1h', '4h']
        
        # State tracking
        self.current_chunk_index = 0
        self.total_chunks = 0
        self.asset_files = {}
        self.asset_parquet_files = {}
        self.asset_row_counts = {}
        
        # Validate inputs
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
        # Initialize the loader
        self._initialize_loader()
    
    def _initialize_loader(self) -> None:
        """Initialize the loader by scanning the data directory and setting up file handles."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # If assets_list is not provided, discover all available assets
        if not self.assets_list:
            self._discover_assets()
        
        # Initialize parquet file handles and row counts for each asset
        for asset in self.assets_list:
            asset_filename = self._get_asset_filename(asset)
            asset_files = {}
            
            # Find the asset's data files across all timeframes
            for tf in self.timeframes:
                tf_dir = self.data_dir / tf
                asset_file = tf_dir / f"{asset_filename}.parquet"
                
                if not asset_file.exists():
                    logger.warning(f"Data file not found for {asset} {tf}: {asset_file}")
                    continue
                    
                try:
                    # Open the parquet file
                    parquet_file = pq.ParquetFile(asset_file)
                    asset_files[tf] = {
                        'file': parquet_file,
                        'rows': parquet_file.metadata.num_rows
                    }
                    
                    # Log the available columns for the first asset and timeframe (for debugging)
                    if asset == self.assets_list[0] and tf == self.timeframes[0]:
                        schema = parquet_file.schema_arrow
                        logger.info(f"Available columns for {asset} {tf}: {schema.names[:10]}...")
                        
                except Exception as e:
                    logger.error(f"Error loading {asset_file}: {e}")
                    continue
            
            if not asset_files:
                logger.warning(f"No valid data files found for {asset} in any timeframe")
                continue
                
            self.asset_files[asset] = asset_files
            # Use the minimum row count across all timeframes for this asset
            self.asset_row_counts[asset] = min(f['rows'] for f in asset_files.values())
            
            # For backward compatibility, store the first timeframe's file as the main one
            first_tf = next(iter(asset_files))
            self.asset_parquet_files[asset] = asset_files[first_tf]['file']
        
        if not self.asset_parquet_files:
            raise ValueError(f"No valid asset files found in {self.data_dir} for split '{self.split}'")
        
        # Calculate total chunks based on the asset with the fewest rows
        min_rows = min(self.asset_row_counts.values())
        self.total_chunks = (min_rows + self.chunk_size - 1) // self.chunk_size
        
        logger.info(f"Initialized ChunkedDataLoader with {len(self.asset_parquet_files)}/{len(self.assets_list)} "
                  f"assets for split '{self.split}' and {self.total_chunks} chunks of size {self.chunk_size}")
    
    def _discover_assets(self) -> None:
        """Discover all available assets in the data directory."""
        # Look for parquet files in the timeframe subdirectories
        for tf in self.timeframes:
            tf_dir = self.data_dir / tf
            if not tf_dir.exists():
                logger.warning(f"Timeframe directory not found: {tf_dir}")
                continue
                
            # Find all parquet files in this timeframe directory
            for parquet_file in tf_dir.glob("*.parquet"):
                # Extract asset name from filename (e.g., ARBUSDT.parquet -> ARBUSDT)
                asset = parquet_file.stem
                # Normalize symbol format (e.g., ARBUSDT -> ARB/USDT)
                if asset.endswith("USDT"):
                    symbol = f"{asset[:-4]}/USDT"
                    if symbol not in self.assets_list:
                        self.assets_list.append(symbol)
        
        # Sort assets for consistent ordering
        self.assets_list = sorted(self.assets_list)
            
        if not self.assets_list:
            raise ValueError(f"No valid asset files found in {self.data_dir} with timeframes {self.timeframes}")
            
        logger.info(f"Discovered {len(self.assets_list)} assets: {', '.join(self.assets_list)}")
        
    def _get_asset_filename(self, symbol: str) -> str:
        """Convert symbol to filename format (e.g., ARB/USDT -> ARBUSDT)."""
        return symbol.replace("/", "")
    
    def load_chunk(self, chunk_index: int) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load a specific chunk of data for all assets and timeframes.
        
        Args:
            chunk_index: Index of the chunk to load
            
        Returns:
            Nested dictionary mapping asset symbols to timeframes to DataFrames
            Example: {'ARB/USDT': {'5m': df_5m, '1h': df_1h}, ...}
        """
        if chunk_index >= self.total_chunks:
            raise IndexError(f"Chunk index {chunk_index} out of range (0-{self.total_chunks-1})")
            
        chunk_data = {asset: {} for asset in self.assets_list}
        start_row = chunk_index * self.chunk_size
        end_row = min(start_row + self.chunk_size, min(self.asset_row_counts.values()))
        
        for asset, tf_files in self.asset_files.items():
            for tf, file_info in tf_files.items():
                parquet_file = file_info['file']
                try:
                    # Read the chunk
                    table = parquet_file.read_row_group(chunk_index)
                    df = table.to_pandas()
                    
                    # Set timestamp as index if it exists
                    if 'timestamp' in df.columns:
                        df = df.set_index('timestamp')
                    
                    # Ensure the index is a DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception as e:
                            logger.error(f"Could not convert index to datetime for {asset} {tf}: {e}")
                            continue
                    
                    # Filter features for this timeframe
                    if tf in self.features_by_timeframe:
                        # Keep only the features specified for this timeframe
                        features = [f for f in self.features_by_timeframe[tf] if f in df.columns]
                        if features:
                            df = df[features]
                    
                    # Add to chunk data
                    chunk_data[asset][tf] = df
                    
                except Exception as e:
                    logger.error(f"Error loading chunk {chunk_index} for {asset} {tf}: {e}", exc_info=True)
                    chunk_data[asset][tf] = pd.DataFrame()
                    continue
        
        self.current_chunk_index = chunk_index
        logger.debug(f"Loaded chunk {chunk_index} with {sum(len(tfs) for tfs in chunk_data.values())} asset-timeframe combinations")
        
        return chunk_data
    
    def _filter_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Filter DataFrame columns based on features_by_timeframe configuration.
        
        Args:
            df: Input DataFrame to filter
            timeframe: The timeframe for which to filter features
            
        Returns:
            Filtered DataFrame with only the requested features for the given timeframe
        """
        if not self.features_by_timeframe or timeframe not in self.features_by_timeframe:
            return df
        
        # Get the list of requested features for this timeframe
        requested_features = self.features_by_timeframe[timeframe]
        
        # Find which of the requested features are actually in the DataFrame
        available_features = [f for f in requested_features if f in df.columns]
        
        if not available_features:
            logger.warning(f"None of the requested features for {timeframe} are available in the data. Available columns: {df.columns.tolist()}")
            return df
        
        # Return only the requested features that are available
        return df[available_features]
    
    def calculate_optimal_pnl_for_chunk(self, chunk_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate the optimal possible PnL for each asset in the chunk.
        
        This simulates a perfect trader that buys at the absolute low and sells at the absolute high.
        
        Args:
            chunk_data: Dictionary of DataFrames, one per asset
            
        Returns:
            Dictionary mapping asset symbols to their optimal PnL percentages
        """
        optimal_pnl = {}
        
        for asset, df in chunk_data.items():
            if df is None or df.empty:
                logger.warning(f"No data for {asset} in chunk {self.current_chunk_index}")
                optimal_pnl[asset] = 0.0
                continue
            
            try:
                # Find the price column (prioritize close, then open, then first numeric column)
                price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
                if not price_cols:
                    price_cols = [col for col in df.columns if 'open' in col.lower()]
                if not price_cols:
                    # Fall back to first numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    price_cols = [numeric_cols[0]] if len(numeric_cols) > 0 else []
                
                if not price_cols:
                    logger.warning(f"No price column found for {asset}")
                    optimal_pnl[asset] = 0.0
                    continue
                
                price_series = df[price_cols[0]]
                
                if len(price_series) < 2:
                    optimal_pnl[asset] = 0.0
                    continue
                
                # Find the minimum and maximum prices in the chunk
                min_price = price_series.min()
                max_price = price_series.max()
                start_price = price_series.iloc[0]
                
                # Calculate optimal PnL (buy at absolute low, sell at absolute high)
                if min_price > 0:
                    optimal_pnl[asset] = ((max_price - min_price) / min_price) * 100
                else:
                    optimal_pnl[asset] = 0.0
                
                logger.debug(f"Chunk {self.current_chunk_index} - {asset} - "
                           f"Optimal PnL: {optimal_pnl[asset]:.2f}% "
                           f"(Price range: {min_price:.8f} - {max_price:.8f})")
                
            except Exception as e:
                logger.error(f"Error calculating PnL for {asset} in chunk {self.current_chunk_index}: {e}")
                optimal_pnl[asset] = 0.0
        
        return optimal_pnl
    
    def __iter__(self):
        """Make the loader iterable over chunks."""
        for i in range(self.total_chunks):
            yield self.load_chunk(i)
    
    def __len__(self) -> int:
        """Return the total number of chunks."""
        return self.total_chunks
